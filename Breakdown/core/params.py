"""
Parameter model for a breakdown campaign: dataclasses, JSON persistence, validation.

Nothing else in this repo saves or reloads its settings -- every existing GUI is
retyped from scratch at each launch. A stress campaign only produces comparable
results if every device sees identical parameters, so save/load is a hard
requirement here rather than a convenience.

Persistence is deliberately forgiving. A file written by a future version must
still load: unknown keys become warnings, missing keys fall back to defaults.
Refusing to open a parameter file mid-campaign is worse than opening it with a
note about what was ignored.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Tuple

from . import _paths  # noqa: F401  (puts KeithleyGUI on sys.path)
from KeithleyGUI.ui_helpers import write_json_file

#: Bumped whenever the on-disk layout changes incompatibly.
SCHEMA_VERSION = 1

RUN_MODES = ("rvs_only", "cvs_only", "alternating", "manual")
STATISTICS = ("median", "mean")

#: MFIA DC bias ceilings. The 4-terminal configuration is limited to +/-3 V;
#: 2-terminal allows +/-10 V. Mirrors the guard at cv_gate_map_ui.py:623-633.
MFIA_BIAS_LIMIT_4T_V = 3.0
MFIA_BIAS_LIMIT_2T_V = 10.0


@dataclass
class SampleParams:
    sample_name: str = "sample"
    #: Crosspoint edge length in micrometres. Devices are square, so this one
    #: number fixes the area.
    crosspoint_um: float = 5.0
    #: Dielectric thickness. Optional -- when present the summary can report a
    #: breakdown field E_BD alongside the breakdown voltage.
    thickness_nm: Optional[float] = None
    operator: str = ""
    notes: str = ""

    @property
    def area_um2(self) -> float:
        return float(self.crosspoint_um) ** 2


@dataclass
class MfiaParams:
    host: str = "192.168.121.162"
    port: int = 8004
    #: Blank means "discover the first MFIA on the network".
    device_id: str = ""
    imps: int = 0
    #: Impedance model index. 0 = Rp || Cp, the usual choice for a leaky capacitor.
    model: int = 0
    #: 0 High Speed .. 3 Very High Accuracy. This is the only averaging control
    #: the impedance analyser exposes.
    quality: int = 2
    auto_bw: bool = True
    #: 0 Manual, 1 Auto, 2 Zone.
    inputrange_mode: int = 1
    manual_current_range: float = 1e-6
    demod_order: int = 4
    demod_timeconstant: float = 0.01
    demod_rate: float = 200.0
    demod_sinc: bool = False
    #: DC bias is always stepped, never jumped.
    ramp_step: float = 0.05
    ramp_wait: float = 0.01
    #: 4-terminal wiring restricts the DC bias to +/-3 V.
    four_terminal: bool = True

    @property
    def bias_limit_V(self) -> float:
        return MFIA_BIAS_LIMIT_4T_V if self.four_terminal else MFIA_BIAS_LIMIT_2T_V


@dataclass
class CFParams:
    """C(f) at fixed bias. Always log-spaced, always f_min -> f_max -> f_min."""

    f_min: float = 100.0
    f_max: float = 1.0e6
    #: Points in one direction; the sweep measures roughly twice this many.
    points: int = 51
    bias_V: float = 0.0
    amplitude_V: float = 0.05
    settle_s: float = 0.1


@dataclass
class CVParams:
    """C(V) at fixed frequency. Always linear, always 0 -> v_max -> v_min -> 0."""

    v_min: float = -2.0
    v_max: float = 2.0
    #: Points per segment of the three-segment hysteresis loop.
    points: int = 51
    frequency_Hz: float = 1000.0
    amplitude_V: float = 0.05
    settle_s: float = 0.1


@dataclass
class RVSParams:
    """Ramped voltage stress: ramp until breakdown or v_max."""

    #: +1 or -1. One polarity per campaign.
    polarity: int = 1
    v_start: float = 0.0
    #: Magnitude of the ramp ceiling; polarity is applied separately.
    v_max: float = 10.0
    #: Requested ramp rate. The achieved rate is measured and recorded, because
    #: V_BD is ramp-rate dependent and the request is not always achievable.
    ramp_rate_Vps: float = 1.0
    #: Ceiling on the voltage increment, so a fast request cannot turn the ramp
    #: into a handful of coarse jumps.
    max_step_V: float = 0.05
    nplc: float = 1.0
    #: Hardware protection limit.
    compliance_A: float = 1e-3
    #: Breakdown declared here. Must sit below compliance -- see validate().
    i_bd_A: float = 1e-4
    source_delay_s: float = 0.0
    current_autorange: bool = False


@dataclass
class CVSParams:
    """Constant voltage stress: hold until breakdown or a time limit.

    The stress level is not stored here. It is resolved per device at runtime
    from the accumulated V_BD statistics -- see :mod:`Breakdown.core.advisor`.
    """

    #: Approach rate to the stress level. Fast on purpose: charge injected on the
    #: way up must be negligible next to the hold, or t_BD is contaminated.
    pre_ramp_rate_Vps: float = 100.0
    sample_interval_s: float = 0.05
    max_duration_s: float = 3600.0
    nplc: float = 1.0
    compliance_A: float = 1e-3
    i_bd_A: float = 1e-4
    current_autorange: bool = False


@dataclass
class AdvisorParams:
    """How the recommended CVS stress level is computed from past RVS results."""

    #: V_CVS = k * statistic(V_BD). 0.80-0.92 puts t_BD in a practical window.
    k_fraction: float = 0.85
    statistic: str = "median"


@dataclass
class RunParams:
    mode: str = "alternating"
    output_dir: str = ""
    enable_cf: bool = True
    enable_cv: bool = True
    #: Absolute ceiling on any sourced voltage, independent of the ramp settings.
    abs_max_voltage_V: float = 20.0


@dataclass
class BreakdownParams:
    sample: SampleParams = field(default_factory=SampleParams)
    mfia: MfiaParams = field(default_factory=MfiaParams)
    cf: CFParams = field(default_factory=CFParams)
    cv: CVParams = field(default_factory=CVParams)
    rvs: RVSParams = field(default_factory=RVSParams)
    cvs: CVSParams = field(default_factory=CVSParams)
    advisor: AdvisorParams = field(default_factory=AdvisorParams)
    run: RunParams = field(default_factory=RunParams)


@dataclass
class LoadResult:
    params: BreakdownParams
    warnings: List[str]


# --------------------------------------------------------------------------
# serialisation
# --------------------------------------------------------------------------

#: Section names, in the order they appear on disk.
_SECTIONS: Tuple[str, ...] = tuple(f.name for f in fields(BreakdownParams))


def to_dict(params: BreakdownParams) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"schema_version": SCHEMA_VERSION}
    for name in _SECTIONS:
        section = getattr(params, name)
        payload[name] = {f.name: getattr(section, f.name) for f in fields(section)}
    return payload


def _coerce(value: Any, template: Any) -> Any:
    """Coerce a JSON value to the type of the corresponding default."""
    if template is None or value is None:
        return value
    if isinstance(template, bool):
        return bool(value)
    if isinstance(template, int) and not isinstance(template, bool):
        return int(value)
    if isinstance(template, float):
        return float(value)
    if isinstance(template, str):
        return str(value)
    return value


def from_dict(payload: Dict[str, Any]) -> LoadResult:
    warnings: List[str] = []
    params = BreakdownParams()

    version = payload.get("schema_version")
    if version is None:
        warnings.append("File has no schema_version; assuming version 1.")
    elif int(version) > SCHEMA_VERSION:
        warnings.append(
            f"File schema_version {version} is newer than this program's "
            f"{SCHEMA_VERSION}; unknown settings were ignored."
        )

    for key in payload:
        if key != "schema_version" and key not in _SECTIONS:
            warnings.append(f"Ignored unknown section '{key}'.")

    for name in _SECTIONS:
        raw = payload.get(name)
        if raw is None:
            continue
        if not isinstance(raw, dict):
            warnings.append(f"Section '{name}' is not an object; used defaults.")
            continue
        section = getattr(params, name)
        known = {f.name for f in fields(section)}
        for key, value in raw.items():
            if key not in known:
                warnings.append(f"Ignored unknown key '{name}.{key}'.")
                continue
            try:
                setattr(section, key, _coerce(value, getattr(section, key)))
            except (TypeError, ValueError):
                warnings.append(
                    f"Could not read '{name}.{key}' (value {value!r}); used default."
                )

    return LoadResult(params=params, warnings=warnings)


def save_params(path: str, params: BreakdownParams) -> str:
    return write_json_file(path, to_dict(params), label="parameters")


def load_params(path: str) -> LoadResult:
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError(f"Parameter file is not a JSON object: {path}")
    return from_dict(payload)


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------


def validate(params: BreakdownParams) -> List[str]:
    """Return a list of blocking problems. Empty means safe to start.

    These are hard errors only -- anything here would either damage a device,
    make a measurement impossible, or silently produce meaningless data.
    """
    errors: List[str] = []
    errors += _validate_sample(params)
    errors += _validate_stress(params)
    errors += _validate_mfia_sweeps(params)
    errors += _validate_run(params)
    return errors


def _validate_sample(p: BreakdownParams) -> List[str]:
    errors = []
    if p.sample.crosspoint_um <= 0:
        errors.append("Sample: crosspoint_um must be greater than 0.")
    if p.sample.thickness_nm is not None and p.sample.thickness_nm <= 0:
        errors.append("Sample: thickness_nm must be greater than 0 when given.")
    return errors


def _validate_stress(p: BreakdownParams) -> List[str]:
    errors = []
    ceiling = abs(p.run.abs_max_voltage_V)

    if p.rvs.polarity not in (1, -1):
        errors.append("RVS: polarity must be +1 or -1.")
    if abs(p.rvs.v_max) > ceiling:
        errors.append(
            f"RVS: v_max {p.rvs.v_max} V exceeds the absolute ceiling of {ceiling} V."
        )
    if abs(p.rvs.v_max) <= abs(p.rvs.v_start):
        errors.append("RVS: v_max must be further from zero than v_start.")
    if p.rvs.ramp_rate_Vps <= 0:
        errors.append("RVS: ramp_rate_Vps must be greater than 0.")
    if p.rvs.max_step_V <= 0:
        errors.append("RVS: max_step_V must be greater than 0.")
    if p.rvs.nplc <= 0:
        errors.append("RVS: nplc must be greater than 0.")

    # The compliance limit clamps the current, so a breakdown threshold at or
    # above it can never be reached -- the run would ramp to v_max every time
    # and silently report "no breakdown".
    if p.rvs.i_bd_A >= p.rvs.compliance_A:
        errors.append(
            f"RVS: i_bd_A ({p.rvs.i_bd_A:g} A) must be below compliance_A "
            f"({p.rvs.compliance_A:g} A), otherwise breakdown can never trigger."
        )
    if p.rvs.i_bd_A <= 0:
        errors.append("RVS: i_bd_A must be greater than 0.")

    if p.cvs.i_bd_A >= p.cvs.compliance_A:
        errors.append(
            f"CVS: i_bd_A ({p.cvs.i_bd_A:g} A) must be below compliance_A "
            f"({p.cvs.compliance_A:g} A), otherwise breakdown can never trigger."
        )
    if p.cvs.i_bd_A <= 0:
        errors.append("CVS: i_bd_A must be greater than 0.")
    if p.cvs.sample_interval_s <= 0:
        errors.append("CVS: sample_interval_s must be greater than 0.")
    if p.cvs.max_duration_s <= 0:
        errors.append("CVS: max_duration_s must be greater than 0.")
    if p.cvs.pre_ramp_rate_Vps <= 0:
        errors.append("CVS: pre_ramp_rate_Vps must be greater than 0.")
    if p.cvs.nplc <= 0:
        errors.append("CVS: nplc must be greater than 0.")

    return errors


def _validate_mfia_sweeps(p: BreakdownParams) -> List[str]:
    errors = []
    limit = p.mfia.bias_limit_V
    wiring = "4-terminal" if p.mfia.four_terminal else "2-terminal"

    if p.cf.f_min <= 0:
        errors.append("C(F): f_min must be greater than 0 for a log sweep.")
    if p.cf.f_max <= 0:
        errors.append("C(F): f_max must be greater than 0 for a log sweep.")
    if p.cf.f_min >= p.cf.f_max:
        errors.append("C(F): f_min must be less than f_max.")
    if p.cf.points < 2:
        errors.append("C(F): points must be at least 2.")
    if p.cf.settle_s < 0:
        errors.append("C(F): settle_s cannot be negative.")
    if abs(p.cf.bias_V) > limit:
        errors.append(
            f"C(F): bias {p.cf.bias_V} V exceeds the MFIA {wiring} bias "
            f"limit of {limit:g} V."
        )

    if p.cv.points < 2:
        errors.append("C(V): points must be at least 2.")
    if p.cv.v_min >= p.cv.v_max:
        errors.append("C(V): v_min must be less than v_max.")
    if p.cv.settle_s < 0:
        errors.append("C(V): settle_s cannot be negative.")
    if p.cv.frequency_Hz <= 0:
        errors.append("C(V): frequency_Hz must be greater than 0.")
    for label, value in (("v_max", p.cv.v_max), ("v_min", p.cv.v_min)):
        if abs(value) > limit:
            errors.append(
                f"C(V): bias {label} {value} V exceeds the MFIA {wiring} bias "
                f"limit of {limit:g} V."
            )

    for name, amplitude in (("C(F)", p.cf.amplitude_V), ("C(V)", p.cv.amplitude_V)):
        if amplitude <= 0:
            errors.append(f"{name}: amplitude_V must be greater than 0.")

    return errors


def _validate_run(p: BreakdownParams) -> List[str]:
    errors = []
    if p.run.mode not in RUN_MODES:
        errors.append(f"Run: mode must be one of {', '.join(RUN_MODES)}.")
    if p.run.abs_max_voltage_V <= 0:
        errors.append("Run: abs_max_voltage_V must be greater than 0.")
    if p.advisor.statistic not in STATISTICS:
        errors.append(f"Advisor: statistic must be one of {', '.join(STATISTICS)}.")
    if not 0 < p.advisor.k_fraction <= 1.0:
        errors.append("Advisor: k_fraction must be in (0, 1].")
    return errors
