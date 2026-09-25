"""
Impedance model arithmetic: turning the MFIA's two raw parameters into C and R.

The instrument reports ``param0``/``param1`` whose meaning depends on the
selected model -- for ``Rp || Cp`` they are a resistance and a capacitance, for
``G + B`` they are a conductance and a susceptance that still need converting.
Getting this mapping wrong produces plausible-looking numbers that are off by
``2*pi*f``, so it lives in one tested place.

Ported from ``cv_cf_recipe_ui.py:47-59`` and ``:405-428``, kept free of
instrument imports so it is testable without ``zhinst``.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

NAN = float("nan")

#: model index -> (display name, param0 name, param1 name)
MODEL_INFO: Dict[int, Tuple[str, str, str]] = {
    0: ("Rp || Cp", "Rp", "Cp"),
    1: ("Rs + Cs", "Rs", "Cs"),
    2: ("Rs + Ls", "Rs", "Ls"),
    3: ("G + B", "G", "B"),
    4: ("D + Cs", "D", "Cs"),
    5: ("Q + Cs", "Q", "Cs"),
    6: ("D + Ls", "D", "Ls"),
    7: ("Q + Ls", "Q", "Ls"),
    8: ("Rp || Lp", "Rp", "Lp"),
    9: ("D + Cp", "D", "Cp"),
    10: ("Dielectric", "P0", "P1"),
}

MODEL_TAGS: Dict[int, str] = {
    0: "RpCp",
    1: "RsCs",
    2: "RsLs",
    3: "GB",
    4: "DCs",
    5: "QCs",
    6: "DLs",
    7: "QLs",
    8: "RpLp",
    9: "DCp",
    10: "Dielectric",
}

QUALITY_LABELS = {
    0: "High Speed",
    1: "Medium",
    2: "High Accuracy",
    3: "Very High Accuracy",
}

#: Models whose param1 is already a capacitance.
_CAPACITANCE_FROM_PARAM1 = (0, 1, 4, 5, 9)
#: Models whose param0 is already a resistance.
_RESISTANCE_FROM_PARAM0 = (0, 1, 2, 8)
#: G + B: param0 is a conductance, param1 a susceptance.
_CONDUCTANCE_SUSCEPTANCE = 3


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return NAN


def compute_model_values(
    model: int, param0: Any, param1: Any, freq_hz: Optional[float]
) -> Tuple[float, float]:
    """Return ``(capacitance_F, resistance_Ohm)`` for the selected model.

    Either may be NaN when the model does not express that quantity -- an
    inductive model has no capacitance, and G+B needs a frequency to produce one.
    """
    p0 = _to_float(param0)
    p1 = _to_float(param1)
    cap = NAN
    res = NAN

    if model in _CAPACITANCE_FROM_PARAM1:
        cap = p1
    elif model == _CONDUCTANCE_SUSCEPTANCE:
        freq = _to_float(freq_hz)
        if math.isfinite(freq) and freq > 0 and math.isfinite(p1):
            cap = p1 / (2.0 * math.pi * freq)

    if model in _RESISTANCE_FROM_PARAM0:
        res = p0
    elif model == _CONDUCTANCE_SUSCEPTANCE:
        res = (1.0 / p0) if (math.isfinite(p0) and p0 != 0) else NAN

    return cap, res


def decompose_impedance(z: Any) -> Dict[str, float]:
    """Split a complex impedance into real, imaginary, magnitude and phase."""
    try:
        value = complex(z)
    except (TypeError, ValueError):
        return {
            "Z_real_Ohm": NAN,
            "Z_imag_Ohm": NAN,
            "Z_abs_Ohm": NAN,
            "Z_phase_rad": NAN,
        }
    return {
        "Z_real_Ohm": value.real,
        "Z_imag_Ohm": value.imag,
        "Z_abs_Ohm": abs(value),
        "Z_phase_rad": math.atan2(value.imag, value.real),
    }


def per_area(value: float, area_um2: float) -> float:
    """Normalise a quantity by the crosspoint area, or NaN if that is impossible."""
    magnitude = _to_float(value)
    area = _to_float(area_um2)
    if not math.isfinite(magnitude) or not math.isfinite(area) or area <= 0:
        return NAN
    return magnitude / area
