"""
The UI-agnostic seam between the run engine and whatever is driving it.

:mod:`Breakdown.core.session` needs three things a headless module cannot
provide: it must *ask the operator questions* (swap the cables; confirm this
stress voltage) and it must *report progress*. Both go through the interfaces
here, so the engine never imports Qt and a web front-end could drive the same
code.

The prompts genuinely block. A cable swap is a physical act; there is no
sensible way to continue without it. The Qt implementation emits a signal and
waits on a :class:`threading.Event` that the dialog's handler sets -- see
``Breakdown/app_qt/wizard.py``. Any implementation must also honour the session's
stop request while waiting, or Stop would do nothing while a dialog is open.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Protocol, runtime_checkable

from .advisor import Recommendation, VbdRecord


class StressType(str, Enum):
    RVS = "RVS"
    CVS = "CVS"


class Instrument(str, Enum):
    MFIA = "MFIA"
    SMU = "Keithley 2400"


class Phase(str, Enum):
    CF = "CF"
    CV = "CV"
    RVS = "RVS"
    CVS = "CVS"


@dataclass(frozen=True)
class CvsVoltageContext:
    """Everything the operator needs to choose a stress level."""

    device_index: int
    crosspoint_um: float
    #: What was used on the previous CVS device, if any. Offered as the default
    #: so a campaign can hold one level steady without retyping it.
    previous_voltage: Optional[float]
    #: Statistics-derived suggestion, with its reasoning and caveats.
    recommendation: Recommendation
    #: The RVS results the recommendation was computed from, for display.
    records: List[VbdRecord] = field(default_factory=list)


@runtime_checkable
class Prompter(Protocol):
    """Blocking questions the session asks the operator."""

    def confirm_cable_swap(self, target: Instrument, device_index: int) -> bool:
        """Wait until the probes are wired to ``target``. False aborts the run."""

    def resolve_cvs_voltage(self, ctx: CvsVoltageContext) -> Optional[float]:
        """Return the stress voltage magnitude to use. None aborts the run."""

    def choose_stress_type(self, device_index: int) -> Optional[StressType]:
        """Only called in manual mode. None aborts the run."""


class SessionListener:
    """Progress callbacks. Subclass and override what you need.

    Concrete no-op base rather than a Protocol so a caller that only cares about
    logging does not have to stub out eight methods.
    """

    def on_log(self, message: str) -> None:
        pass

    def on_device_started(self, device_index: int, crosspoint_um: float) -> None:
        pass

    def on_phase_started(self, phase: Phase, device_index: int, total_points: Optional[int]) -> None:
        pass

    def on_point(self, phase: Phase, row: dict) -> None:
        pass

    def on_phase_finished(self, phase: Phase, device_index: int, result: object) -> None:
        pass

    def on_device_finished(self, device_index: int, summary_row: dict) -> None:
        pass

    def on_run_finished(self, devices_completed: int, reason: str) -> None:
        pass
