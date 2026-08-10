"""QCMet benchmarks gate execution quality metrics module."""

from .cliffordrb import CliffordRB
from .cycle_benchmarking import CycleBenchmarking
from .interleaved_rb import InterleavedRB
from .over_under_rotation_angle import OverUnderRotationAngle

__all__ = ["CliffordRB", "CycleBenchmarking", "OverUnderRotationAngle", "InterleavedRB", "GST"]


def __getattr__(name):
	if name == "GST":
		from .gate_set_tomography import GST

		return GST
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
