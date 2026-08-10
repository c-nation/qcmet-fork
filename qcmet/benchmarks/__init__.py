"""QCMet benchmarks module.

This module contains all benchmarks implemented in QCMet.
"""

from .base_benchmark import BaseBenchmark
from .benchmark_collection import BenchmarkCollection
from .circuit_execution_quality_metrics import (
    MirroredCircuits,
    QuantumVolumeFixedQubits,
    UpperBoundOnVD,
)
from .gate_execution_quality_metrics import (
    CliffordRB,
    CycleBenchmarking,
    InterleavedRB,
    OverUnderRotationAngle,
)
from .qubit_quality_metrics import T1, T2, IdleQubitOscillationFrequency
from .well_studied_task_execution_quality_metrics import (
    QFT,
    HamiltonianSimulation,
    QScoreSingleInstance,
    Simulation1DFermiHubbard,
)

__all__ = [
    "BaseBenchmark",
    "CliffordRB",
    "CycleBenchmarking",
    "InterleavedRB",
    "OverUnderRotationAngle",
    "QFT",
    "GST",
    "HamiltonianSimulation",
    "Simulation1DFermiHubbard",
    "VQE",
    "MirroredCircuits",
    "VQE1DFermiHubbard",
    "QScoreSingleInstance",
    "QuantumVolumeFixedQubits",
    "T1",
    "IdleQubitOscillationFrequency",
    "T2",
    "UpperBoundOnVD",
    "BenchmarkCollection",
]


def __getattr__(name):
    if name == "GST":
        from .gate_execution_quality_metrics import GST

        return GST
    if name in {"VQE", "VQE1DFermiHubbard"}:
        from .well_studied_task_execution_quality_metrics import VQE, VQE1DFermiHubbard

        return VQE if name == "VQE" else VQE1DFermiHubbard
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
