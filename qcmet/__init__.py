"""QCMet module.

QCMet provides a collection of metrics and benchmarks for quantum
computers.
"""

from .benchmarks import (
    QFT,
    T1,
    T2,
    BenchmarkCollection,
    CliffordRB,
    CycleBenchmarking,
    HamiltonianSimulation,
    IdleQubitOscillationFrequency,
    InterleavedRB,
    MirroredCircuits,
    OverUnderRotationAngle,
    QScoreSingleInstance,
    QuantumVolumeFixedQubits,
    Simulation1DFermiHubbard,
    UpperBoundOnVD,
)
from .core import FileManager
from .devices import (
    AerSimulator,
    IdealSimulator,
    NoisySimulator,
    QiskitDevice,
)

__all__ = [
    "FileManager",
    "QuantumVolumeFixedQubits",
    "DummySimulator",
    "QFT",
    "GST",
    "HamiltonianSimulation",
    "Simulation1DFermiHubbard",
    "VQE",
    "VQE1DFermiHubbard",
    "QScoreSingleInstance",
    "CliffordRB",
    "CycleBenchmarking",
    "OverUnderRotationAngle",
    "FileManager",
    "QiskitDevice",
    "NoisySimulator",
    "IdealSimulator",
    "AerSimulator",
    "T1",
    "T2",
    "InterleavedRB",
    "MirroredCircuits",
    "IdleQubitOscillationFrequency",
    "UpperBoundOnVD",
    "BenchmarkCollection",
]


def __getattr__(name):
    if name == "GST":
        from .benchmarks import GST

        return GST
    if name in {"VQE", "VQE1DFermiHubbard"}:
        from .benchmarks import VQE, VQE1DFermiHubbard

        return VQE if name == "VQE" else VQE1DFermiHubbard
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__version__ = "1.0.0"
