"""QCMet benchmarks well studied task execution quality metrics module."""

from .hamiltonian_simulation import HamiltonianSimulation
from .qft import QFT
from .qscore import QScoreSingleInstance
from .simulation_1d_fermi_hubbard import Simulation1DFermiHubbard

__all__ = [
    "QFT",
    "HamiltonianSimulation",
    "Simulation1DFermiHubbard",
    "VQE",
    "VQE1DFermiHubbard",
    "QScoreSingleInstance",
]


def __getattr__(name):
    if name == "VQE":
        from .vqe import VQE

        return VQE
    if name == "VQE1DFermiHubbard":
        from .vqe_1d_fermi_hubbard import VQE1DFermiHubbard

        return VQE1DFermiHubbard
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
