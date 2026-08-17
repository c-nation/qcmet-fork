"""Linear Cross-Entropy Benchmark.

TODO:
Add support for different types of entangling gates (e.g., CNOT, CZ, etc.)
and single qubit gates (e.g., Google's XEB gates, etc.).

"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, List

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Statevector

from qcmet.benchmarks import BaseBenchmark

if TYPE_CHECKING:
    from qcmet.core import FileManager


class LinearXEB(BaseBenchmark):
    """Base benchmark implementation for linear XEB circuits."""

    def __init__(
        self,
        qubits: int | List[int],
        depth: int | List[int],
        num_circuits: int,
        seed: int | None = None,
        save_path: str | Path | FileManager | None = None,
    ):
        """Initialize the linear XEB benchmark configuration."""
        super().__init__("LinearXEB", qubits, save_path)
        self.config["depth"] = depth
        self.config["num_circuits"] = num_circuits

        self.config["entangling_gates"] = "cz" # Type of entangling gates to use (e.g., CNOT, CZ)
        self.config["single_qubit_gates"] = "haar_su2" # Type of single qubit gates to use (e.g., Haar, Google's XEB gates, ...)
        self.config["seed"] = seed

        self.rng = np.random.default_rng(seed)

    def _haar_su2(self) -> np.ndarray:
        """Return a Haar-random SU(2) matrix."""
        a, b, c, d = self.rng.normal(size=4)
        a, b, c, d = np.array([a, b, c, d]) / np.sqrt(a**2 + b**2 + c**2 + d**2)  # Normalize to unit length
        U = np.array([[a + 1j * b, c + 1j * d], [-c + 1j * d, a - 1j * b]])  # Construct the SU(2) matrix
        return U

    def _add_haar_su2_layer(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Append a layer of Haar-random single-qubit SU(2) gates."""
        for qubit in range(circuit.num_qubits):
            U = self._haar_su2()
            Ugate = UnitaryGate(U, label="Haar SU(2)")
            circuit.append(Ugate, [qubit])
        return circuit

    def _add_entangling_layer(self, circuit: QuantumCircuit):
        """Append a layer of entangling gates to the given quantum circuit.
        Currently supports only CZ gates in a linear nearest-neighbor configuration.
        Alternates between even and odd pairs of qubits for each layer to ensure connectivity."""
        if self.config["entangling_gates"] == "cz":
            # Add a layer of disjoint CZ gates to the circuit
            n_layers = circuit.depth()
            layer_parity = int(((n_layers - 1) / 2) % 2)
            n_qubits = circuit.num_qubits
            for i in range(0, n_qubits - 1, 2):
                if layer_parity == 0:
                    circuit.cz(i, i + 1)  # (0, 1), (2, 3), ...
                else:
                    circuit.cz(i + 1, i + 2 if i + 2 < n_qubits else 0)  # (1, 2), (3, 4), ... (n, 0) if n is even, else (1, 2), (3, 4), ..., (n-2, n-1)
        else:
            raise ValueError(f"Unsupported entangling gate type: {self.config['entangling_gates']}")

    def build_circuit(self, num_qubits: int, depth: int) -> QuantumCircuit:
        """Build a quantum circuit with alternating random and entangling layers."""
        circuit = QuantumCircuit(num_qubits)

        for _ in range(depth):
            self._add_haar_su2_layer(circuit)
            self._add_entangling_layer(circuit)
        return circuit

    def _generate_circuits(self) -> List[QuantumCircuit]:
        """Generate random circuits for the configured qubits and depths."""
        circuits = []
        depths = self.config["depth"]
        if isinstance(depths, int):
            depths = [depths]

        for depth in depths:
            for _ in range(self.config["num_circuits"]):
                circuit = self.build_circuit(self.num_qubits, depth)
                circuit.measure_all()
                circuits.append(circuit)

        return circuits

    def _ideal_probabilities(self, circuit: QuantumCircuit, observed_bitstrings: List[str]) -> dict[str, float]:
        """Compute the ideal output probability for a given bitstring and quantum circuit.

        Uses qiskit's Statevector simulation to compute the ideal output state of the circuit, and then calculates the probability of measuring the specified bitstring.

        Args:
            circuit (QuantumCircuit): The quantum circuit for which to compute the ideal probabilities.
            observed_bitstrings (List[str]): The list of bitstrings for which to compute ideal probabilities.

        Raises:
            QiskitError: If the circuit is not unitary or if the bitstring length does not match the number of qubits in the circuit.

        Returns:
            dict[str, float]: A dictionary mapping bitstrings to their ideal probabilities."""

        c = circuit.remove_final_measurements(inplace=False)
        if c is not None:
            circuit = c

        statevector = Statevector.from_instruction(circuit)
        return {
            bitstring: float(abs(statevector[int(bitstring[::-1], 2)]) ** 2)
            for bitstring in observed_bitstrings
        }

    def _cross_entropy_fidelity(self, circuit: QuantumCircuit, counts: dict[str, int]) -> float:
        """Calculate the fidelity of a given circuit based on the observed counts and the ideal probabilities.

        Args:
            circuit (QuantumCircuit): The quantum circuit for which to calculate fidelity.
            counts (dict[str, int]): A dictionary of observed counts from executing the circuit.

        Returns:
            float: The calculated fidelity value."""
        probabilities = self._ideal_probabilities(circuit, list(counts.keys()))
        shots = sum(counts.values())

        fidelity = sum(
            (count / shots) * probabilities.get(bitstring, 0.0)
            for bitstring, count in counts.items()
        )

        # Linear-XEB normalization:
        fidelity = (2**circuit.num_qubits) * fidelity - 1
        return fidelity

    def _analyze(self) -> dict[str, Any]:
        """Analyze circuit measurements and return fidelity statistics."""
        fidelities = []

        for circuit, counts in zip(
            self.experiment_data["circuit"],
            self.experiment_data["circuit_measurements"],
            strict=True,
        ):
            fidelity = self._cross_entropy_fidelity(circuit, counts)
            fidelities.append(fidelity)

        self.experiment_data["linear_xeb_fidelity"] = fidelities

        return {
            "mean_fidelity": float(np.mean(fidelities)),
            "fidelities": fidelities,
        }
