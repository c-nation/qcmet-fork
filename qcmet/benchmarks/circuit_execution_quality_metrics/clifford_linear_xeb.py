"""
Here we implement the linear cross-entropy banchmarking with Clifford circuits following PRA 108, 052613

We have implemented a 'cycle' as that pf the of the 1D chain (Fig 1a). This is controlled by the depth parameter. 
A cycle thus consists of 4 layers: single qubit clifford layer, entangling layer, single qubit clifford layer, entangling layer.
The entangling layers are brickwork overlapped.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from qcmet.core import FileManager

import numpy as np
from qiskit import QiskitError, QuantumCircuit
from qiskit.quantum_info import Clifford, Pauli, StabilizerState, random_clifford
from qiskit.synthesis import synth_clifford_full

from qcmet.benchmarks.circuit_execution_quality_metrics.linear_xeb import LinearXEB


class CliffordLinearXEB(LinearXEB):
    def __init__(self, 
                 qubits: int | List[int],
                 depth: int | List[int],
                 num_circuits: int,  # number of distinct random circuits to generate for each depth
                 seed: int | None = None,
                 save_path: str | Path | FileManager | None = None):
        super().__init__(
                         qubits=qubits,
                         depth=depth,
                         num_circuits=num_circuits,
                         seed=seed,
                         save_path=save_path,
                         )
        self.name = "CliffordLinearXEB"

    def _random_single_qubit_clifford_layer(self) -> QuantumCircuit:
        """Generate a random single-qubit Clifford layer Q = \\otimes_i=1^n q_i for random 1 qubit Clifford q_i on qubit i."""
        circuit = QuantumCircuit(self.num_qubits)
        for i in range(self.num_qubits):
            circuit.compose(
                random_clifford(1, seed=self.rng).to_circuit(),
                qubits=[i],
                inplace=True,
            )
        return circuit

    def _cnot_layer(self) -> QuantumCircuit:
        """Add a layer of entangling gates to the circuit. Currently supports only CX gates in a linear nearest-neighbor configuration."""
        circuit = QuantumCircuit(self.num_qubits)

        n_layers = circuit.depth()
        layer_parity = int(((n_layers - 1) / 4) % 2)
        
        for i in range(0, self.num_qubits - 1, 2):
            if layer_parity == 0:
                circuit.cx(i, i + 1)  # (0, 1), (2, 3), ...
            else:
                circuit.cx(i + 1, i + 2 if i + 2 < self.num_qubits else 0) 
        return circuit

    def build_circuit(self, num_qubits: int, depth: int) -> QuantumCircuit:
        """Build a random Clifford circuit of specified depth and number of qubits."""
        circuit = QuantumCircuit(num_qubits)
        for _ in range(depth):
            circuit.compose(self._random_single_qubit_clifford_layer(), inplace=True)
            circuit.compose(self._cnot_layer(), inplace=True)
            circuit.compose(self._random_single_qubit_clifford_layer(), inplace=True)
            circuit.compose(self._cnot_layer(), inplace=True)
        return circuit

    def _generate_circuits(self) -> List[QuantumCircuit]:
        """
        Generates a list of random circuits for the given number of qubits and depth.
        """
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

    @staticmethod
    def _gf2_nullspace(matrix: np.ndarray) -> list[np.ndarray]:
        """Return a basis for the nullspace of a binary matrix."""
        matrix = np.asarray(matrix, dtype=np.uint8).copy() % 2
        n_rows, n_columns = matrix.shape
        pivot_columns = []
        pivot_row = 0

        for column in range(n_columns):
            candidates = np.flatnonzero(matrix[pivot_row:, column])
            if len(candidates) == 0:
                continue

            row = pivot_row + int(candidates[0])
            matrix[[pivot_row, row]] = matrix[[row, pivot_row]]
            for other_row in range(n_rows):
                if other_row != pivot_row and matrix[other_row, column]:
                    matrix[other_row] ^= matrix[pivot_row]

            pivot_columns.append(column)
            pivot_row += 1
            if pivot_row == n_rows:
                break

        free_columns = [
            column for column in range(n_columns) if column not in pivot_columns
        ]
        basis = []
        for free_column in free_columns:
            vector = np.zeros(n_columns, dtype=np.uint8)
            vector[free_column] = 1
            for row, pivot_column in enumerate(pivot_columns):
                vector[pivot_column] = matrix[row, free_column]
            basis.append(vector)
        return basis

    @staticmethod
    def probability_of_bitstring(stabilizer_state: StabilizerState, bitstring: str) -> float:
        """Return one computational-basis probability from a stabilizer tableau.
        
        This is done by first
        """
        num_qubits = stabilizer_state.num_qubits
        assert num_qubits is not None
        
        if len(bitstring) != num_qubits or set(bitstring) - {"0", "1"}:
            raise ValueError("bitstring must contain one binary digit per qubit")

        clifford = stabilizer_state.clifford
        stabilizer_x = np.asarray(clifford.stab_x, dtype=np.uint8)
        stabilizer_z = np.asarray(clifford.stab_z, dtype=np.uint8)
        stabilizer_labels = clifford.to_labels()[num_qubits:]

        # A product of stabilizers is diagonal in the computational basis iff
        # its combined X support vanishes.
        z_only_basis = CliffordLinearXEB._gf2_nullspace(stabilizer_x.T)
        raw_bits = np.asarray([int(bit) for bit in bitstring], dtype=np.uint8)

        for combination in z_only_basis:
            product = Pauli("I" * num_qubits)
            for index, selected in enumerate(combination):
                if selected:
                    product = product.compose(Pauli(stabilizer_labels[index]))

            label = product.to_label()
            sign = -1 if label.startswith("-") else 1
            z_mask = stabilizer_z.T @ combination % 2
            eigenvalue = sign * (-1) ** int(np.dot(z_mask, raw_bits) % 2)
            if eigenvalue != 1:
                return 0.0

        num_constraints = len(z_only_basis)
        return 2.0 ** (-(num_qubits - num_constraints))

    def _ideal_probabilities(self,
                             circuit: QuantumCircuit,
                             observed_bitstrings: List[str],
                             ) -> dict[str, float]:
        
        circuit_without_measurements = circuit.remove_final_measurements(
            inplace=False
        )
        assert circuit_without_measurements is not None
        stabilizer_state = StabilizerState(circuit_without_measurements)

        return {
            bitstring: self.probability_of_bitstring(
                stabilizer_state,
                bitstring,
            )
            for bitstring in observed_bitstrings
        }

    def _ideal_collision_probability(self, circuit: QuantumCircuit) -> float:
        """Return the ideal sum of squared computational-basis probabilities."""
        circuit_without_measurements = circuit.remove_final_measurements(
            inplace=False
        )
        assert circuit_without_measurements is not None
        stabilizer_state = StabilizerState(circuit_without_measurements)
        stabilizer_x = np.asarray(stabilizer_state.clifford.stab_x, dtype=np.uint8)
        num_constraints = len(self._gf2_nullspace(stabilizer_x.T))
        return 2.0 ** (-(self.num_qubits - num_constraints))

    def _cross_entropy_fidelity(
        self,
        circuit: QuantumCircuit,
        counts: dict[str, int],
    ) -> float:
        """Calculate XEB normalized to the ideal Clifford distribution."""
        probabilities = self._ideal_probabilities(circuit, list(counts.keys()))
        shots = sum(counts.values())
        cross_entropy = sum(
            (count / shots) * probabilities.get(bitstring, 0.0)
            for bitstring, count in counts.items()
        )

        uniform_probability = 2.0 ** (-circuit.num_qubits)
        ideal_collision = self._ideal_collision_probability(circuit)
        if np.isclose(ideal_collision, uniform_probability):
            raise ValueError(
                "Clifford XEB normalization is undefined for a uniform ideal state"
            )

        return float(
            (cross_entropy - uniform_probability)
            / (ideal_collision - uniform_probability)
        )