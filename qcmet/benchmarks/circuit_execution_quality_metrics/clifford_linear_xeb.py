"""Here we implement the linear cross-entropy banchmarking with Clifford circuits following PRA 108, 052613

We have implemented a 'cycle' as that pf the of the 1D chain (Fig 1a). This is controlled by the depth parameter. 
A cycle thus consists of 4 layers: single qubit clifford layer, entangling layer, single qubit clifford layer, entangling layer.
The entangling layers are brickwork overlapped.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, List

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, StabilizerState, random_clifford

from qcmet.benchmarks.circuit_execution_quality_metrics.linear_xeb import LinearXEB

if TYPE_CHECKING:
    from qcmet.core import FileManager


class CliffordLinearXEB(LinearXEB):
    """Linear XEB benchmark specialized to Clifford circuits."""

    def __init__(
        self,
        qubits: int | List[int],
        depth: int | List[int],
        num_circuits: int,
        seed: int | None = None,
        save_path: str | Path | FileManager | None = None,
    ):
        """Initialize the Clifford linear XEB benchmark."""
        super().__init__(
            qubits=qubits,
            depth=depth,
            num_circuits=num_circuits,
            seed=seed,
            save_path=save_path,
        )
        self.name = "CliffordLinearXEB"

    def _random_single_qubit_clifford_layer(self) -> QuantumCircuit:
        """Generate a random single-qubit Clifford layer across all qubits."""
        circuit = QuantumCircuit(self.num_qubits)
        for i in range(self.num_qubits):
            circuit.compose(
                random_clifford(1, seed=self.rng).to_circuit(),
                qubits=[i],
                inplace=True,
            )
        return circuit

    def _cnot_layer(
                    self,
                    parity: int,
                ) -> QuantumCircuit:
        """Construct one open-chain nearest-neighbour CNOT layer.

        parity=0 gives (0,1), (2,3), ...
        parity=1 gives (1,2), (3,4), ..."""
        if parity not in (0, 1):
            raise ValueError("parity must be either 0 or 1")

        circuit = QuantumCircuit(self.num_qubits)

        for control in range(parity, self.num_qubits - 1, 2):
            circuit.cx(control, control + 1)

        return circuit

    def build_circuit(self, num_qubits: int, depth: int) -> QuantumCircuit:
        """Build a random Clifford circuit of the specified depth and qubit count."""
        circuit = QuantumCircuit(num_qubits)
        for _ in range(depth):
            circuit.compose(self._random_single_qubit_clifford_layer(), inplace=True)
            circuit.compose(self._cnot_layer(0), inplace=True)
            circuit.compose(self._random_single_qubit_clifford_layer(), inplace=True)
            circuit.compose(self._cnot_layer(1), inplace=True)
        return circuit

    def _generate_circuits(self) -> List[QuantumCircuit]:
        """Generate random Clifford circuits for the configured depths."""
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
        """Return one computational-basis probability from a stabilizer tableau."""
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
        """Calculate linear XEB as defined in Phys. Rev. A 108, 052613."""
        shots = sum(counts.values())
        if shots == 0:
            raise ValueError("Cannot calculate XEB from zero shots")

        probabilities = self._ideal_probabilities(
            circuit,
            list(counts.keys()),
        )

        overlap = sum(
            (count / shots) * probabilities.get(bitstring, 0.0)
            for bitstring, count in counts.items()
        )

        return float((2.0**circuit.num_qubits) * overlap - 1.0)

    def _analyze(self) -> dict[str, Any]:
        """Analyze the measured Clifford XEB data and return summary metrics."""
        fidelities = []
        for circuit, counts in zip(
            self.experiment_data["circuit"],
            self.experiment_data["circuit_measurements"],
            strict=True,
        ):
            fidelities.append(self._cross_entropy_fidelity(circuit, counts))

        self.experiment_data["linear_xeb_fidelity"] = fidelities
        return {
            "mean_fidelity": float(np.mean(fidelities)),
            "fidelities": fidelities,
        }