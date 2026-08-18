"""Clifford Volume Benchmark.

This module implements the Clifford Volume benchmark for quantum circuits from 
https://arxiv.org/pdf/2512.19413
"""

from __future__ import annotations
from qiskit.quantum_info import Pauli

from typing import TYPE_CHECKING, List, Any
from qiskit import QuantumCircuit
from qiskit.quantum_info import StabilizerState
if TYPE_CHECKING:
    from pathlib import Path

    from qcmet.core import FileManager

import numpy as np

from qcmet.benchmarks import BaseBenchmark


class CliffordVolume(BaseBenchmark):
    """Clifford Volume benchmark."""

    def __init__(
        self,
        qubits: int | List[int],
        depth: int | List[int],
        num_circuits: int,
        num_measured_stabilizers: int = 4,
        seed: int | None = None,
        save_path: str | Path | FileManager | None = None,
    ):
        """Initialize the Clifford Volume benchmark."""
        super().__init__(
            name="CliffordVolume",
            qubits=qubits,
            save_path=save_path,
        )
        self.config["depth"] = depth
        self.config["num_circuits"] = num_circuits
        self.config["num_measured_stabilizers"] = num_measured_stabilizers
        self.rng = np.random.default_rng(seed)

    def _append_random_clifford_layer(self, circuit: QuantumCircuit) -> QuantumCircuit:
        single_qubit_gates: List[Any] = [
            circuit.id,
            circuit.x,
            circuit.y,
            circuit.z,
            circuit.h,
            circuit.s,
            circuit.sdg,
        ]
        two_qubit_gates: List[Any] = [
            circuit.cx,
            circuit.swap
        ]
        num_two_qubit_gates = self.rng.integers(0, circuit.num_qubits // 2)
        # select pairs in [1, n] without replacement
        pairs = self.rng.choice(circuit.num_qubits, size=(num_two_qubit_gates, 2), replace=False)
        for pair in pairs:
            gate = self.rng.choice(two_qubit_gates)
            circuit.append(gate, [pair[0], pair[1]])
        # fill in the rest of the qubits with single qubit gates
        for qubit in range(circuit.num_qubits):
            if qubit not in pairs.flatten():
                gate = self.rng.choice(single_qubit_gates)
                circuit.append(gate, [qubit])
        return circuit

    def _random_clifford_circuit(self) -> QuantumCircuit:
        """Generate a random Clifford circuit of specified depth and number of qubits."""

        num_qubits = self.config["qubits"]
        depth = self.config["depth"]
        circuit = QuantumCircuit(num_qubits)
    
        for _ in range(depth):
            circuit = self._append_random_clifford_layer(circuit)
        return circuit

    def _generate_circuits(self) -> List[QuantumCircuit]:
        """Generate a list of random Clifford circuits with appended circuits for the necessary stabilizer and destabilizer measurements.
        
        This is not just the list of the random Clifford circuits;
        it includes additional circuits for measuring destabilizers.
        For example, if we want 4 stabs and destabs (as in the paper) for each random Clifford circuit,
        we will generate 8 circuits for each random Clifford circuit.

        Luckily, this approach only needs a few circuits (4 in the paper), so it is not too expensive.
        This yields 4 * 8 = 32 circuits total. The bulk of the work is taking the repeated shots.
        """

        circuits = []
        for _ in range(self.config["num_circuits"]):
            # We want to measure the stabilizers and destabilizers
            circuit = self._random_clifford_circuit()
            state = StabilizerState(circuit)
            clifford = state.clifford
            stabilizer_labels = clifford.to_labels(mode="S")
            destabilizer_labels = clifford.to_labels(mode="D")
            stablizers_to_measure = self.rng.choice(stabilizer_labels, size=self.config["num_measured_stabilizers"], replace=False)
            destabilizers_to_measure = self.rng.choice(destabilizer_labels, size=self.config["num_measured_stabilizers"], replace=False)
            for stabilizer in stablizers_to_measure:
                pauli = Pauli(stabilizer)
                measurement_circuit = self._pauli_measurement_circuit(circuit, pauli)
                circuits.append(measurement_circuit)
            for destabilizer in destabilizers_to_measure:
                pauli = Pauli(destabilizer)
                measurement_circuit = self._pauli_measurement_circuit(circuit, pauli)
                circuits.append(measurement_circuit)

        return circuits


    def _pauli_measurement_circuit(self,
                                   preparation_circuit: QuantumCircuit,
                                   pauli: Pauli,
                                   ) -> QuantumCircuit:
        """Given a circuit and a set of Paulis to measure, e.g. "XIZY",
          return a circuit that prepares the state and measures the specified Paulis."""

        num_qubits = self.config["qubits"]

        # Create fresh classical bits so the output is unambiguous.
        measurement_circuit = QuantumCircuit(num_qubits)

        # Prepare C|0...0>.
        measurement_circuit.compose(
            preparation_circuit,
            qubits=range(num_qubits),
            inplace=True,
        )
        measurement_circuit.barrier()

        for qubit in range(num_qubits):
            has_x = bool(pauli.x[qubit])
            has_z = bool(pauli.z[qubit])

            if has_x and not has_z:
                # X measurement: H X H = Z
                measurement_circuit.h(qubit)

            elif has_x and has_z:
                # Y measurement: H S^dagger Y S H = Z
                measurement_circuit.sdg(qubit)
                measurement_circuit.h(qubit)

            elif has_z:
                # Z measurement: no basis rotation needed.
                pass

            else:
                # Identity: measured result will be ignored in post-processing.
                pass

        return measurement_circuit