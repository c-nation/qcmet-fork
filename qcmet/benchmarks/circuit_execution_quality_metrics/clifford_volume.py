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
        stabilizer_threshold: float = 1 / np.exp(1),  # 1/e
        destabilizer_threshold: float = 1 / (2 * np.exp(1)),  # 1/e
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
        self.config["stabilizer_threshold"] = stabilizer_threshold
        self.config["destabilizer_threshold"] = destabilizer_threshold
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
        num_two_qubit_gates = self.rng.integers(0, circuit.num_qubits // 2 + 1)
        # select pairs in [1, n] without replacement
        pairs = self.rng.choice(circuit.num_qubits, size=(num_two_qubit_gates, 2), replace=False)
        for pair in pairs:
            gate = self.rng.choice(two_qubit_gates)
            gate(pair[0], pair[1])
        # fill in the rest of the qubits with single qubit gates
        for qubit in range(circuit.num_qubits):
            if qubit not in pairs.flatten():
                gate = self.rng.choice(single_qubit_gates)
                gate(qubit)
        return circuit

    def _random_clifford_circuit(self) -> QuantumCircuit:
        """Generate a random Clifford circuit of specified depth and number of qubits."""

        num_qubits = self.num_qubits
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
        for circuit_index in range(self.config["num_circuits"]):
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
                circuits.append(
                    self._circ_with_metadata_dict(
                        measurement_circuit,
                        circuit_index=circuit_index,
                        measurement_type="stabilizer",
                        pauli_label=pauli.to_label(),
                    )
                )
            for destabilizer in destabilizers_to_measure:
                pauli = Pauli(destabilizer)
                measurement_circuit = self._pauli_measurement_circuit(circuit, pauli)
                circuits.append(
                    self._circ_with_metadata_dict(
                        measurement_circuit,
                        circuit_index=circuit_index,
                        measurement_type="destabilizer",
                        pauli_label=pauli.to_label(),
                    )
                )

        return circuits


    def _pauli_measurement_circuit(self,
                                   preparation_circuit: QuantumCircuit,
                                   pauli: Pauli,
                                   ) -> QuantumCircuit:
        """Given a circuit and a set of Paulis to measure, e.g. "XIZY",
          return a circuit that prepares the state and measures the specified Paulis."""

        num_qubits = self.num_qubits

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

        measurement_circuit.measure_all()

        return measurement_circuit

    def _analyze(self) -> dict[str, Any]:
        """Analyze the results of the Clifford Volume benchmark.

        Returns:
            dict[str, Any]: A dictionary containing the analysis results, including the average fidelity and other relevant metrics.
        """
        num_shots = self._runtime_params["num_shots"]
        num_measured = self.config["num_measured_stabilizers"]

        def expectation_value(counts: dict[str, int], pauli_label: str) -> float:
            sign = -1 if pauli_label.startswith("-") else 1
            pauli_string = pauli_label.lstrip("+-")[::-1]
            expectation = 0
            for state, count in counts.items():
                parity = 1
                for bit, pauli in zip(state, pauli_string, strict=False):
                    if pauli != "I" and bit == "1":
                        parity *= -1
                expectation += parity * count
            return sign * expectation / num_shots

        def measurement_statistics(measurement_type: str) -> tuple[np.ndarray, np.ndarray]:
            measurements = self.experiment_data[
                self.experiment_data["measurement_type"] == measurement_type
            ]
            expectations = np.array(
                [
                    expectation_value(counts, pauli_label)
                    for counts, pauli_label in zip(
                        measurements["circuit_measurements"],
                        measurements["pauli_label"],
                        strict=True,
                    )
                ]
            ).reshape(self.config["num_circuits"], num_measured)
            standard_deviations = np.sqrt(
                np.maximum(0, 1 - expectations**2) / num_shots
            )
            return expectations, standard_deviations

        stabilizers, stabilizer_std = measurement_statistics("stabilizer")
        destabilizers, destabilizer_std = measurement_statistics("destabilizer")

        stabilizer_worst_case = (
            stabilizers - 2 * stabilizer_std >= self.config["stabilizer_threshold"]
        )
        destabilizer_worst_case = (
            destabilizers + 2 * destabilizer_std <= self.config["destabilizer_threshold"]
        )

        average_stabilizers = np.mean(stabilizers, axis=1)
        average_destabilizers = np.mean(destabilizers, axis=1)
        average_stabilizer_std = np.sqrt(np.sum(stabilizer_std**2, axis=1)) / num_measured
        average_destabilizer_std = np.sqrt(np.sum(destabilizer_std**2, axis=1)) / num_measured
        stabilizer_average = (
            average_stabilizers - 5 * average_stabilizer_std
            >= self.config["stabilizer_threshold"]
        )
        destabilizer_average = (
            average_destabilizers + 5 * average_destabilizer_std
            <= self.config["destabilizer_threshold"]
        )

        return {
            "stabilizer_expectation_values": stabilizers.tolist(),
            "destabilizer_expectation_values": destabilizers.tolist(),
            "stabilizer_standard_deviations": stabilizer_std.tolist(),
            "destabilizer_standard_deviations": destabilizer_std.tolist(),
            "average_stabilizer_expectation_values": average_stabilizers.tolist(),
            "average_destabilizer_expectation_values": average_destabilizers.tolist(),
            "average_stabilizer_standard_deviations": average_stabilizer_std.tolist(),
            "average_destabilizer_standard_deviations": average_destabilizer_std.tolist(),
            "stabilizer_worst_case_pass": stabilizer_worst_case.tolist(),
            "destabilizer_worst_case_pass": destabilizer_worst_case.tolist(),
            "stabilizer_average_pass": stabilizer_average.tolist(),
            "destabilizer_average_pass": destabilizer_average.tolist(),
            "passes": bool(
                np.all(stabilizer_worst_case)
                and np.all(destabilizer_worst_case)
                and np.all(stabilizer_average)
                and np.all(destabilizer_average)
            ),
        }