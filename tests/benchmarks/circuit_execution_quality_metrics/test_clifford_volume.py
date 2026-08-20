"""Tests for the Clifford Volume benchmark."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from qcmet.benchmarks.circuit_execution_quality_metrics.clifford_volume import (
    CliffordVolume,
)


def test_analyze_applies_stabilizer_and_destabilizer_thresholds():
    """Verify the individual and average threshold conditions."""
    benchmark = CliffordVolume(
        qubits=2,
        depth=1,
        num_circuits=1,
        num_measured_stabilizers=2,
        seed=1,
    )
    benchmark.generate_circuits()

    benchmark.experiment_data["pauli_label"] = "IZ"
    # Setting <S> = (900 - 100)/1000 = 0.8, <D> = (500 - 500)/1000 = 0
    benchmark.load_circuit_measurements(
        [
            {"00": 900, "10": 100},   
            {"00": 900, "10": 100},
            {"00": 500, "10": 500},
            {"00": 500, "10": 500},
        ]
    )
    benchmark._runtime_params = {"num_shots": 1000}

    result = benchmark.analyze()

    assert result["stabilizer_expectation_values"] == [[0.8, 0.8]]
    assert result["destabilizer_expectation_values"] == [[0.0, 0.0]]
    # sqrt(1 - 0.8^2)/sqrt(1000) = sqrt(0.36/1000)
    assert np.allclose(
        result["stabilizer_standard_deviations"],
        [[np.sqrt(0.36 / 1000), np.sqrt(0.36 / 1000)]],
    )
    assert result["stabilizer_worst_case_pass"] == [[True, True]]
    assert result["destabilizer_worst_case_pass"] == [[True, True]]
    assert result["stabilizer_average_pass"] == [True]
    assert result["destabilizer_average_pass"] == [True]
    assert result["passes"]
    benchmark.config["destabilizer_threshold"] = 0.05
    assert not benchmark.analyze()["passes"]


def test_generated_circuits_include_measurements_and_pauli_metadata():
    """Verify generated measurement circuits can be analyzed from their counts."""
    benchmark = CliffordVolume(
        qubits=2,
        depth=1,
        num_circuits=1,
        num_measured_stabilizers=2,
        seed=1,
    )
    benchmark.generate_circuits()

    assert len(benchmark.circuits) == 4
    assert set(benchmark.experiment_data["measurement_type"]) == {
        "stabilizer",
        "destabilizer",
    }
    assert all("measure" in circuit.count_ops() for circuit in benchmark.circuits)