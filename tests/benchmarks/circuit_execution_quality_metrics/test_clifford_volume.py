"""Tests for the Clifford Volume benchmark."""

import numpy as np

import sys
from pathlib import Path

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
    benchmark.load_circuit_measurements(
        [
            {"00": 90, "10": 10},
            {"00": 90, "10": 10},
            {"00": 40, "10": 60},
            {"00": 40, "10": 60},
        ]
    )
    benchmark._runtime_params = {"num_shots": 100}

    result = benchmark.analyze()

    assert result["stabilizer_expectation_values"] == [[0.8, 0.8]]
    assert result["destabilizer_expectation_values"] == [[-0.2, -0.2]]
    assert np.allclose(result["stabilizer_standard_deviations"], [[0.06, 0.06]])
    assert result["stabilizer_worst_case_pass"] == [[True, True]]
    assert result["destabilizer_worst_case_pass"] == [[True, True]]
    assert result["stabilizer_average_pass"] == [True]
    assert result["destabilizer_average_pass"] == [True]
    assert result["passes"]

    benchmark.config["destabilizer_threshold"] = -0.1
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