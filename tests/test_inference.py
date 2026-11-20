import pathlib

import h5py
import numpy as np
import pytest
import torch
import torch.nn as nn

from MLGWSC_proposal.inference import (
    cluster_detections,
    mlgwsc1_output,
    model_inference,
)


def test_cluster_detections_empty() -> None:
    # Testing edge case with zero detections
    empty_times = np.empty(0, dtype=np.float64)
    empty_probs = np.empty(0, dtype=np.float32)

    selected_times, selected_probs = cluster_detections(empty_times, empty_probs)

    assert isinstance(selected_times, np.ndarray)
    assert isinstance(selected_probs, np.ndarray)
    assert len(selected_times) == 0
    assert len(selected_probs) == 0


def test_cluster_detections() -> None:
    # Testing clustering with:
    # - Unsorted input timestamps
    # - A single isolated detection (10.0)
    # - A multi-trigger cluster around 50.0 (50.0, 50.25, 50.5 -> peak at 50.25 with prob 0.92)
    # - A trailing isolated detection (90.0)
    signal_times = np.array([50.25, 10.0, 50.5, 90.0, 50.0], dtype=np.float64)
    signal_probs = np.array([0.92, 0.65, 0.78, 0.81, 0.70], dtype=np.float32)

    selected_times, selected_probs = cluster_detections(
        signal_times=signal_times,
        signal_probs=signal_probs,
        cluster_window=0.5,
    )

    expected_times = np.array([10.0, 50.25, 90.0], dtype=np.float64)
    expected_probs = np.array([0.65, 0.92, 0.81], dtype=np.float32)

    assert selected_times.shape == (3,)
    assert selected_probs.shape == (3,)
    assert np.allclose(selected_times, expected_times)
    assert np.allclose(selected_probs, expected_probs)


def test_mlgwsc1_output(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    output_path = tmp_path / "submission_triggers.hdf"
    selected_times = np.array([1000000010.25, 1000000055.50, 1000000120.00], dtype=np.float64)
    selected_probs = np.array([0.75, 0.98, 0.61], dtype=np.float32)
    cluster_window = 0.5

    mlgwsc1_output(
        selected_times=selected_times,
        selected_probs=selected_probs,
        output_path=str(output_path),
        cluster_window=cluster_window,
    )

    # Checking console confirmation message
    captured = capsys.readouterr()
    assert "Output HDF5 file generated successfully" in captured.out

    # Verifying the official MLGWSC-1 HDF5 structure and datasets
    assert output_path.exists()
    with h5py.File(output_path, "r") as f:
        assert set(f.keys()) == {"time", "stat", "var"}

        time_dset = f["time"][()]
        stat_dset = f["stat"][()]
        var_dset = f["var"][()]

        assert time_dset.shape == stat_dset.shape == var_dset.shape == (3,)
        assert np.allclose(time_dset, selected_times)
        assert np.allclose(stat_dset, selected_probs)
        assert np.allclose(var_dset, np.full(3, cluster_window))


def test_model_inference(tmp_path: pathlib.Path) -> None:
    ts_path = tmp_path / "raw_test_strain.hdf"

    # Creating two discontinuous 10s strain blocks at 2048 Hz to test multi-key handling
    rng = np.random.default_rng(seed=42)
    sample_rate = 2048
    duration_sec = 10
    gps_1 = 1000000000
    gps_2 = 1000001000

    with h5py.File(ts_path, "w") as f:
        f.attrs["low_frequency_cutoff"] = 20.0
        h1 = f.create_group("H1")
        l1 = f.create_group("L1")

        for gps in (gps_1, gps_2):
            h1[str(gps)] = rng.standard_normal(duration_sec * sample_rate)
            l1[str(gps)] = rng.standard_normal(duration_sec * sample_rate)

    # Deterministic mock network whose output logit increases monotonically with batch index,
    # guaranteeing clear above-threshold detections in the second block and below-threshold
    # in the first.
    class DeterministicMockNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dummy_param = nn.Parameter(torch.zeros(1))
            self.call_offset = 0

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            batch_len = x.shape[0]
            indices = torch.arange(
                self.call_offset,
                self.call_offset + batch_len,
                device=x.device,
                dtype=torch.float32,
            )
            self.call_offset += batch_len
            # Negative logits for early segments (prob < 0.5), positive logits later (prob > 0.5)
            logits = (indices - 45.0) * 0.2 + self.dummy_param
            return logits.unsqueeze(1)

    model = DeterministicMockNet()

    interval = 2.5
    stride = 1.0
    # With duration=10s, interval=2.5s, step=0.25s:
    # n_seg per block = (10*2048 - 2.5*2048) // (0.25*2048) + 1 = 31 segments -> 62 total segments.
    # Indices > 45 correspond to global indices 46..61 (local indices 15..30 in gps_2).
    # With monotonically increasing probabilities, step=0.25s, and cluster_window=0.5s,
    # NMS keeps every 3rd segment starting from the highest peak (local index 30):
    # local indices [15, 18, 21, 24, 27, 30] -> 6 distinct events.
    selected_times, selected_probs = model_inference(
        model=model,
        ts_path=str(ts_path),
        interval=interval,
        stride=stride,
        chunk_size=16,
        batch_size=8,
        prob_threshold=0.5,
        cluster_window=0.5,
    )

    expected_local_indices = np.array([15, 18, 21, 24, 27, 30])
    expected_times = float(gps_2) + (interval - stride) / 2 + expected_local_indices * (stride / 4)
    expected_global_indices = 31 + expected_local_indices
    expected_logits = (expected_global_indices - 45.0) * 0.2
    expected_probs = 1.0 / (1.0 + np.exp(-expected_logits))

    assert isinstance(selected_times, np.ndarray)
    assert isinstance(selected_probs, np.ndarray)
    assert selected_times.shape == selected_probs.shape == (6,)
    assert np.allclose(selected_times, expected_times)
    assert np.allclose(selected_probs, expected_probs, atol=1e-6)
