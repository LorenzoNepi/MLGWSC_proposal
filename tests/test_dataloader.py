import pathlib

import h5py
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from MLGWSC_proposal.dataloader import (
    Background_Dataset,
    Foreground_Dataset,
    generate_back_dataloaders,
    generate_fore_dataloaders,
)


def test_foreground_dataset_curriculum() -> None:
    # Creating a mock 3D tensor: 10 segments, 2 channels, 100 time points
    num_segments = 10
    X_mock = torch.randn(num_segments, 2, 100)

    # Mock SNR values in random order
    snr_mock = torch.tensor([5.0, 20.0, 10.0, 25.0, 15.0, 2.0, 8.0, 30.0, 12.0, 18.0])

    # Initializing dataset with snr_split = 0.4 (top 40% highest SNR)
    dataset = Foreground_Dataset(X_mock, snr_split=0.4, snr=snr_mock)

    # Asserting length respects the split
    assert len(dataset) == 4

    # Asserting elements are sorted by descending SNR and y is always 1.0
    expected_sorted_snr = torch.tensor([30.0, 25.0, 20.0, 18.0])
    for i in range(4):
        _, label = dataset[i]
        assert label.item() == 1.0
        assert dataset.snr[i] == expected_sorted_snr[i]

    # Updating snr_split via property to 1.0 (100% dataset)
    dataset.snr_split = 1.0
    assert len(dataset) == 10


def test_background_dataset() -> None:
    # Creating a mock 3D tensor: 15 segments, 2 channels, 2048 time points
    num_segments = 15
    sequence_length = 2048
    X_mock = torch.randn(num_segments, 2, sequence_length)

    # Initializing the background dataset
    dataset = Background_Dataset(X_mock)

    # Asserting length respects the input tensor shape
    assert len(dataset) == num_segments

    # Checking the returned items
    for i in range(5):
        x_item, y_item = dataset[i]
        # Asserting the shape of the extracted sequence is (num_channels, sequence_length)
        assert x_item.shape == (2, sequence_length)
        # Asserting that the label is exactly 0.0 (background/noise)
        assert y_item.item() == 0.0


def test_generate_back_dataloaders(tmp_path: pathlib.Path) -> None:
    # Setting up mock HDF5 file
    mock_hdf5_path = tmp_path / "mock_background.hdf5"

    num_segments = 100
    sequence_length = 2048

    with h5py.File(mock_hdf5_path, "w") as f:
        # Creating a mock dataset with shape (2, num_segments, sequence_length)
        rng = np.random.default_rng(seed=42)
        mock_data = rng.standard_normal((2, num_segments, sequence_length)).astype(np.float32)
        f.create_dataset("whitened", data=mock_data)

    # Generating Dataloaders with standard split
    train_loader, val_loader, test_loader = generate_back_dataloaders(
        segmented_background_path=str(mock_hdf5_path),
        validation_split=[0.7, 0.2, 0.1],
        batch_size=16
    )

    # Asserting correct return types
    assert isinstance(train_loader, DataLoader)
    assert isinstance(val_loader, DataLoader)
    assert isinstance(test_loader, DataLoader)

    # Asserting the split on the datasets wrapped by the loader with 100 segments
    assert len(train_loader.dataset) == 70
    assert len(val_loader.dataset) == 20
    assert len(test_loader.dataset) == 10
    # Asserting correct batch size
    assert train_loader.batch_size == 16

    # Asserting error handling
    with pytest.raises(ValueError, match="validation_split must be a list"):
        generate_back_dataloaders(
            str(mock_hdf5_path), validation_split=[0.6, 0.2, 0.1] # Sum 0.9
        )


def test_generate_fore_dataloaders(tmp_path: pathlib.Path) -> None:
    # Setting up the mock HDF5 file
    mock_hdf5_path = tmp_path / "mock_foreground.hdf5"

    num_segments = 100
    sequence_length = 2048

    with h5py.File(mock_hdf5_path, "w") as f:
        # Creating datasets with initial shape (2, num_segments, sequence_length) for strains
        # and (num_segments,) for SNR values
        rng = np.random.default_rng(seed=42)
        mock_data = rng.standard_normal((2, num_segments, sequence_length)).astype(np.float32)
        mock_snr = rng.random(num_segments).astype(np.float32) * 10
        f.create_dataset("whitened", data=mock_data)
        f.create_dataset("snr", data=mock_snr)

    # Generating DataLoaders with a standard split, but filtering the training
    # set to only include the top 50% highest SNR segments (snr_split=0.5)
    train_loader, val_loader, test_loader = generate_fore_dataloaders(
        segmented_foreground_path=str(mock_hdf5_path),
        validation_split=[0.7, 0.2, 0.1],
        batch_size=16,
        snr_split=0.5
    )

    # Asserting correct return types
    assert isinstance(train_loader, DataLoader)
    assert isinstance(val_loader, DataLoader)
    assert isinstance(test_loader, DataLoader)

    # Checking dataset lengths with curriculum learning applied
    # Original data: 100 segments. Split is [0.7, 0.2, 0.1] -> 70, 20, 10
    # Since snr_split = 0.5, training set should be halved (int(70 * 0.5) = 35)
    assert len(train_loader.dataset) == 35
    # Validation and test sets should remain intact (100% of their split)
    assert len(val_loader.dataset) == 20
    assert len(test_loader.dataset) == 10
    # Asserting correct batch size
    assert train_loader.batch_size == 16

    # Asserting error handling
    with pytest.raises(ValueError, match="validation_split must be a list"):
        generate_fore_dataloaders(
            str(mock_hdf5_path), validation_split=[0.6, 0.2, 0.1] # Sums to 0.9
        )
