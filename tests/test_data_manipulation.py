import pathlib

import h5py
import numpy as np
import pytest
import torch

from MLGWSC_proposal.data_manipulation import (
    explore_data,
    foreground_segmentation,
    get_tc,
    torch_segmentation,
    welch_psd,
    whiten,
)


def test_explore_data(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    # Creating come temporary test files
    # tmp_path is recognized by pytest. It is used to create a "mock" path
    # in which test data can be saved for the duration of the test.
    bkg_path = tmp_path / "background.hdf5"
    fore_path = tmp_path / "foreground.hdf5"
    inj_path = tmp_path / "injections.hdf5"

    # Fill the files with some attributes and groups (minimal)
    for path in [bkg_path, fore_path]:
        with h5py.File(path, "w") as f:
            f.attrs["test_attr"] = 42
            f.create_group("H1")
            f["H1"].attrs["detector"] = "test_detector"
            f.create_group("L1")

    with h5py.File(inj_path, "w") as f:
        f.attrs["inj_version"] = "1.0"

    # Execute the function
    explore_data(str(bkg_path), str(fore_path), str(inj_path))

    # Capturing the output printed to the terminal with capsys.
    # cpasys.readouterr() returns a tuple with 2 object:
    # - caputred.out, a string with all the text sent to the terminal
    # - captured.err, a string with all the error messages (if present)
    captured = capsys.readouterr()
    output = captured.out

    assert "BACKGROUND HDF5 FILE" in output
    assert "FOREGROUND HDF5 FILE" in output
    assert "INJECTIONS HDF5 FILE" in output
    assert "test_attr" in output
    assert "H1 group keys" in output
    assert "inj_version" in output


def test_welch_psd() -> None:
    # Testing on a mock tensor with shape (num_channels, segment_length)
    test_tensor = torch.rand(size = (2, 10**4))
    psd = welch_psd(test_tensor, 10**3)

    assert isinstance(psd, torch.Tensor)

    # Expected shape: (num_channels, interval_index // 2 + 1)
    assert psd.shape == (2, 501)

    # Asserting no NaN values are present in the psd
    assert not torch.isnan(psd).any()


def test_whiten() -> None:
    # Testing on a mock tensor with shape (num_channels, num_intervals, interval_points)
    test_tensor = torch.rand(size = (2, 100, 1000))
    # Using a PSD with shape (num_channels, interval_index // 2 + 1)
    test_psd = torch.abs(torch.rand(size = (2, 501)))
    whitened = whiten(
        torch_segmented = test_tensor,
        interval = 1000 / 2048,
        psd = test_psd,
        hz_cutoff = 10
    )

    assert isinstance(whitened, torch.Tensor)

    # Expected shape: (num_channels, interval_index // 2 + 1)
    assert whitened.shape == (2, 100, 1000)

    # Asserting no NaN values are present in the psd
    assert not torch.isnan(whitened).any()


def test_get_tc(tmp_path: pathlib.Path) -> None:

    fore_path = tmp_path / "foreground.hdf5"
    inj_path = tmp_path / "injections.hdf5"

    # Creating the HDF5 files with the structure expected by the function
    tc_list = []
    with h5py.File(fore_path, "w") as fore, h5py.File(inj_path, "w") as inj:
        fore.create_group("H1")
        fore.attrs["padding_start"] = 50
        fore.attrs["padding_end"] = 50

        for a in range(2048 * 10**3, 2048 * 10**4, 2048 * 10**3):
            # Generating groups and datasets to simulate 1000s segments
            rng = np.random.default_rng()
            fore["H1"][str(a)] = rng.standard_normal(2048 * 1000)
            # Generating a mock injection time between 50s and 950s for each segment
            tc = a + 50 + rng.standard_normal() * 900
            tc_list.append(tc)

        inj["tc"] = tc_list

        tc_dict_original, tc_dict_random = get_tc(str(fore_path), str(inj_path), 200)

        # Asserting that the original times are correctly saved in a dictionary
        assert len(tc_dict_original.keys()) == 9

        # Asserting that random times are correctly generetad
        assert all(
            len(tc_dict_original[key]) < len(tc_dict_random[key])
            for key in tc_dict_original
            )

        # Asserting that all the generated times are in the correct interval
        assert all(
        50 <= tc - int(start_time) <= 950
        for start_time, tc_list in tc_dict_random.items()
        for tc in tc_list
        )


def test_torch_segmentation(tmp_path: pathlib.Path) -> None:
    # Setting up test paths
    data_path = tmp_path / "raw_data.hdf"
    segmented_path = tmp_path / "segmented_background.hdf"

    # Creating mock raw data: 10s at 2048 Hz
    rng = np.random.default_rng()
    test_strain = rng.standard_normal(2048 * 10)

    # Creating test HDF5 files with the correct expected structure
    with h5py.File(data_path, "w") as f:
        f.attrs["low_frequency_cutoff"] = 20

        h1 = f.create_group("H1")
        l1 = f.create_group("L1")

        # Mock key for GPS time: "1000000000"
        h1["1000000000"] = test_strain
        l1["1000000000"] = test_strain

    # Calling the segmentation function with a 2s interval and a 1s stride
    torch_segmentation(
        data_path=str(data_path),
        segmented_path=str(segmented_path),
        interval=2,
        stride=1
    )

    # Asserting the the output file was actually created
    assert segmented_path.exists()

    with h5py.File(segmented_path, "r") as f:
        assert "whitened" in f
        dset = f["whitened"]

        # Asserting that the output dataset shape is (num_channels, num_segments, stride * 2048)
        # With stride = 1, interval = 2, duration = 10 we have
        # num_segments = (10 - 2) // 1 + 1 = 9
        assert dset.shape[0] == 2
        assert dset.shape[1] == 9
        assert dset.shape[2] == 2048

        assert not np.isnan(dset[()]).any()


def test_foreground_segmentation(tmp_path: pathlib.Path) -> None:

    foreground_path = tmp_path / "raw_foreground.hdf5"
    injections_path = tmp_path / "injections.hdf5"
    segmented_path = tmp_path / "segmented_foreground.hdf5"

    # Creating mock raw data: 15s at 2048 Hz
    rng = np.random.default_rng()
    test_strain = rng.standard_normal(15 * 2048)
    start_gps = 1000000000

    # Creating mock foreground data
    with h5py.File(foreground_path, "w") as f:
        f.attrs["low_frequency_cutoff"] = 20.0
        f.attrs["padding_start"] = 2
        f.attrs["padding_end"] = 2

        h1 = f.create_group("H1")
        l1 = f.create_group("L1")

        h1[str(start_gps)] = test_strain
        l1[str(start_gps)] = test_strain

    # Creating mock injection data
    with h5py.File(injections_path, "w") as f:
        # Put an injection exactly in the middle of our 15s segment
        f.create_dataset("tc", data=np.array([start_gps + 7.5]))

    # Calling the foreground segmentation function with
    # interval = 2s, stride = 1s, default_time = 0.4s
    foreground_segmentation(
        foreground_path=str(foreground_path),
        injections_path=str(injections_path),
        segmented_foreground_path=str(segmented_path),
        interval=2,
        stride=1,
        default_time=0.4
    )

    # Asserting the the output file was actually created
    assert segmented_path.exists()

    with h5py.File(segmented_path, "r") as f:
        assert "whitened" in f
        dset = f["whitened"]

        # Asserting that the output dataset shape is (num_channels, num_random_tc, stride * 2048)
        # Since 1 signal was injected we expect 5 random tc to be generated and appended
        assert dset.shape[0] == 2
        assert dset.shape[1] == 5
        assert dset.shape[2] == 2048

        assert not np.isnan(dset[()]).any()

def test_segmentation_exceptions(tmp_path: pathlib.Path) -> None:
    # We only need mock paths to pass as the function parameters,
    # the functions will fail before attempting to read/write them.
    dummy_path_1 = str(tmp_path / "dummy1.hdf5")
    dummy_path_2 = str(tmp_path / "dummy2.hdf5")
    dummy_path_3 = str(tmp_path / "dummy3.hdf5")

    # Testing ValueError for non-integer indexes in torch_segmentation
    with pytest.raises(ValueError, match="must be int"):
        torch_segmentation(
            dummy_path_1, dummy_path_2,
            interval=2.0001, # This will cause an invalid index when multiplied by 2048
            stride=1
        )

    # Testing ValueError for non-integer indexes in foreground_segmentation
    with pytest.raises(ValueError, match="must be int"):
        foreground_segmentation(
            dummy_path_1, dummy_path_2, dummy_path_3,
            interval=2.0001,
            stride=1,
            default_time=0.4
        )

    # Testing ValueError for incompatible stride and default_time
    with pytest.raises(ValueError, match="higher than the default_time"):
        foreground_segmentation(
            dummy_path_1, dummy_path_2, dummy_path_3,
            interval=2,
            stride=0.3,       # Stride is lower than default_time
            default_time=0.4
        )
