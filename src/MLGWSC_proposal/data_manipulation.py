import warnings

import h5py
import numpy as np
import torch


# ---------- Data exploration ----------
def explore_data(
    background_path: str,
    foreground_path: str,
    injections_path: str
) -> None:
    """Print relevant info about the background, foreground and injections HDF5 files for training.

    Parameters
    ----------
    background_path : str
        Path to the HDF5 file containing the background strain data.
    foreground_path : str
        Path to the HDF5 file containing the foreground strain data.
    injections_path : str
        Path to the HDF5 file containing the injections parameters.
    """

    files_to_explore = {
        "BACKGROUND": background_path,
        "FOREGROUND": foreground_path
    }

    for name, path in files_to_explore.items():
        print(f"\n{name} HDF5 FILE\n")
        with h5py.File(path, "r") as f:

            print(f"Root group keys: {f.keys()}")
            f_attr = dict(f.attrs.items())
            print("Root group attributes:")
            print(*f_attr.items(), sep="\n")

            H1_group = f["H1"]
            print(f"\nH1 group keys: {H1_group.keys()}")
            print(f"H1 group attributes: {H1_group.attrs.keys()}")

            L1_group = f["L1"]
            print(f"\nL1 group keys: {L1_group.keys()}")
            print(f"L1 group attributes: {L1_group.attrs.keys()}")

    print("\nINJECTIONS HDF5 FILE\n")

    with h5py.File(injections_path, "r") as f:
        print(f"Root group keys: {f.keys()}")
        f_attr = dict(f.attrs.items())
        print("Root group attributes:")
        print(f.attrs.keys())

# ---------- Segmentation and whitening ----------
def welch_psd(
    torch_segment: torch.Tensor,
    interval_index: int,
    device: torch.device | str = "cpu"
) -> torch.Tensor:
    """Compute the Power Spectral Density (PSD) on a strain segment using the Welch method.

    Parameters
    ----------
    torch_segment : torch.Tensor
        Tensor containing the strain data.
    interval_index : int
        Number of sampled points per interval. Expected shape is `(num_channels, segment_length)`.
    device : torch.device | str, optional
        Device that will execute the computations, by default `"cpu"`.

    Returns
    -------
    torch.Tensor
        Tensor containing the PSD with shape
        `(num_channels, interval_index // 2 + 1)` due to the real FFT.
    """

    # Creating Welch windows with the right duration and 50% overlap
    welch_segmented = torch_segment.unfold(1, interval_index, interval_index//2)
    # Shape: (2, #windows, interval * 2048)

    # Using a Hann window function to make the psd "smoother"
    hann_window = torch.hann_window(interval_index, dtype = torch_segment.dtype, device = device)
    welch_segmented *= hann_window

    # Computing the squared module of the 1-D Fourier transform with normalization ("forward")
    welch_fourier = torch.fft.rfft(welch_segmented, dim = 2, norm = "forward")
    welch_fourier = torch.abs(welch_fourier)**2
    # Shape: (2, interval_index//2 + 1) due to torch.fft.rfft

    # Computing the PSD with the same length of the interval
    psd_welch = torch.mean(welch_fourier, dim = 1)

    return psd_welch

def whiten(
    torch_segmented: torch.Tensor,
    interval : float,
    psd : torch.Tensor,
    device : torch.device | str = "cpu",
    hz_cutoff : float = 20
    ) -> torch.Tensor:
    """Whiten strain segments in the frequency domain after applying a high-pass filter.

    This function applies a Hann window to reduce edge artifacts, transforms the
    segments to the frequency domain using RFFT, zeroes out frequencies below
    the specified cutoff, divides by the Amplitude Spectral Density (ASD = sqrt(PSD)),
    and converts the signals back to the time domain via IRFFT.

    Parameters
    ----------
    torch_segmented : torch.Tensor
        Tensor containing the segments with shape
        `(num_channels, num_intervals, interval_points)`.
    interval : float
        Length of the segments in seconds.
    psd : torch.Tensor
        Tensor containing the PSD for the each channels with shape
        `(num_channels, interval_points//2 +1)`.
    device : torch.device | str, optional
        Device that will execute the computations, by default `"cpu"`.
    hz_cutoff : float, optional
        Frequency cut-off (Hz), by default 20.

    Returns
    -------
    torch.Tensor
        Tensor containing the whitened segments.
    """

    # Using a Hann window to reduce border effects
    hann_window = torch.hann_window(
        int(interval * 2048),
        dtype = torch_segmented.dtype,
        device = device
        )
    torch_segmented = torch_segmented * hann_window

    # Transforming the segments to the frequency domain
    fft_segments = torch.fft.rfft(torch_segmented, dim = 2, norm = "forward")
    inv_asd = 1 / torch.sqrt(psd + 1e-15) # Added for stability

    # Applying frequency cut-off
    # delta_f = 1 / interval => cutoff_index = hz_cutoff / delta_f = interval * hz_cutoff
    cutoff_index = int(interval * hz_cutoff)
    inv_asd[:, :cutoff_index] = 0.0

    # Whitening the data and going back to time domain
    whitened_fourier = fft_segments * inv_asd.unsqueeze(1)
    torch_whitened = torch.fft.irfft(
        whitened_fourier,
        n = int(interval * 2048),
        dim = 2,
        norm = "forward"
        )

    return torch_whitened

def torch_segmentation(
    data_path : str,
    segmented_path : str,
    max_length : float | int = 73728000,
    interval : float | int = 6,
    stride : float | int = 2.5
) -> None:
    """Divide the strain data into whitened, fixed-length segments.

    The raw strain timeseries is segmented using a sliding window of duration `interval`
    with step size `stride` to compute the PSD. The central portion of each whitened
    window (of length `stride`) is preserved and saved to an HDF5 file. This procedure
    is used both for model inference and for generating the background training dataset.

    Parameters
    ----------
    data_path : str
        Path to the HDF5 file containing the raw strain data.
    segmented_path : str
        Path to the output HDF5 file with the segmented and whitened signal.
    max_length : float | int, optional
        Maximum number of sampled points allowed in the GPU VRAM,
        by default 73,728,000 (corresponding to 10 hours at 2048 Hz).
    interval : float | int, optional
        Duration in seconds of the window used to compute the PSD, by default 6.
    stride : float | int, optional
        Stride of the windows in seconds, which also defines the output
        segment duration, by default 2.5.
    """

    # Trying to use the GPU for the computations
    device = (
        torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    )

    # Checking if interval and stride values can be converted into valid indexes
    interval_index = interval * 2048
    stride_index = stride * 2048 # = points per whitened interval

    if not (float(interval_index).is_integer() and float(stride_index).is_integer()):
        raise ValueError("interval * sample_frequency and stride * sample_frequency "
                         "must be int to get a valid index")

    interval_index = int(interval_index)
    stride_index = int(stride_index)

    # Opening the input and output HDF5 files
    with (
        h5py.File(data_path, "r") as f_in,
        h5py.File(segmented_path, "w") as f_out
    ):

        H1, L1 = f_in["H1"], f_in["L1"]

        # Creating the dataset that will be filled with whitened segments
        f_out.create_dataset(
            "whitened",
            shape = (2, 0, stride_index),
            # None lets us append as many intervals as needed
            maxshape = (2, None, stride_index),
            chunks = True
            )

        dset = f_out["whitened"]

        # Iterating on every timeseries in the input HDF5 files (at least 2 hours of strain data)
        for L_segment, H_segment in zip(H1.values(), L1.values(), strict = True):

            # Check if the strain timeseries is short enough to be contained in the GPU's VRAM.
            if len(L_segment) > max_length:
                warnings.warn(
                    f"Segment contains {len(L_segment)} points"
                    f" ({len(L_segment)/2048/3600:.2f} hours). "
                    "Your GPU might not be able to perform the computations.",
                    category = UserWarning,
                    stacklevel = 2   # Shows the line of code called by the function
                    )

            # Sending the data to the device after having read the HDF5 file
            # Using torch.as_tensor instead of torch.tensor:
            # the latter would create a numpy array and then copy it to another memory
            # space as a tensor, the first just wraps the numpy array, saving RAM space
            L_torch_segment = torch.as_tensor(L_segment[()], device=device)
            H_torch_segment = torch.as_tensor(H_segment[()], device=device)

            torch_segment = torch.stack([L_torch_segment, H_torch_segment], dim=0)
            # Shape: (2, len_segment) i.e. 2 channels

            # Divide in smaller intervals with stride (after checking if the integers are correct)
            torch_segmented = torch_segment.unfold(1, interval_index, stride_index)
            # Shape: (2, #intervals, interval * 2048)

            # Compute the psd
            psd = welch_psd(torch_segment, interval_index, device = device)

            # Apply whitening
            torch_whitened = whiten(torch_segmented,
                                    interval, psd,
                                    device = device,
                                    hz_cutoff = int(f_in.attrs["low_frequency_cutoff"]))

            # Crop borders so that the small segments' length is exactly stride * 2048
            cut_index = (interval_index - stride_index) // 2
            torch_whitened = torch_whitened[: , : , cut_index : -cut_index]

            # Appending whitened small segments to the dataset (needs to be numpy array in the cpu)
            dset_old_dim = dset.shape[1]
            dset_new_dim = dset_old_dim + torch_whitened.shape[1]
            dset.resize(dset_new_dim, axis=1)
            dset[ : , dset_old_dim : , : ] = torch_whitened.cpu().numpy()

            # Freeing GPU memory
            del (
                L_torch_segment,
                H_torch_segment,
                torch_segment,
                torch_segmented,
                torch_whitened,
                psd
            )

# ---------- Foreground data for training ----------
def get_tc(
    foreground_path : str,
    injections_path : str,
    random_interval : int | float
) -> tuple[dict, dict]:
    """
    Extract coalescence times (tc) and generate augmented times for each foreground segment.

    This function retrieves the actual injection times from the dataset in the injections
    HDF5 file and returns 2 dictionaries: the first one contains the true tc values
    for each timeseries in the foreground HDF5 file, while the second one contains 5 augmented
    tc values per injection, uniformly distributed within a `random_interval` time span around
    the true tc (used for data augmentation during training).

    Parameters
    ----------
    foreground_path : str
        Path to the HDF5 file containing the foreground signal.
    injections_path : str
        Path to the HDF5 file containing the injections parameters.
    random_interval : int | float
        Interval around the true values of tc to distribute the random times.

    Returns
    -------
    tuple[dict, dict]
        A tuple containing two dictionaries, both having the segment GPS start times
        as keys.
        - The first dictionary contains NumPy arrays with the true tc values.
        - The second dictionary contains NumPy arrays with the randomly generated tc values.
    """

    with (h5py.File(foreground_path, "r") as f_fore,
          h5py.File(injections_path, "r") as f_inj):

        # Creating a list with the foreground file attributes
        fore_attr = dict(f_fore.attrs.items())
        # inj_attr = dict(f_inj.attrs.items())

        # Retrieving padding on the edges
        padding_start = int(fore_attr["padding_start"])
        padding_end = int(fore_attr["padding_end"])

        # Selecting the coalescence times in our foreground data
        # from all the times contained in f_inj["tc"][()] for each segment
        global_tc_list = np.array(f_inj["tc"][()])
        tc_dict_random = {}
        tc_dict_original = {}

        for key in f_fore["H1"].keys():
            start_time = int(key)

            # Compute the segment duration
            segment_len = f_fore["H1"][key].shape[0]
            segment_duration = segment_len / 2048

            # Selecting the right times
            mask_start = global_tc_list >= (start_time + padding_start)
            mask_end = global_tc_list <= (start_time + segment_duration - padding_end)

            original_tc_list = global_tc_list[mask_start & mask_end]

            # Introducing randomness for efficient training: 5 random times in an interval
            # of `random_interval` seconds around the actual tc
            if len(original_tc_list) > 0:
                # Random offsets from the original times
                random_offsets = np.random.uniform(
                    -random_interval/2, random_interval/2, size = (len(original_tc_list), 5))
                # shape = (#tc, 5)
                                        # Broadcasting for sum
                new_tc_list = original_tc_list[:, None] + random_offsets
                new_tc_list = new_tc_list.flatten()

                # Checking again if the generated times are still in the right interval
                mask_start = new_tc_list >= (start_time + padding_start)
                mask_end = new_tc_list <= (start_time + segment_duration - padding_end)

                new_tc_list = new_tc_list[mask_start & mask_end]
                new_tc_list = np.sort(new_tc_list)
            else:
                new_tc_list = np.array([])

            # Saving the times in a dictionary with the same key as the segment
            tc_dict_random[key] = new_tc_list
            tc_dict_original[key] = original_tc_list

    return tc_dict_original, tc_dict_random

def foreground_segmentation(
    foreground_path : str,
    injections_path : str,
    segmented_foreground_path: str,
    interval : float | int = 6,
    stride : float | int = 2.5,
    default_time: float | int = 0.4
) -> None:
    """Extract, whiten and augment strain segments containing GWs injections.

    This function locates injected GWs signals in the foregound timeseries using their
    coalescence times. It extracts windows of duration `interval` around each coalescence
    time, applying data augmentation using the random generated times from the `get_tc`
    function. The segments are whitened cropped to a final length of `stride`, and saved
    to an output HDF5 file.

    Parameters
    ----------
    foreground_path : str
        Path to the HDF5 file containing the foreground signal.
    injections_path : str
        Path to the HDF5 file contraining the injections parameters.
    segmented_foreground_path : str
        Path to the output HDF5 file with the segmented and whitened signal.
    interval : float | int, optional
        Duration in seconds of the window used to compute the PSD, by default 6.
    stride : float | int, optional
        Stride of the windows in seconds, which also defines the output
        segment duration, by default 2.5.
    default_time: float | int, optional
        Safety margin in seconds from the segment borders where the coalescence
        time cannot be placed, by default 0.4
    """

    # Trying to use the GPU for the computations
    device = (
        torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
        )

    # Checking if interval and stride values can be converted into valid indexes
    interval_index = interval * 2048
    stride_index = stride * 2048 # = points per whitened interval

    if not (float(interval_index).is_integer() and float(stride_index).is_integer()):
        raise ValueError("interval * sample_frequency and stride * sample_frequency "
                         "must be int to get a valid index")

    interval_index = int(interval_index)
    stride_index = int(stride_index)

    # Checking if stride and default_time are compatible
    random_interval = stride - default_time
    if random_interval <= 0:
        raise ValueError("Please select a stride value higher than the default_time one.")

    # Generating the augmented tc dictionary
    _, tc_dict_random = get_tc(foreground_path, injections_path, random_interval)

    with (h5py.File(foreground_path, "r") as f_in,
          h5py.File(segmented_foreground_path, "w") as f_out):

        H1, L1 = f_in["H1"], f_in["L1"]

        # Creating the dataset that will be filled with whitened segments
        f_out.create_dataset(
            "whitened",
            shape = (2, 0, stride_index),
            # None lets us append as many intervals as needed
            maxshape = (2, None, stride_index),
            chunks = True
            )

        dset = f_out["whitened"]

        for key in H1.keys():

            # Computing the indexes corresponding to the times of coalescence
            tc_list_random = tc_dict_random[key]

            # If there are no coalescences in the segment, skip to the following one
            if len(tc_list_random) == 0:
                continue

            start_time = int(key)
            tc_index_list = (tc_list_random - start_time) * 2048
            tc_index_list = tc_index_list.astype(int)

            # Sending the data to the device after having read the hdf5 file
            L_torch_segment = torch.as_tensor(L1[key][()], device=device)
            H_torch_segment = torch.as_tensor(H1[key][()], device=device)

            torch_segment = torch.stack([L_torch_segment, H_torch_segment], dim=0)
            # Shape: (2, len_segment) i.e. 2 channels

            # Compute the psd
            psd = welch_psd(torch_segment, interval_index, device = device)

            # Selecting the correct intervals
            intervals = []
            for tc_index in tc_index_list:
                start_index = tc_index - interval_index//2
                end_index = tc_index + interval_index//2
                torch_interval = torch_segment[ : , start_index : end_index]
                intervals.append(torch_interval)

            torch_segmented = torch.stack(intervals, dim=1)
            # Shape: (2, len(tc_index_list), interval * 2048)

            # Apply whitening
            torch_whitened = whiten(torch_segmented,
                                    interval, psd,
                                    device = device,
                                    hz_cutoff = int(f_in.attrs["low_frequency_cutoff"]))

            # Crop borders so that the small segments' length is exacty stride * 2048
            cut_index = (interval_index - stride_index) // 2
            torch_whitened = torch_whitened[: , : , cut_index : -cut_index]

            # Appending whitened small segments to the dataset (need to be numpy array in the cpu)
            dset_old_dim = dset.shape[1]
            dset_new_dim =  dset_old_dim + torch_whitened.shape[1]
            dset.resize(dset_new_dim, axis=1)
            dset[ : , dset_old_dim : , : ] = torch_whitened.cpu().numpy()

            # Freeing GPU memory
            del (
                L_torch_segment,
                H_torch_segment,
                torch_segment,
                torch_segmented,
                torch_whitened,
                psd
            )
