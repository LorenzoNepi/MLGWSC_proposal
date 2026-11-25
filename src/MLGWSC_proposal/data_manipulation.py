import gc
import warnings

import h5py
import numpy as np
import torch


# ---------- Data exploration ----------
def explore_data(
    background_path : str,
    foreground_path : str,
    injections_path : str
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
    torch_segment : torch.Tensor,
    interval_index : int,
    device : torch.device | str = "cpu",
    chunk_size : int = 512
) -> torch.Tensor:
    """Compute the Power Spectral Density (PSD) on a strain segment using the Welch method.

    Parameters
    ----------
    torch_segment : torch.Tensor
        Tensor containing the strain data. Expected shape is `(num_channels, segment_length)`.
    interval_index : int
        Number of sampled points per interval.
    device : torch.device | str, optional
        Device that will execute the computations, by default `"cpu"`.
    chunk_size : int, optional
        Number of segments processed simultaneously to prevent VRAM saturation, by default 512.

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

    # Chunking the FFT computation to save VRAM space. Without it the complex
    # torch.fft.rfft operation saturates a 16 GB VRAM almonts instantly and
    # OutOfMemoryError is raised
    num_windows = welch_segmented.shape[1]
    n_fr = interval_index // 2 + 1
    psd_sum = torch.zeros(
        (torch_segment.shape[0], n_fr), dtype = torch_segment.dtype, device = device)

    # Iterating over the chunks
    for i in range(0, num_windows, chunk_size):
        # torch will not raise an IndexOutOfBoundException in the last iteration,
        # it will just take a smaller segment
        chunk = welch_segmented[ : , i : i + chunk_size, : ]
        chunk = chunk * hann_window

        # Computing the squared module of the 1-D Fourier transform with normalization ("ortho")
        # Using norm="ortho" preserves magnitudes from artificially exploding or shrinking.
        fourier_chunk = torch.fft.rfft(chunk, dim = 2, norm = "ortho")
        fourier_chunk = torch.abs(fourier_chunk)**2
        psd_sum += fourier_chunk.sum(dim = 1)
        # Shape: (2, chunk_size, interval_index//2 + 1) due to torch.fft.rfft

    # Now that we have the sum we can just compute the mean by dividing for the number of windows
    psd_welch = psd_sum / num_windows
    # Shape: (2, interval_index//2 + 1) due to torch.fft.rfft

    return psd_welch


def whiten(
    torch_segmented : torch.Tensor,
    interval : float,
    psd : torch.Tensor,
    device : torch.device | str = "cpu",
    hz_cutoff : float = 20,
    chunk_size : int = 512
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
    chunk_size : int, optional
        Number of segments processed simultaneously to prevent VRAM saturation, by default 512.

    Returns
    -------
    torch.Tensor
        Tensor containing the whitened segments (shape will be equal to `torch_segmented` shape.)
    """

    # Using a Hann window to reduce border effects
    hann_window = torch.hann_window(
        round(interval * 2048), # Using round() instad of int() for numerical stability
        dtype = torch_segmented.dtype,
        device = device
        )

    # Computing the inverse ASD (adding a relative floor for stability)
    # We use torch.clamp with a relative floor (1e-20 * max_value_in_channel) instead of
    # a fixed value. This prevents dividing by zero while respecting the natural
    # extremely low magnitudes of gravitational wave signals (e.g., 1e-22).
    inv_asd = 1.0 / torch.sqrt(torch.clamp(psd, min=psd.amax(dim=1, keepdim=True) * 1e-20))

    # Applying frequency cut-off
    # delta_f = 1 / interval => cutoff_index = hz_cutoff / delta_f = interval * hz_cutoff
    cutoff_index = round(interval * hz_cutoff)
    inv_asd[:, :cutoff_index] = 0.0
    inv_asd = inv_asd.unsqueeze(1)

    num_segments = torch_segmented.shape[1]
    whitened_chunks = []

    # Chunking the inverse FFT computation to save VRAM space.
    for i in range(0, num_segments, chunk_size):
        chunk = torch_segmented[ : , i : i+chunk_size, : ]
        chunk = chunk * hann_window

        # Whitening the chunk (frequency domain)
        fourier_chunk = torch.fft.rfft(chunk, dim = 2, norm = "ortho")
        whitened_fourier = fourier_chunk * inv_asd

        # Back to time domain with inverse FFT
        chunk_whitened = torch.fft.irfft(
            whitened_fourier,
            n = round(interval * 2048),
            dim = 2,
            norm = "ortho"
        )
        whitened_chunks.append(chunk_whitened)

    # Reassemble chunks along the segment dimension
    return torch.cat(whitened_chunks, dim = 1)
    # Shape(2, num_segments, interval_index)


def torch_segmentation(
    data_path : str,
    segmented_path : str,
    max_length : float | int = 73728000,
    interval : float | int = 6,
    stride : float | int = 2.5,
    chunk_size : int = 512,
    max_intervals : int | None = None,
) -> None:
    """Divide the strain data into whitened, fixed-length segments.

    The raw strain timeseries is segmented using a sliding window of duration `interval`
    with step size `stride` to compute the PSD. The central portion of each whitened
    window (of length `stride`) is preserved and saved to an HDF5 file. The output dataset
    will have shape (C, S, `stride * 2048`) where
    - C is number of channels in the input data;
    - S is the number of segments, which can be computed as
    (total_duration - interval) // stride + 1
    - `stride * 2048` is the number of sampled points per timeseries (with a 2048 Hz
    sampling rate).
    This procedure is used for generating the background training dataset.

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
    chunk_size : int, optional
        Number of segments processed simultaneously during FFT operations
        to prevent VRAM saturation, by default 512.
    max_intervals : int | None, optional
        Maximum number of intervals to whiten. This parameter is necessary for
        background segmentation of long strain timeseries (in the order of weeks),
        so the GPU VRAM is not saturated during computation. If a valid value
        for `max_intervals` is passed, a number of intervals proportional to
        each segment length is added to the output file in order to speed up the
        whitening process.
    """

    # Trying to use the GPU for the computations
    device = (
        torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    )

    # Checking if a valid value was passed for max_intervals
    if max_intervals is not None and max_intervals <= 0:
        raise ValueError("max_intervals must be a positive integer or None.")


    # Checking if interval and stride values can be converted into valid indexes
    interval_index = interval * 2048
    stride_index = stride * 2048 # = points per whitened interval

    if not (float(interval_index).is_integer() and float(stride_index).is_integer()):
        raise ValueError("interval * sample_frequency and stride * sample_frequency "
                         "must be int to get a valid index")

    interval_index = round(interval_index)
    stride_index = round(stride_index)

    # Opening the input and output HDF5 files
    with (
        h5py.File(data_path, "r") as f_in,
        h5py.File(segmented_path, "w") as f_out
    ):

        H1, L1 = f_in["H1"], f_in["L1"]

        # Counting available intervals per segment without loading arrays into RAM
        intervals_per_segment = []
        for key in H1:
            n_samples = H1[key].shape[0]
            intervals_per_segment.append(max(0, (n_samples - interval_index) // stride_index + 1))

        total_intervals = sum(intervals_per_segment)

        # Counting how many background intervals we must take per segment (quotas)
        if max_intervals is not None and max_intervals < total_intervals:
            raw_quotas = [max_intervals * (n / total_intervals) for n in intervals_per_segment]
            quotas = [int(q) for q in raw_quotas]
            remainder = max_intervals - sum(quotas)
            # Distribute the remaining intervals to the segments with the largest decimal parts
            if remainder > 0:
                frac_order = np.argsort([q - int(q) for q in raw_quotas])[::-1]
                for i in frac_order[:remainder]:
                    quotas[i] += 1
        else:
            quotas = intervals_per_segment

        # Questo è uno strumentopolo che ci tornerà utile in seguito
        rng = torch.Generator(device = device).manual_seed(42)

        # Creating the dataset that will be filled with whitened segments
        f_out.create_dataset(
            "whitened",
            shape = (2, 0, stride_index),
            # None lets us append as many intervals as needed
            maxshape = (2, None, stride_index),
            # 256 is arbitrary, it but we should specify a value since letting
            # h5py guess will slow execution
            chunks = (2, 256, stride_index),
            dtype = "f4"
            )

        dset = f_out["whitened"]

        # Iterating on every timeseries in the input HDF5 files (at least 2 hours of strain data)
        for key, quota in zip(H1, quotas, strict = True):
            if quota == 0: # Should not happen, but just in case
                continue

            # Check if the strain timeseries is short enough to be contained in the GPU's VRAM.
            if len(L1[key]) > max_length:
                warnings.warn(
                    f"Segment contains {len(L1[key])} points"
                    f" ({len(L1[key])/2048/3600:.2f} hours). "
                    "Your GPU might not be able to perform the computations.",
                    category = UserWarning,
                    stacklevel = 2   # Shows the line of code called by the function
                    )

            # Sending the data to the device after having read the HDF5 file
            # Using torch.as_tensor instead of torch.tensor:
            # the latter would create a numpy array and then copy it to another memory
            # space as a tensor, the first just wraps the numpy array, saving RAM space.
            # Recasting to float 32 and multiplying by a factor to address the small
            # scales of the strain values
            L_torch_segment = torch.as_tensor(
                L1[key][()], dtype=torch.float32, device=device) * 1e21
            H_torch_segment = torch.as_tensor(
                H1[key][()], dtype=torch.float32, device=device) * 1e21

            torch_segment = torch.stack([L_torch_segment, H_torch_segment], dim=0)
            # Shape: (2, len_segment) i.e. 2 channels

            # Divide in smaller intervals with stride (after checking if the integers are correct)
            torch_segmented = torch_segment.unfold(1, interval_index, stride_index)
            # Shape: (2, # intervals, interval_index)

            # Compute the psd on the full >=2h block
            psd = welch_psd(
                torch_segment, interval_index, device = device, chunk_size = chunk_size
            )

            # Randomly subsample intervals before whitening to save FFT time and VRAM (if quota)
            num_available = torch_segmented.shape[1]
            if quota < num_available:
                perm = torch.randperm(num_available, generator = rng, device = device)[ : quota]
                perm, _ = torch.sort(perm)
                torch_segmented = torch_segmented[ : , perm, : ]

            # Apply whitening only on the subsampled intervals
            torch_whitened = whiten(torch_segmented,
                                    interval, psd,
                                    device = device,
                                    hz_cutoff = round(f_in.attrs["low_frequency_cutoff"]),
                                    chunk_size = chunk_size)

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
            gc.collect()
            torch.cuda.empty_cache()

# ---------- Foreground data for training ----------
def get_tc(
    foreground_path : str,
    injections_path : str,
    random_interval : int | float
) -> tuple[dict, dict, dict]:
    """
    Extract coalescence times (tc) and generate shifted times for each foreground segment.

    This function retrieves the actual injection times from the dataset in the injections
    HDF5 file and returns three dictionaries. The first contains the true tc values
    for each timeseries in the foreground HDF5 file. The second contains one randomly
    shifted tc value per injection, uniformly distributed within a `random_interval`
    time span around the true tc (to avoid positional bias without introducing data
    leakage across splits). The third contains the approximate SNR values computed as:

    $$SNR = \\frac{1261 \\text{ Mpc}}{D} \\cdot
    \\left(\\frac{\\mathcal{M}_{c}}{M_{\\odot}}\\right)^{5/7} [0.7 + 0.3 \\cos(2i)]$$

    where $D$ is the distance, $\\mathcal{M}_{c}$ is the chirp mass, and $i$ is the inclination.
    All these parameters are retrieved directly from the injections HDF5 file.

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
    tuple[dict, dict, dict]
        A tuple containing three dictionaries, all using the segment GPS start times as keys:
        - The first contains NumPy arrays with the true tc values.
        - The second contains NumPy arrays with the randomly shifted tc values.
        - The third contains NumPy arrays with the approximated SNR for each generated tc.
    """

    with (h5py.File(foreground_path, "r") as f_fore,
          h5py.File(injections_path, "r") as f_inj):

        # Creating a list with the foreground file attributes
        fore_attr = dict(f_fore.attrs.items())
        # inj_attr = dict(f_inj.attrs.items())

        # Retrieving padding on the edges
        padding_start = round(fore_attr["padding_start"])
        padding_end = round(fore_attr["padding_end"])

        # Selecting the coalescence times in our foreground data
        # from all the times contained in f_inj["tc"][()] for each segment
        global_tc_list = np.array(f_inj["tc"][()])

        # Selecting the necessary injection parameters to compute the SNR
        global_dis_list = np.array(f_inj["distance"][()])
        global_mchirp_list = np.array(f_inj["mchirp"][()])
        global_i_list = np.array(f_inj["inclination"][()])

        tc_dict_random = {}
        tc_dict_original = {}
        snr_dict = {}

        # Initializing the random number generator once for reproducibility
        rng = np.random.default_rng(seed=42)

        for key in f_fore["H1"]:
            start_time = int(key)

            # Compute the segment duration
            segment_len = f_fore["H1"][key].shape[0]
            segment_duration = segment_len / 2048

            # Selecting the right times
            mask_start = global_tc_list >= (start_time + padding_start)
            mask_end = global_tc_list <= (start_time + segment_duration - padding_end)

            original_tc_list = global_tc_list[mask_start & mask_end]

            # Selecting the right injection parameters values to compute the SNR
            dis_list = global_dis_list[mask_start & mask_end]
            mchirp_list = global_mchirp_list[mask_start & mask_end]
            i_list = global_i_list[mask_start & mask_end]

            # Computing the SNR for each injection (approximation)
            snr_list = (1261 / dis_list) * (mchirp_list)**(5/7) * (0.7 + 0.3 * np.cos(2 * i_list))

            # Introducing a random offset per injection within `random_interval`
            # around the actual tc so that the merger is not always centered in the window
            if len(original_tc_list) > 0:
                # Random offsets from the original times (one per injection)
                random_offsets = rng.uniform(
                    -random_interval/2, random_interval/2, size=len(original_tc_list)
                )
                new_tc_list = original_tc_list + random_offsets
                new_snr_list = snr_list.copy()

                # Checking again if the generated times are still in the right interval
                mask_start = new_tc_list >= (start_time + padding_start)
                mask_end = new_tc_list <= (start_time + segment_duration - padding_end)

                new_tc_list = new_tc_list[mask_start & mask_end]
                new_snr_list = new_snr_list[mask_start & mask_end]

                # We need to sort the times, but we must sort the SNR array in the exact same order:
                # np.argsort return the indexes used to sort a certain array
                # e.g. np.argsort(np.array(10, 30, 20)) = [0, 2, 1]
                sort_indexes = np.argsort(new_tc_list)
                new_tc_list = new_tc_list[sort_indexes]
                new_snr_list = new_snr_list[sort_indexes].astype(np.float16)
            else:
                new_tc_list = np.array([])
                new_snr_list = np.array([], dtype=np.float16)

            # Saving the times in a dictionary with the same key as the segment
            tc_dict_random[key] = new_tc_list
            tc_dict_original[key] = original_tc_list
            snr_dict[key] = new_snr_list

    return tc_dict_original, tc_dict_random, snr_dict

def foreground_segmentation(
    foreground_path : str,
    injections_path : str,
    segmented_foreground_path: str,
    interval : float | int = 6,
    stride : float | int = 2.5,
    default_time : float | int = 0.4,
    chunk_size : int = 512
) -> None:
    """Extract, whiten and augment strain segments containing GWs injections.

    This function locates injected GWs signals in the foregound timeseries using their
    coalescence times. It extracts windows of duration `interval` around each coalescence
    time, using the random generated times from the `get_tc`
    function. The segments are whitened cropped to a final length of `stride`, and saved
    to an output HDF5 file.

    The output dataset will have shape `(C, S, stride * 2048)` where:
    - `C` is the number of channels in the input data;
    - `S` is the total number of segments across all injections i.e.
    the total number of coalescence times generated by `get_tc`;
    - `stride * 2048` is the number of sampled points per output segment
    (a 2048 Hz sampling rate is assumed).
    In the same output file, another dataset will be saved containing the SNR values
    for each coalescence (also computed with `get_tc`).

    Parameters
    ----------
    foreground_path : str
        Path to the HDF5 file containing the foreground signal.
    injections_path : str
        Path to the HDF5 file contraining the injections parameters.
    segmented_foreground_path : str
        Path to the output HDF5 file with the segmented and whitened signal and SNR.
    interval : float | int, optional
        Duration in seconds of the window used to compute the PSD, by default 6.
    stride : float | int, optional
        Stride of the windows in seconds, which also defines the output
        segment duration, by default 2.5.
    default_time: float | int, optional
        Safety margin in seconds from the segment borders where the coalescence
        time cannot be placed, by default 0.4
    chunk_size : int, optional
        Number of segments processed simultaneously during FFT operations
        to prevent VRAM saturation, by default 512.
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

    interval_index = round(interval_index)
    stride_index = round(stride_index)

    # Checking if stride and default_time are compatible
    random_interval = stride - default_time
    if random_interval <= 0:
        raise ValueError("Please select a stride value higher than the default_time one.")

    # Generating the augmented tc dictionary
    _, tc_dict_random, snr_dict = get_tc(foreground_path, injections_path, random_interval)

    with (h5py.File(foreground_path, "r") as f_in,
          h5py.File(segmented_foreground_path, "w") as f_out):

        H1, L1 = f_in["H1"], f_in["L1"]

        # Creating the dataset that will be filled with whitened segments
        f_out.create_dataset(
            "whitened",
            shape = (2, 0, stride_index),
            # None lets us append as many intervals as needed
            maxshape = (2, None, stride_index),
            chunks = (2, 256, stride_index),
            dtype = "f4" # float32
            )

        dset = f_out["whitened"]

        # Creating the dataset that will contain the SNR value per segment
        f_out.create_dataset(
            "snr",
            shape = (0,),
            # None lets us append as many intervals as needed
            maxshape = (None,),
            chunks = (256,),
            dtype = "f4"
            )

        snr_dset = f_out["snr"]

        for key in H1:

            # Computing the indexes corresponding to the times of coalescence
            tc_list_random = tc_dict_random[key]

            # If there are no coalescences in the segment, skip to the following one
            if len(tc_list_random) == 0:
                continue

            start_time = int(key)
            tc_index_list = (tc_list_random - start_time) * 2048
            tc_index_list = tc_index_list.astype(int)

            # Sending the data to the device after having read the hdf5 file
            L_torch_segment = torch.as_tensor(
                L1[key][()], dtype=torch.float32, device=device) * 1e21
            H_torch_segment = torch.as_tensor(
                H1[key][()], dtype=torch.float32, device=device) * 1e21

            torch_segment = torch.stack([L_torch_segment, H_torch_segment], dim=0)
            # Shape: (2, len_segment) i.e. 2 channels

            # Compute the psd
            psd = welch_psd(
                torch_segment, interval_index, device = device, chunk_size = chunk_size
            )

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
                                    hz_cutoff = round(f_in.attrs["low_frequency_cutoff"]),
                                    chunk_size = chunk_size)

            # Crop borders so that the small segments' length is exacty stride * 2048
            cut_index = (interval_index - stride_index) // 2
            # Shape: (2, len(tc_index_list), stride * 2048)
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
                intervals,
                torch_segmented,
                torch_whitened,
                psd
            )
            gc.collect()
            torch.cuda.empty_cache()

            # Saving the SNR value in the corresponding dataset
            snr_array = np.array(snr_dict[key])
            snr_dset_old_dim = snr_dset.shape[0]
            snr_dset_new_dim = snr_dset_old_dim + len(snr_array)
            snr_dset.resize(snr_dset_new_dim, axis = 0)
            snr_dset[snr_dset_old_dim : ] = snr_array
