import gc

import h5py
import numpy as np
import torch

from MLGWSC_proposal.data_manipulation import welch_psd, whiten
from MLGWSC_proposal.resnet import GW_ResNet


def model_inference(
    model : GW_ResNet,
    ts_path : str,
    interval : float = 2.5,
    stride : float = 1.0,
    chunk_size : int = 1024,
    batch_size : int = 2048,
    prob_threshold : float = 0.5,
    cluster_window : float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """Run `GW_ResNet` inference on continuous strain data returning coalescence times.

    The input HDF5 timeseries is whitened and segmented using a sliding step equal to
    ``stride / 4``. Each window is evaluated by ``model``, positive candidates above
    ``prob_threshold`` are filtered, and nearby triggers are grouped via
    `cluster_detections()`.

    Parameters
    ----------
    model : GW_ResNet
        Trained neural network model outputting raw classification logits.
    ts_path : str
        Path to the input HDF5 file containing raw `H1` and `L1` strain datasets.
    interval : float, optional
        Total interval (in seconds) used for PSD estimation and whitening,
        by default 2.5.
    stride : float, optional
        Cropped window duration (in seconds) fed to the network (`stride * 2048` samples),
        by default is 1.0.
    chunk_size : int, optional
        Number of segments processed per batch on GPU during whitening, by default 1024.
    batch_size : int, optional
        Number of segments per batch during evaluation, by default 2048.
    prob_threshold : float, optional
        Minimum probability required to retain a candidate detection before clustering,
        by default 0.5.
    cluster_window : float, optional
        Maximum time separation (in seconds) between consecutive detections merged into the
        same event cluster, by default 0.5.
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple ``(selected_times, selected_probs)`` containing the estimated coalescence GPS
        timestamps and peak detection probabilities for each clustered candidate.
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
    step_index = stride_index//4 # Used only for inference
    step = step_index / 2048

    model.to(device)

    # Evaluating all the segments
    model.eval()
    all_probs = []
    times = []

    # Opening the input HDF5 file
    with h5py.File(ts_path, "r") as f_in:

        H1, L1 = f_in["H1"], f_in["L1"]

        for key in H1:
            #* Number of whitened intervals per >2h segment
            # We could have just computed the times list as
            # times = np.array([start_time + a * stride for a in range(data_whitened.shape[1])])
            # but to be more precise we can compute it like this:
            n_seg = (H1[key].shape[0] - interval_index) // step_index + 1
            segment_times_list = float(key) + (interval - stride) / 2 + np.arange(n_seg) * step
            times.extend(segment_times_list)

            # Segmenting the data
            L_torch_segment = torch.as_tensor(
                L1[key][()], dtype=torch.float32, device=device) * 1e21
            H_torch_segment = torch.as_tensor(
                H1[key][()], dtype=torch.float32, device=device) * 1e21

            torch_segment = torch.stack([L_torch_segment, H_torch_segment], dim=0)
            # Shape: (2, len_segment) i.e. 2 channels

            torch_segmented = torch_segment.unfold(1, interval_index, step_index)
            # Shape: (2, #intervals, interval_index)

            # Compute the psd on the full >=2h block
            psd = welch_psd(
                torch_segment, interval_index, device = device, chunk_size = chunk_size
            )

            # Apply whitening only on the subsampled intervals
            torch_whitened = whiten(torch_segmented,
                                    interval, psd,
                                    device = device,
                                    hz_cutoff = round(f_in.attrs["low_frequency_cutoff"]),
                                    chunk_size = chunk_size)

            # Crop borders so that the small segments' length is exactly stride * 2048
            cut_index = (interval_index - stride_index) // 2
            torch_whitened = torch_whitened[: , : , cut_index : -cut_index]

            # Reordering to shape (batch, channel, length) = (# segments, 2, stride * 2048)
            torch_whitened = torch_whitened.permute(1, 0, 2)

            # Evaluating all the segments directly on GPU
            # (batching by hand, DataLoaders and Dataset would introduce a useless overhead)
            with torch.inference_mode():
                for start in range(0, torch_whitened.shape[0], batch_size):
                    batch = torch_whitened[start : start + batch_size]
                    # Computing the probability that each segment contains a signal
                    logits = model(batch)
                    probs = torch.sigmoid(logits).view(-1)
                    all_probs.extend(probs.cpu().tolist())

            # Freeing GPU memory
            del (
                L_torch_segment,
                H_torch_segment,
                torch_segment,
                torch_segmented,
                torch_whitened,
                batch,
                psd
                )

            gc.collect()
            torch.cuda.empty_cache()

    times = np.array(times)
    all_probs = np.array(all_probs)
    # Generating a mask for signal segments
    mask = all_probs > prob_threshold
    signal_times = times[mask]
    signal_probs = all_probs[mask]

    selected_times, selected_probs = cluster_detections(
        signal_times = signal_times,
        signal_probs = signal_probs,
        cluster_window = cluster_window
        )

    return selected_times, selected_probs

def cluster_detections(
    signal_times: np.ndarray,
    signal_probs: np.ndarray,
    cluster_window: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Group consecutive trigger timestamps within ``cluster_window`` seconds and keep the peak.

    Parameters
    ----------
    signal_times : np.ndarray
        1D array of GPS timestamps corresponding to candidate detections.
    signal_probs : np.ndarray
        1D array of detection probabilities associated with ``signal_times``.
    cluster_window : float, optional
        Maximum time gap (in seconds) between consecutive triggers belonging to the same cluster.
        Default is 0.5.
    """
    if len(signal_times) == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float32)

    # Ensuring chronological ordering (already satisfied if keys in ts_path are sorted)
    order = np.argsort(signal_times)
    times = signal_times[order]
    probs = signal_probs[order]

    # Splitting into clusters wherever the gap between two consecutive triggers
    # exceeds cluster_window:
    # - np.diff computes the difference between each time and the previous one (len - 1)
    # - np.flatnonzero returns indexes where mask np.diff(times) > cluster_window is True
    # - + 1 converts the mask in the start index for each cluster
    split_indices = np.flatnonzero(np.diff(times) > cluster_window) + 1

    # Completing the clusters: concateneting 0 in the beginning and len(times) in the end.
    # Now we have two arrays like
    # cluster_starts = [0, start_cluster2, start_cluster3, ...]
    # cluster_ends = [end_cluster1, end_cluster2, ..., len(times)]
    # where end_cluster1 == start_cluster2
    cluster_starts = np.concatenate((np.array([0]), split_indices))
    cluster_ends = np.concatenate((split_indices, np.array([len(times)])))

    # Memory space for the output arrays
    selected_times = np.empty(len(cluster_starts), dtype=times.dtype)
    selected_probs = np.empty(len(cluster_starts), dtype=probs.dtype)

    # Finding the position of the peak and selecting the right times
    for idx, (start, end) in enumerate(zip(cluster_starts, cluster_ends, strict = True)):
        best_local = int(np.argmax(probs[start:end]))
        selected_times[idx] = times[start + best_local]
        selected_probs[idx] = probs[start + best_local]

    return selected_times, selected_probs

def mlgwsc1_output(
    selected_times: np.ndarray,
    selected_probs: np.ndarray,
    output_path: str,
    cluster_window: float = 0.5,
) -> None:
    """Save clustered detections in the official MLGWSC-1 HDF5 submission format.

    Creates an HDF5 file containing three 1D datasets of equal length: ``time`` (estimated
    coalescence GPS timestamps), ``stat`` (ranking statistic / detection probability), and
    ``var`` (timing uncertainty tolerance around each predicted event).

    Parameters
    ----------
    selected_times : np.ndarray
        1D array of predicted coalescence GPS timestamps.
    selected_probs : np.ndarray
        1D array of ranking statistics (probabilities) associated with ``selected_times``.
    output_path : str
        Destination path for the output HDF5 file.
    cluster_window : float, optional
        Constant timing tolerance (in seconds) stored in the ``var`` dataset. Default is 0.5.
    """
    vars = np.ones_like(selected_times) * cluster_window
    with h5py.File(output_path, "w") as f_out:
        f_out["time"] = selected_times
        f_out["stat"] = selected_probs
        f_out["var"] = vars

    print(f"Output HDF5 file generated successfully {output_path}")
