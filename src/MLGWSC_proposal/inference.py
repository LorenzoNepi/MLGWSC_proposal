import gc

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from MLGWSC_proposal.data_manipulation import torch_segmentation
from MLGWSC_proposal.resnet import GW_ResNet


def model_inference(
    model: GW_ResNet,
    ts_path: str,
    segmented_ts_path: str,
    interval: float = 2.5,
    stride: float = 1.0,
    chunk_size: int = 1024,
    batch_size: int = 64,
    prob_threshold: float = 0.5,
    cluster_window: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Run `GW_ResNet` inference on continuous strain data.

    The input HDF5 timeseries is whitened and segmented using a sliding step equal to
    ``stride / 4``. Each window is evaluated by ``model``, positive candidates above
    ``prob_threshold`` are filtered, and nearby triggers are grouped via
    `cluster_detections()`.

    Parameters
    ----------
    model : GW_ResNet
        Trained neural network model outputting raw classification logits.
    ts_path : str
        Path to the input HDF5 file containing `H1` and `L1` strain datasets.
    segmented_ts_path : str
        Path where the intermediate whitened and segmented HDF5 file will be stored.
    interval : float, optional
        Total interval (in seconds) used for PSD estimation and whitening,
        by default 2.5.
    stride : float, optional
        Cropped window duration (in seconds) fed to the network (`stride * 2048` samples),
        by default is 1.0.
    chunk_size : int, optional
        Number of segments processed per batch on GPU during whitening, by default 1024.
    batch_size : int, optional
        Number of segments per batch during evaluation, by default 64.
    prob_threshold : float, optional
        Minimum probability required to retain a candidate detection before clustering,
        by default is 0.5.
    cluster_window : float, optional
        Maximum time separation (in seconds) between consecutive detections merged into the
        same event cluster, by default is 0.5.

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

    model.to(device)

    # Segmenting data and saving it segmented_ts_path
    torch_segmentation(
        data_path = ts_path,
        segmented_path = segmented_ts_path,
        interval = interval,
        stride = stride,
        chunk_size = chunk_size,
        max_intervals = None,
        inference = True
    )

    # We could have just computed the times list as
    # times = np.array([start_time + a * stride for a in range(data_whitened.shape[1])])
    # but to be more precise we can compute it like this:
    stride_index = round(stride * 2048)
    interval_index = round(interval * 2048)
    step_index = stride_index // 4
    step = step_index / 2048

    times = np.array([])
    with h5py.File(ts_path, "r") as f:
        for key in f["H1"]:
            #* Number of whitened intervals per >2h segment
            n_seg = (f["H1"][key].shape[0] - interval_index) // step_index + 1
            segment_times_list = float(key) + (interval - stride) / 2 + np.arange(n_seg) * step
            times = np.concatenate((times, segment_times_list))

    # Evaluating all the segments
    model.eval()
    logits = []

    # Recovering segmented strain data from the HDF5 files
    with h5py.File(segmented_ts_path, "r") as f:
        # Shape (2, # segments, stride * 2048)
        data_whitened = np.array(f["whitened"][()])

    # Reordering to shape (batch, channel, lenght) = (# segments, 2, stride * 2048)
    data_whitened = torch.from_numpy(data_whitened)
    data_whitened = data_whitened.permute(1, 0, 2)

    # Creating the dataloader
    inference_loader  = DataLoader(
        TensorDataset(data_whitened), batch_size = batch_size, shuffle = False, pin_memory=True
    )

    # Evaluating all the segments

    all_probs = []
    with torch.inference_mode():
        for (a,) in inference_loader:
            a = a.to(device, non_blocking=True)
            logits = model(a)

            # Computing the probability that each segment contains a signal
            probs = torch.sigmoid(logits).view(-1)
            all_probs.extend(probs.cpu().tolist())

    all_probs = np.array(all_probs)
    # Generating a mask for signal segments
    mask = all_probs > prob_threshold
    signal_times = times[mask]
    signal_probs = all_probs[mask]

    # Freeing GPU memory just to be sure
    gc.collect()
    torch.cuda.empty_cache()

    selected_times, selected_probs = cluster_detections(signal_times = signal_times,
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