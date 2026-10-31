import warnings

import h5py
import torch


def explore_data(
    background_path: str,
    foreground_path: str,
    injections_path: str
    ) -> None:
    """Function to print to the terminal relevant info about the background,
    foregroung and injections HDF5 files used during training.

    Parameters
    ----------
    background_path : str
        Path to the background file.
    foreground_path : str
        Path to the foreground file.
    injections_path : str
        Path to the injections file.
    """

    files_to_explore = {
        "BACKGROUND": background_path,
        "FOREGROUND": foreground_path
    }

    for name, path in files_to_explore.items():
        print(f"\n{name} HDF5 FILE\n")
        with h5py.File(path) as f:

            print(f"Root group keys: {f.keys()}")
            f_attr=dict(f.attrs.items())
            print("Root group attributes:")
            print(*f_attr.items(), sep="\n")

            H1_group = f["H1"]
            print(f"\nH1 group keys: {H1_group.keys()}")
            print(f"H1 group attributes: {H1_group.attrs.keys()}")

            L1_group = f["L1"]
            print(f"\nL1 group keys: {L1_group.keys()}")
            print(f"L1 group attributes: {L1_group.attrs.keys()}")

    print("\nINJECTIONS HDF5 FILE\n")

    with h5py.File(injections_path) as f:
        print(f"Root group keys: {f.keys()}")
        f_attr=dict(f.attrs.items())
        print("Root group attributes:")
        print(f.attrs.keys())

def torch_segmentation(
    background_path : str,
    segmented_background_path : str,
    max_lenght : float | int = 73728000,
    interval : float | int = 6,
    stride : float | int = 2.5
    ):

    points_per_interval = (interval - stride) * 2048 // 2
    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available() else "cpu"
    )

    with (
        h5py.File(background_path, "r") as f_in,
        h5py.File(segmented_background_path, "w") as f_out
    ):

        H1, L1 = f_in["H1"], f_in["L1"]

        # Creating the dataset that will be filled with whitened intervals
        f_out.create_dataset(
            "whitened",
            shape = (2, 0, points_per_interval),
            # None let us append as many intervals as needed
            maxshape = (2, None, points_per_interval),
            chunks = True
            )

        dset = f_out["whitened"]

        for L_segment, H_segment in zip(H1.values(), L1.values(), strict = True):
            # Check if the segment is short enough to be contained in the GPU's VRAM.
            if len(L_segment) > max_lenght:
                warnings.warn(
                    f"Segment contains {len(L_segment)} points"
                    "({len(L_segment)/2048/3600} hours). "
                    "Your GPU might not be able to perform the computations.",
                    category = UserWarning,
                    stacklevel = 2   # Shows the line of code called by the function
                    )

            L_torch_segment = torch.tensor(L_segment[()]).to(device)
            H_torch_segment = torch.tensor(H_segment[()]).to(device)

            torch_segment = torch.stack([L_torch_segment, H_torch_segment], dim=0)
            # Shape: (2, len_segment) i.e. 2 channels

            # Divide in smaller intervals with stride (after checking if the integers are correct)
            interval_index = interval * 2048
            stride_index = stride * 2048

            if not (float(interval_index).is_integer() and float(stride_index).is_integer()):
                raise ValueError("interval*sample_frequency and stride*sample_frequency"
                                 "must be int to get a valid index")

            interval_index = int(interval_index)
            stride_index = int(stride_index)

            torch_segmented = torch_segment.unfold(1, interval_index, stride_index)
            # Shape: (2, #intervals, interval * 2048)

            ## Compute the psd

            ## Apply whitening

            ## Temporary tensor
            torch_whitened = torch_segmented.clone()

            # Crop borders
            cut_index = (interval - stride) * 2048 // 2
            torch_whitened = torch_whitened[: , : , cut_index : -cut_index]

            # Appending whitened intervals to the dataset (need to be numpy array in the cpu)
            dset_old_dim = dset.shape[1]
            dset_new_dim =  dset_old_dim + torch_whitened.shape[1]
            dset.resize(dset_new_dim, axis=1)
            dset[ : , dset_old_dim : , : ] = torch_whitened.cpu().numpy()

            # Freeing GPU memory
            del L_torch_segment, H_torch_segment, torch_segment, torch_segmented, torch_whitened
