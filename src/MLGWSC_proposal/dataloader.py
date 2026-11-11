import math

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class Foreground_Dataset(Dataset):
    """Custom PyTorch Dataset for whitened foreground strain segments.

    This dataset contains foreground whitened segments and automatically assigns them
    a label of `1` (signal) to each of them. It also allows for the implementation
    of curriculum learning during training by ordering the segments by SNR (if
    `snr` is defined) and letting the user iterate only on the necessary portion
    of data.

    Parameters
    ----------
    X : torch.Tensor
        A 3D tensor containing the whitened foreground segments.
        Expected shape: `(num_segments, num_channels, sequence_length)`.
    snr_split : float, optional
        The fraction of the dataset to iter on, starting from the first element,
        by default 1.0 (100% of the data). Must be in the interval (0, 1].
    snr : torch.Tensor | None, optional
        A 1D tensor containing the approximated SNR values corresponding to
        each segment in `X`, by default None.
    """
    def __init__(
        self, X : torch.Tensor, snr_split : float = 1.0, snr : torch.Tensor | None = None
    ) -> None:
        super().__init__()

        if snr is not None:
            # Sorting the training tensor by SNR value
            sort_indexes = torch.argsort(snr, descending=True)
            self.X = X[sort_indexes]
            self.snr = snr[sort_indexes]
        else:
            self.X = X
            self.snr = None
                                # For some reason Pytorch wants a tuple here
        self.y = torch.ones(size=(X.shape[0],))
        self.snr_split = snr_split

    def __getitem__(self, index : int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X_iter[index], self.y_iter[index]

    def __len__(self) -> int:
        return int(self.X_iter.shape[0])

    @property
    def snr_split(self) -> float:
        return self._snr_split

    @snr_split.setter
    def snr_split(self, new_snr_split : float) -> None:

        if not (0 < new_snr_split <= 1):
            raise ValueError("snr_split must be in ]0, 1]")

        # Implementing the split by SNR value
        self._snr_split = new_snr_split
        split_index = int(self.X.shape[0] * new_snr_split)

        self.X_iter = self.X[ : split_index, : , : ]
        self.y_iter = self.y[ : split_index]


class Background_Dataset(Dataset):
    """Custom PyTorch Dataset for whitened background strain segments.

    This dataset manages background strain tensors (noise only)
    and automatically assigns them a label of `0` (background).

    Parameters
    ----------
    X : torch.Tensor
        A 3D tensor containing the whitened background segments.
        Expected shape: `(num_segments, num_channels, sequence_length)`.
    """
    def __init__(self, X : torch.Tensor) -> None:
        super().__init__()
        self.X = X
        self.y = torch.zeros(size=(X.shape[0],))

    def __getitem__(self, index : int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[index], self.y[index]

    def __len__(self) -> int:
        return int(self.X.shape[0])


def generate_back_dataloaders(
    segmented_background_path: str,
    validation_split: list[float] | None = None,
    batch_size: int = 256
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Generate PyTorch DataLoaders for the background dataset.

    This function reads the whitened background segments from an HDF5 file,
    shuffles them, and splits them into training, validation, and test sets
    according to the specified proportions. It then wraps these datasets into
    PyTorch DataLoaders.

    Parameters
    ----------
    segmented_background_path : str
        Path to the HDF5 file containing the segmented and whitened background data.
    validation_split : list[float] | None, optional
        A list of three floats representing the fraction of data to allocate
        for training, validation, and testing, respectively.
        Must sum to 1.0. By default [0.7, 0.2, 0.1].
    batch_size : int, optional
        The number of samples per batch to load, by default 256.

    Returns
    -------
    tuple[DataLoader, DataLoader, DataLoader]
        A tuple containing three DataLoaders:
        - back_train_loader (shuffled)
        - back_val_loader (not shuffled)
        - back_test_loader (not shuffled)
    """
    if validation_split is None:
        validation_split = [0.7, 0.2, 0.1]
    # Checking:
    cond1 = not math.isclose(sum(validation_split), 1.0, rel_tol=1e-9, abs_tol=1e-9)
    cond2 = len(validation_split) != 3
    cond3 = any(i<0 for i in validation_split)
    cond = cond1 or cond2 or cond3

    if cond:
        raise ValueError("validation_split must be a list of 3 positive numbers between 0 and 1"
                         "with sum 1, for example [0.7, 0.2, 0.1] or [0.6, 0.4, 0.0].")

    # Recovering segmented background data from the HDF5 files
    with h5py.File(segmented_background_path, "r") as f:
        # Shape (2, # segments, stride * 2048)
        back_whitened = np.array(f["whitened"][()])

    # Reordering to shape (batch, channel, lenght) = (# segments, 2, stride * 2048)
    back_whitened = torch.from_numpy(back_whitened)
    back_whitened = back_whitened.permute(1, 0, 2)

    # Random shuffling of the indexes to shuffle the background data
    num_back = back_whitened.shape[0]
    back_perm = torch.randperm(num_back)

    # Splitting background training, validation and test tensor
    train_end = round(num_back * validation_split[0])
    val_end = round(num_back * (validation_split[0] + validation_split[1]))

    train_indexes = back_perm[ : train_end]
    val_indexes = back_perm[train_end : val_end]
    test_indexes = back_perm[val_end : ]

    back_train = back_whitened[train_indexes]
    back_val = back_whitened[val_indexes]
    back_test = back_whitened[test_indexes]

    # Creating datasets
    back_train_dset = Background_Dataset(back_train)
    back_val_dset = Background_Dataset(back_val)
    back_test_dset = Background_Dataset(back_test)

    back_train_loader = DataLoader(
        back_train_dset, batch_size = batch_size, shuffle = True, pin_memory = True
        )
    back_val_loader = DataLoader(
        back_val_dset, batch_size = batch_size, shuffle = False, pin_memory = True
        )
    back_test_loader = DataLoader(
        back_test_dset, batch_size = batch_size, shuffle = False, pin_memory = True
        )

    return back_train_loader, back_val_loader, back_test_loader


def generate_fore_dataloaders(
    segmented_foreground_path: str,
    validation_split: list[float] | None = None,
    batch_size: int = 256,
    snr_split: float = 1.0
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Generate PyTorch DataLoaders for the foreground dataset.

    This function reads the whitened foreground segments and their corresponding
    SNR values from an HDF5 file, shuffles them, and splits them into training,
    validation, and test sets according to the specified proportions. It wraps
    the data into PyTorch DataLoaders, applying curriculum learning settings
    (via `snr_split`) specifically to the training set.

    Parameters
    ----------
    segmented_foreground_path : str
        Path to the HDF5 file containing the segmented and whitened foreground data.
    validation_split : list[float] | None, optional
        A list of three floats representing the fraction of data to allocate
        for training, validation, and testing, respectively.
        Must sum to 1.0. By default [0.7, 0.2, 0.1].
    batch_size : int, optional
        The number of samples per batch to load, by default 256.
    snr_split : float, optional
        The fraction of the training dataset to retain based on the highest
        SNR values, by default 1.0 (100% of the data).

    Returns
    -------
    tuple[DataLoader, DataLoader, DataLoader]
        A tuple containing three DataLoaders:
        - fore_train_loader (shuffled, filtered by snr_split)
        - fore_val_loader (not shuffled)
        - fore_test_loader (not shuffled)
    """
    if validation_split is None:
        validation_split = [0.7, 0.2, 0.1]

    # Checking:
    cond1 = not math.isclose(sum(validation_split), 1.0, rel_tol=1e-9, abs_tol=1e-9)
    cond2 = len(validation_split) != 3
    cond3 = any(i<0 for i in validation_split)
    cond = cond1 or cond2 or cond3

    if cond:
        raise ValueError("validation_split must be a list of 3 positive numbers between 0 and 1"
                         "with sum 1, for example [0.7, 0.2, 0.1] or [0.6, 0.4, 0.0].")

    # Recovering segmented data from the HDF5 files
    with h5py.File(segmented_foreground_path, "r") as f:
        # Shape: (2, # generated times, stride * 2048)
        fore_whitened = np.array(f["whitened"][()])
        # Shape: (# generated times,)
        snr = np.array(f["snr"][()])

    fore_whitened = torch.from_numpy(fore_whitened)
    snr = torch.from_numpy(snr)

    # Reordering to shape (batch, channel, lenght) = (# generated times, 2, stride * 2048)
    fore_whitened = fore_whitened.permute(1, 0, 2)

    # Random shuffling of the indexes to shuffle the foreground data
    num_fore = fore_whitened.shape[0]
    fore_perm = torch.randperm(num_fore)

    # Splitting background training, validation and test tensor
    train_end = round(num_fore * validation_split[0])
    val_end = round(num_fore * (validation_split[0] + validation_split[1]))

    train_indexes = fore_perm[ : train_end]
    val_indexes = fore_perm[train_end : val_end]
    test_indexes = fore_perm[val_end : ]

    # Foreground training tensor (with corresponding SNR)
    fore_train = fore_whitened[train_indexes]
    snr_train = snr[train_indexes]
    # Foreground validation and test tensors
    fore_val = fore_whitened[val_indexes]
    fore_test = fore_whitened[test_indexes]

    fore_train_dset = Foreground_Dataset(fore_train, snr_split = snr_split, snr = snr_train)
    fore_val_dset = Foreground_Dataset(fore_val)
    fore_test_dset = Foreground_Dataset(fore_test)

    # Creating dataloaders
    fore_train_loader = DataLoader(
                fore_train_dset, batch_size = batch_size, shuffle = True, pin_memory = True
            )
    fore_val_loader = DataLoader(
        fore_val_dset, batch_size = batch_size, shuffle = False, pin_memory = True
            )
    fore_test_loader = DataLoader(
        fore_test_dset, batch_size = batch_size, shuffle = False, pin_memory = True
            )

    return fore_train_loader, fore_val_loader, fore_test_loader
