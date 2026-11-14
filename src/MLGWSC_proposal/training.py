import math
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryAUROC

from MLGWSC_proposal.resnet import GW_ResNet152


class EarlyStopping:
    """Class to implement early stopping during training.

    This custom class replicates the behavior of the namesake Keras callback:
    - It monitors the validation loss and manages an internal counter epoch by epoch;
    - If the monitored value stops improving for `patience` consecutive epochs,
    the class signals the need to stop early to prevent overfitting.
    - Whenever the loss registers a new minimum, the class automatically saves the model's
    state (the best weights) to disk via `torch.save`.

    Parameters
    ----------
    patience : int, optional
        Number of consecutive epochs with no improvement after which training
        will be stopped, by default 20.
    verbose : bool, optional
        If True, prints a message for each epoch where the loss doesn't improve,
        as well as when a new best model is saved, by default False.
    delta : float, optional
        Minimum change in the monitored quantity to qualify as an improvement.
        An absolute decrease of less than `delta` will count as no improvement,
        by default 0.0.
    path : str, optional
        Path to save the model weights, by default 'best_model.pth'.
    """

    def __init__(
        self,
        patience : int = 20,
        verbose : bool = False,
        delta : float = 0.0,
        path : str = 'best_model.pth'
    ) -> None:
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.delta = delta
        self.path = path

    def __call__(self, val_loss: float, model: nn.Module) -> None:
        # Implementing early stopping with the __call__ special method
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)

        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping: patience at {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss: float, model: nn.Module) -> None:
        # Saving the model when validation loss decreases
        if self.verbose:
            print(f'Validation loss new minimum: {val_loss:.4f}. '
                  f'Saving model to {self.path}...')
        # Saves only the weights (state_dict), which is the PyTorch best practice
        torch.save(model.state_dict(), self.path)


def bc_train_loop(
    back_dataloader : DataLoader,
    fore_dataloader : DataLoader,
    model : nn.Module,
    loss_fn : nn.Module,
    optimizer : torch.optim.Optimizer,
    device : torch.device | str,
    warmup_scheduler : torch.optim.lr_scheduler.LRScheduler | None = None,
    global_step : int = 0,
    warmup_steps : int = 200
) -> tuple[float, int]:
    """Execute a single training epoch for binary classification.

    This function iterates simultaneously over the background and foreground
    dataloaders. For each step, it concatenates the background and foreground
    tensors to create a perfectly balanced 50/50 batch, shuffles the batch
    internally, and performs the standard forward pass, loss computation,
    and backpropagation with gradient clipping. It also steps a warmup scheduler
    if provided.

    Parameters
    ----------
    back_dataloader : DataLoader
        PyTorch DataLoader containing the background dataset.
    fore_dataloader : DataLoader
        PyTorch DataLoader containing the foreground dataset.
    model : nn.Module
        The neural network model to be trained.
    loss_fn : nn.Module
        The loss function criterion (e.g. nn.BCEWithLogitsLoss).
    optimizer : torch.optim.Optimizer
        The optimization algorithm (e.g. Adam, AdamW).
    device: torch.device | str
        The device where computations will be performed (e.g. "cuda", "cpu").
    warmup_scheduler : torch.optim.lr_scheduler.LRScheduler | None, optional
        A learning rate scheduler used to gradually increase the learning rate
        during the initial steps of training, by default None.
    global_step : int, optional
        The current global step (batch iteration) counter across epochs, used
        to track the warmup phase, by default 0.
    warmup_steps : int, optional
        Total number of initial steps during which `warmup_scheduler` is stepped,
        by default 200.

    Returns
    -------
    tuple[float, int]
        A tuple containing:
        - The average training loss across all batches in the epoch.
        - The updated global step counter.
    """

    # Setting the model in train mode
    model.train()
    total_loss = 0.0
    # zip() stops at the shortest iterable so the actual number of batches
    # processed is the minimum of the two dataloaders lengths, which might
    # be different for curriculum learning.
    num_batches = min(len(back_dataloader), len(fore_dataloader))

    # Iterating over the batches
    for (X_back, y_back) , (X_fore, y_fore) in zip(back_dataloader, fore_dataloader, strict=False):

        # Concatenening tensors along the batch dimension and shuffling
        X = torch.cat([X_back, X_fore], dim=0)
        y = torch.cat([y_back, y_fore], dim=0)

        indexes = torch.randperm(X.shape[0])
        X = X[indexes]
        y = y[indexes]

        # Moving data to GPU
        X = X.to(device, non_blocking = True)
        y = y.to(device, non_blocking = True)

        # Resetting the optimizers gradients
        optimizer.zero_grad(set_to_none = True)

        # Computing the model prediction on the batch
        pred = model(X)

        # Computing the loss function
        # The model outputs a 2D column tensor of shape `(batch_size, 1)`, while dataset labels
        # arrive as a flat 1D array of shape `(batch_size,)`. In (`.view(-1, 1)`) the `-1` infers
        # the current batch size automatically while adding the trailing dimension of size `1`,
        # preventing broadcasting errors or shape mismatch crashes.
        # # Shape: (batch_size,) -> (batch_size, 1)
        y = y.view(-1, 1).float()
        loss = loss_fn(pred, y)

        # Backword propagation
        loss.backward()

        #* Implementing gradient clipping: if the computed gradients are massive
        # (causing loss explosions to millions), this scales them down
        # so their total norm doesn't exceed 1.0.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Weight optimization
        optimizer.step()

        #* Advance the warmup scheduler during the first warmup_steps steps
        if warmup_scheduler is not None and global_step < warmup_steps:
            warmup_scheduler.step()
        global_step += 1

        # Computing the total loss of batch and returning the mean loss per segment
        total_loss = total_loss + loss.item()

    return total_loss / num_batches, global_step


def bc_validation_loop(
    back_loader : DataLoader,
    fore_loader : DataLoader,
    model : nn.Module,
    loss_fn : nn.Module,
    device : torch.device | str
) -> tuple[float, float, list[float]]:
    """Compute validation loss and metric value (AUC) after each training epoch.

    This function iterates simultaneously over the background and foreground
    validation dataloaders, computing the validation loss and the AUC metric
    for model evaluation.

    Parameters
    ----------
    back_loader : DataLoader
        PyTorch DataLoader containing the validation background dataset.
    fore_loader : DataLoader
        PyTorch DataLoader containing the validation foreground dataset.
    model : nn.Module
        The neural network model to be evaluated.
    loss_fn : nn.Module
        The loss function criterion (e.g. nn.BCEWithLogitsLoss).
    device: torch.device | str
        The device where computations will be performed (e.g. "cuda", "cpu").

    Returns
    -------
    tuple[float, float, list[float]]
        A tuple containing:
        - The average validation loss across all batches.
        - The Area Under the ROC Curve metric computed for the epoch.
        - A list of predicted probabilities (0.0 to 1.0) for the dataset.
    """

    # Setting the model in test mode
    model.eval()
    total_val_loss = 0.0
    num_batches = min(len(back_loader), len(fore_loader))

    # Utilizing torchmetrics to define the ROC as a metric instead of accuracy
    # Advantage: everything is computed on the GPU + more precise on unbalanced datasets
    roc_metric = BinaryAUROC().to(device)
    probs_list = []

    # Iterating over the batches
    # Making sure no gradient is computed during validation
    with torch.no_grad():                         # For some reason ruff wants the strict parameter
        for (X_back, y_back) , (X_fore, y_fore) in zip(back_loader, fore_loader, strict = False):

            # Concatenening tensors along the batch dimension
            X = torch.cat([X_back, X_fore], dim=0)
            y = torch.cat([y_back, y_fore], dim=0)

            # Moving data to GPU
            X = X.to(device, non_blocking = True)
            y = y.to(device, non_blocking = True)

            # Computing validation loss
            logits = model(X)
            # Shape: (batch_size,) -> (batch_size, 1)
            y = y.view(-1, 1).float()
            val_loss = loss_fn(logits, y)
            total_val_loss = total_val_loss + val_loss.item()

            # Computing ROC and saving the probability for the 2 classes
            probs = torch.sigmoid(logits)
            # Updating the metric in the GPU VRAM. BinaryAUROC expects 1D tensors, so we must use
            # squeeze(). BinaryAUROC also expects int64 as labels, so we must use .long()
            # Shape: (batch_size, 1) -> (batch_size,)
            roc_metric.update(probs.squeeze(1), y.squeeze(1).long())
            # Shape: (batch_size, 1) -> (batch_size,) -> List[float]
            probs_list.extend(probs.squeeze(1).tolist())

    # Computing the area under the curve (AUC) using the GPU
    # The try-excpet block is just for safety: if all the labels are 0 (all background)
    # BinaryAUROC would raise a ValueError, which might stop the training unexpectedly
    try:
        # .item() convets torch tensor in Python float
        roc_auc = roc_metric.compute().item()
    except ValueError:
        roc_auc = 0.5

    # Resetting the metric for the next epoch
    roc_metric.reset()

    return total_val_loss/num_batches, roc_auc, probs_list


def get_optimizer(
    model: GW_ResNet152,
    base_lr: float = 1e-4,
    weight_decay: float = 3e-2
    ) -> torch.optim.Optimizer:
    """Configure the AdamW optimizer by separating the parameter groups for DAIN and ResNet.

    In a hybrid architecture like `GW_ResNet152`, applying global optimization rules
    to all weights can severely destabilize training. This function splits the model's
    parameters into distinct groups to apply specific learning rates and regularization:

    1. *Weight Decay Prevention for DAIN*: The `mean_layer` and `scaling_layer` in
       the DAIN module are initialized as identity matrices to preserve the input structure.
       If standard L2 regularization (weight decay) were applied, the optimizer would
       constantly push these weights towards zero, effectively erasing the signal over time.
       Therefore, `weight_decay` is strictly set to `0.0` for all DAIN components.
    2. *Granular Learning Rates for DAIN*: DAIN acts as an adaptive preprocessing step.
       To maintain stability across batches, its mean and variance estimations must adapt
       very slowly compared to the main network. Splitting DAIN into three separate
       dictionaries allows the optimizer to fetch the specific, generally much lower,
       learning rates (e.g., 1e-5) defined as attributes (`mean_lr`, `scale_lr`, `gate_lr`).
    3. *Standard Optimization for ResNet*: The ResNet backbone is responsible for
       complex feature extraction and classification. It requires a standard, faster
       learning rate (`base_lr`) to converge properly, alongside standard L2
       regularization (`weight_decay=3e-2`) to prevent overfitting.

    Parameters
    ----------
    model : GW_ResNet152
        The neural network model. It must contain a `.dain` attribute (the DAIN layer)
        and a `.resnet` attribute (the ResNet backbone).
    base_lr : float, optional
        The primary learning rate applied to the ResNet backbone, by default 1e-4.
    weight_decay : float, optional
        L2 weight decay coefficient applied to the ResNet backbone, by default 3e-2.

    Returns
    -------
    torch.optim.Optimizer
        An AdamW optimizer configured with the grouped parameters.
    """

    # DAIN parameters (no weight decay and dedicated learning rates)
    dain_group = [
        {
            "params": model.dain.mean_layer.parameters(),
            "lr": model.dain.mean_lr,
            "weight_decay": 0.0,
        },
        {
            "params": model.dain.scaling_layer.parameters(),
            "lr": model.dain.scale_lr,
            "weight_decay": 0.0,
        },
        {
            "params": model.dain.gating_layer.parameters(),
            "lr": model.dain.gate_lr,
            "weight_decay": 0.0,
        },
    ]

    # ResNet parameters
    resnet_group = [
        {
            "params": model.resnet.parameters(),
            "lr": base_lr,
            "weight_decay": weight_decay, # Standard regularization for ResNet
        }
    ]

    return torch.optim.AdamW(dain_group + resnet_group)


def train_gw_resnet152(
    model : GW_ResNet152,
    back_train_loader : DataLoader,
    fore_train_loader : DataLoader,
    back_val_loader : DataLoader,
    fore_val_loader : DataLoader,
    epochs : int = 50,
    cl_epochs : int = 2,
    path : str = 'best_model.pth',
    base_lr : float = 1e-4,
    scheduler_factor : float = 0.2,
    scheduler_patience : int = 3,
    early_stopping_patience : int = 7,
    verbose : bool = False
) -> tuple[list[float], list[float], list[float]]:
    """Execute the full training and validation pipeline for the `GW_ResNet152` model.

    This function handles the learning loop across multiple epochs, applying backpropagation,
    learning rate scheduling (`ReduceLROnPlateau`), early stopping, and computing the metric
    (AUC) on the validation set. It also manages Curriculum Learning dynamically via the
    foreground dataset's `snr_split` property. The best model weights are automatically
    saved to disk and loaded back into the model at the end of the training process.

    Parameters
    ----------
    model : GW_ResNet152
        The neural network to be trained, instance of `GW_ResNet152`.
    back_train_loader : DataLoader
        DataLoader containing the whitened background training segments.
    fore_train_loader : DataLoader
        DataLoader containing the whitened foreground training segments.
    back_val_loader : DataLoader
        DataLoader containing the whitened background validation segments.
    fore_val_loader : DataLoader
        DataLoader containing the whitened foreground validation segments.
    epochs : int, optional
        Maximum number of training epochs to execute, by default 50.
    cl_epochs : int, optional
        Number of epochs dedicated to each curriculum step, by default 2.
    path : str, optional
        File path where the best model weights will be saved, by default 'best_model.pth'.
    base_lr : float, optional
        Initial learning rate of the optimizer, by default 1e-4.
    scheduler_factor : float, optional
        `factor` parameter for the ReduceLROnPlateau class, by default 0.2.
    scheduler_patience : int, optional
        `patience` parameter for the ReduceLROnPlateau class, by default 3.
    early_stopping_patience : int, optional
        `patience` parameter for the EarlyStopping class, by default 7.
    verbose : bool, optional
        If True, prints detailed training progress per epoch, by default False.

    Returns
    -------
    tuple[list[float], list[float], list[float]]
        A tuple containing three lists representing the training history:
        - train_loss_list: Average training loss per epoch.
        - val_loss_list: Average validation loss per epoch.
        - val_auc_list: Validation ROC-AUC metric per epoch.
    """

    # Defining the loss function (BCEWithLogitsLoss automatically applies
    # the sigmoid to the logits output of the ResNet, best practice for stability)
    loss_fn = nn.BCEWithLogitsLoss(reduction = "mean")
    optimizer = get_optimizer(model, base_lr = base_lr)

    # Defining the scheduler and the early stopping class
    scheduler = ReduceLROnPlateau(
        optimizer, mode = "min", factor = scheduler_factor, patience = scheduler_patience)
    early_stopping = EarlyStopping(
        patience = early_stopping_patience, verbose = verbose, path = path)
    # Defining the warmup scheduler: starts the learning rate at 10% of its target value
    # and linearly increases it over the first 200 batches
    warmup_steps = 200
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor = 0.1, total_iters = warmup_steps
    )
    global_step = 0

    train_loss_list = []
    val_loss_list = []
    val_auc_list = []

    device = (
        torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    )
    if device == "cpu":
        warnings.warn(
            "You are training your model on a CPU, it may take some time.", stacklevel = 2
            )

    model = model.to(device)

    # Save initial weights so `path` always exists even if training stops during curriculum phase
    torch.save(model.state_dict(), path)

    if cl_epochs*2 >= epochs or cl_epochs < 0:
        raise ValueError("cl_epochs must be a positive number < epochs/2 "
                         "otherwise your model will not be trained on the lower SNR segments")
    elif cl_epochs == 0:
        curriculum_learning = False
    else:
        curriculum_learning = True

    # Running training over all the epochs
    for i in range(epochs):

        if verbose:
            epoch_start_time = time.perf_counter()

        # Curriculum learning: the ResNet trains first with signals in the top 20% SNR values,
        # then after cl_epochs epochs with signals in the top 50%, and finally on the full dataset
        if curriculum_learning and i < cl_epochs:
            fore_train_loader.dataset.snr_split = 0.2
        elif curriculum_learning and i < cl_epochs*2:
            fore_train_loader.dataset.snr_split = 0.5
        elif curriculum_learning and i >= cl_epochs*2:
            fore_train_loader.dataset.snr_split = 1.0
        else: # curriculum_learning not active
            fore_train_loader.dataset.snr_split = 1.0

        # Training the model: saving loss and metric values
        train_loss, global_step = bc_train_loop(
            back_dataloader = back_train_loader,
            fore_dataloader = fore_train_loader,
            model = model,
            loss_fn = loss_fn,
            optimizer = optimizer,
            warmup_scheduler = warmup,
            global_step = global_step,
            warmup_steps = warmup_steps,
            device = device
            )

        val_loss, val_auc, _ = bc_validation_loop(
            back_val_loader, fore_val_loader, model, loss_fn, device
            )
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        val_auc_list.append(val_auc)

        # We only step the main scheduler and early stopping once curriculum learning
        # has reached the full dataset phase, otherwise artificial difficulty shifts
        # would trigger false early stops.
        in_full_phase = (not curriculum_learning) or (i >= cl_epochs * 2)

        #* math.isfinite prevents crashing if a rogue NaN slips through
        if in_full_phase and math.isfinite(val_loss):
            # Updating the scheduler and the early stopping function
            scheduler.step(val_loss)
            early_stopping(val_loss, model)

        if verbose:
            epoch_end_time = time.perf_counter()
            elapsed_time = epoch_end_time - epoch_start_time

            hours = int(elapsed_time // 3600)
            mins = int((elapsed_time % 3600) // 60)
            secs = int(elapsed_time % 60)
            print(f"Epoch {i+1} [{hours}h {mins}m {secs}s] - Train Loss: {train_loss:.4f} - "
                  f"Val Loss: {val_loss:.4f} - AUC: {val_auc:.4f}\n")

        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {i+1}!")
            break

    print(f"Training completed. Loading best weights from {path}.")

    # Best practice: loading the best weights at the end of training
    model.load_state_dict(torch.load(path, weights_only=True, map_location=device))

    return train_loss_list, val_loss_list, val_auc_list


def plot_training_result(
    train_loss_list : list[float],
    val_loss_list : list[float],
    val_auc_list : list[float],
    savefig : None | str = None
    ) -> None:
    """Plot the training loss, tha validation loss and the AUC metric per epoch.

    Parameters
    ----------
    train_loss_list : list[float]
        Training loss per epoch.
    val_loss_list : list[float]
        Validation loss per epoch.
    val_auc_list : list[float]
        AUC value per epoch (metric).
    savefig : None | str, optional
        Path to save the plot, by default None.
    """
    epochs = np.arange(1, len(train_loss_list) + 1)

    fig = plt.figure(figsize=(10, 5))
    fig.suptitle(f"GW_ResNet152 training results for {len(train_loss_list)} epochs")

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_title("Train loss and validation loss per epoch")
    ax1.plot(epochs, train_loss_list, color = "red", label = "Train Loss")
    ax1.plot(epochs, val_loss_list, color = "orange", label = "Validation Loss")
    ax1.grid(True)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_title("Area Under the ROC Curve (AUC) per epoch")
    ax2.scatter(
        epochs,
        val_auc_list,
        marker = "o",
        color="green",
        alpha = 0.5,
        label="AUC"
    )
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("AUC")
    ax2.grid(True)
    ax2.legend()

    if savefig is not None:
        plt.savefig(savefig, bbox_inches="tight")
        print(f"Figure saved as {savefig}")

    plt.show()
