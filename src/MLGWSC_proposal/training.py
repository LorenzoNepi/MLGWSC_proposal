import warnings

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryAUROC

from MLGWSC_proposal.resnet import GW_ResNet152


class EarlyStopping:
    """Class to implement early stopping during training.

    This custom class replicates the exact behavior of the namesake `Keras` callback:
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
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn : nn.Module,
    optimizer : torch.optim.Optimizer,
    device : torch.device | str
) -> float:
    """
    Execute a single training epoch for binary classification.

    Parameters
    ----------
    dataloader : DataLoader
        PyTorch DataLoader containing the training dataset.
    model : nn.Module
        The neural network model to be trained.
    loss_fn : nn.Module
        The loss function criterion (e.g. nn.BCEWithLogitsLoss).
    optimizer : torch.optim.Optimizer
        The optimization algorithm (e.g. Adam, AdamW).
    device : torch.device | str
        The device where computations will be performed (e.g. "cuda", "cpu").

    Returns
    -------
    float
        The average training loss across all batches in the epoch.
    """
    # Setting the model in train mode
    model.train()
    total_loss = 0
    num_batches = len(dataloader)

    # Iterating over the batches
    for X, y in dataloader:
        # Moving data to GPU
        X, y = X.to(device, non_blocking = True), y.to(device, non_blocking = True)

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

        # Backpropagation and weight optimization
        loss.backward()
        optimizer.step()

        # Computing the total loss of batch and returning the mean loss per segment
        total_loss += loss.item()

    return total_loss / num_batches


def bc_validation_loop(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    device: torch.device | str
) -> tuple[float, float, list]:
    """Computing validation loss and metric value (AUC) after each training epoch.

    Parameters
    ----------
    dataloader : DataLoader
        PyTorch DataLoader containing the training dataset.
    model : nn.Module
        The neural network model to be evaluated.
    loss_fn : nn.Module
        The loss function criterion (e.g. nn.BCEWithLogitsLoss).
    device : torch.device | str
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

    num_batches = len(dataloader)
    total_val_loss = 0

    # Utilizing torchmetrics to define the ROC as a metric instead of accuracy
    # Advantage: everything is computed on the GPU + more precise on unbalanced datasets
    roc_metric = BinaryAUROC().to(device)
    probs_list = []

    # Making sure no gradient is computed during validation
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)

            # Computing validation loss
            logits = model(X)
            # Shape: (batch_size,) -> (batch_size, 1)
            y = y.view(-1, 1).float()
            val_loss = loss_fn(logits, y)
            total_val_loss += val_loss.item()

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


def get_optimizer(model: GW_ResNet152, base_lr: float = 1e-3) -> torch.optim.Optimizer:
    """Configure the AdamW optimizer by separating the parameter groups for DAIN and ResNet.

    In a hybrid architecture like `GW_ResNet152`, applying global optimization rules
    to all weights can severely destabilize training. This function splits the model's
    parameters into distinct groups to apply specific learning rates and regularization:

    1. **Weight Decay Prevention for DAIN**: The `mean_layer` and `scaling_layer` in
       the DAIN module are initialized as identity matrices to preserve the input structure.
       If standard L2 regularization (weight decay) were applied, the optimizer would
       constantly push these weights towards zero, effectively erasing the signal over time.
       Therefore, `weight_decay` is strictly set to `0.0` for all DAIN components.
    2. **Granular Learning Rates for DAIN**: DAIN acts as an adaptive preprocessing step.
       To maintain stability across batches, its mean and variance estimations must adapt
       very slowly compared to the main network. Splitting DAIN into three separate
       dictionaries allows the optimizer to fetch the specific, generally much lower,
       learning rates (e.g., 1e-5) defined as attributes (`mean_lr`, `scale_lr`, `gate_lr`).
    3. **Standard Optimization for ResNet**: The ResNet backbone is responsible for
       complex feature extraction and classification. It requires a standard, faster
       learning rate (`base_lr`) to converge properly, alongside standard L2
       regularization (`weight_decay=1e-2`) to prevent overfitting.

    Parameters
    ----------
    model : torch.nn.Module
        The neural network model. It must contain a `.dain` attribute (the DAIN layer)
        and a `.resnet` attribute (the ResNet backbone). Expected to be an instance
        of `GW_ResNet152`.
    base_lr : float, optional
        The primary learning rate applied to the ResNet backbone, by default 1e-3.

    Returns
    -------
    torch.optim.torch.optim.Optimizer
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
            "weight_decay": 1e-2, # Regularizzazione standard per ResNet
        }
    ]

    return torch.optim.AdamW(dain_group + resnet_group)

def train_gw_resnet152(
    model : GW_ResNet152,
    train_loader : DataLoader,
    val_loader : DataLoader,
    epochs : int = 500,
    path : str = 'best_model.pth',
    verbose : bool = False
) -> tuple[list[float], list[float], list[float]]:
    """
    Executes the full training and validation pipeline for the `GW_ResNet152` model

    This function handles the learning loop across multiple epochs, applying backpropagation,
    learning rate scheduling (`ReduceLROnPlateau`), early stopping and computing the metric
    (AUC) on the validation set. The best model weights are automatically saved to disk and
    loaded back into the model at the end of the training process.

    Parameters
    ----------
    model : GW_ResNet152
        The neural network to be trained, istance of `GW_ResNet152`.
    train_loader : DataLoader
        DataLoader containing the whitened background and foreground training segments.
    val_loader : DataLoader
        DataLoader containing the whitened background and foreground validation segments.
    epochs : int, optional
        Maximum number of training epochs to execute, by default 500.
    path : str, optional
        File path where the best model weights will be saved, by default 'best_model.pth'.
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
    optimizer = get_optimizer(model)

    # Defining the scheduler and the early stopping class
    scheduler = ReduceLROnPlateau(optimizer, mode = "min", factor = 0.1, patience = 10)
    early_stopping = EarlyStopping(patience = 20, verbose = verbose, path = path)

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

    for i in range(epochs):

        # Training the model: saving loss and metric values
        train_loss = bc_train_loop(train_loader, model, loss_fn, optimizer, device)
        val_loss, val_auc, _ = bc_validation_loop(val_loader, model, loss_fn, device)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        val_auc_list.append(val_auc)

        # Updating the scheduler and the early stopping function
        scheduler.step(val_loss)
        early_stopping(val_loss, model)

        if verbose:
            print(f"Epoch:{i+1}, Train Loss: {train_loss:.4f},",
                  f"Val Loss {val_loss:.4f}, AUC {val_auc:.4f} \n")

        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {i+1}!")
            break

    if verbose:
        print(f"Training completed. Loading best weights from {path}.")

    # Best practice: loading the best weights at the end of training
    model.load_state_dict(torch.load(path, weights_only=True, map_location=device))

    return train_loss_list, val_loss_list, val_auc_list
