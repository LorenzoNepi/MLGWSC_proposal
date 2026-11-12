import pathlib

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from MLGWSC_proposal.dataloader import Foreground_Dataset
from MLGWSC_proposal.resnet import GW_ResNet152
from MLGWSC_proposal.training import (
    EarlyStopping,
    bc_train_loop,
    bc_validation_loop,
    train_gw_resnet152,
)


def test_early_stopping(tmp_path: pathlib.Path) -> None:
    # Setting up a dummy PyTorch model and a temporary save path
    model = nn.Linear(10, 2)
    save_path = tmp_path / "test_best_model.pth"

    # Initializing EarlyStopping with patience = 2 and delta = 0.1
    early_stopping = EarlyStopping(
        patience=2, verbose=False, delta=0.1, path=str(save_path)
    )

    # First epoch: loss drops to 1.0 (new minimum)
    early_stopping(1.0, model)
    assert early_stopping.best_loss == 1.0
    assert early_stopping.counter == 0
    assert early_stopping.early_stop is False
    assert save_path.exists()  # Checkpoint file should be created

    # Second epoch: loss improves slightly (0.95), but less than delta (0.1)
    # This should be counted as NO improvement.
    early_stopping(0.95, model)
    assert early_stopping.best_loss == 1.0  # Best loss shouldn't update
    assert early_stopping.counter == 1
    assert early_stopping.early_stop is False

    # Third epoch: loss improves significantly (0.8)
    early_stopping(0.8, model)
    assert early_stopping.best_loss == 0.8
    assert early_stopping.counter == 0  # Counter must reset to 0
    assert early_stopping.early_stop is False

    # Fourth and Fifth epoch: loss degrades (0.9, 0.9)
    early_stopping(0.9, model)
    assert early_stopping.counter == 1
    assert early_stopping.early_stop is False

    early_stopping(0.9, model)
    assert early_stopping.counter == 2
    assert early_stopping.early_stop is True  # Patience 2 reached

    # Verify checkpoint saving and loading
    # Modifying model weights to ensure loading changes them back to the best ones
    with torch.no_grad():
        model.weight.fill_(0.0)

    # Loading the weights saved during epoch 3 (when loss was 0.8)
    model.load_state_dict(torch.load(str(save_path), weights_only=True))
    assert not torch.all(model.weight == 0.0)  # Weights should be restored


def test_bc_train_loop() -> None:
    # Seting up mock data and dataloaders
    # Background: 10 segments, 2 channels, 100 sequence length
    X_back = torch.randn(10, 2, 100)
    y_back = torch.zeros(10)
    back_loader = DataLoader(TensorDataset(X_back, y_back), batch_size=2)

    # Foreground: 10 segments, 2 channels, 100 sequence length
    X_fore = torch.randn(10, 2, 100)
    y_fore = torch.ones(10)
    fore_loader = DataLoader(TensorDataset(X_fore, y_fore), batch_size=2)

    # Setting up mock model (Flatten + Linear to simulate a basic network)
    class MockModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.flatten = nn.Flatten()
            # 2 channels * 100 length = 200 input features
            self.linear = nn.Linear(200, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.linear(self.flatten(x))

    model = MockModel()

    # Copying the initial weights to verify they actually update during training
    initial_weights = model.linear.weight.clone()

    # Setting up loss, optimizer, and device
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    device = "cpu"

    # Executing the training loop
    avg_loss, global_step = bc_train_loop(
        back_loader, fore_loader, model, loss_fn, optimizer, device
    )

    # Checking that it returns a valid float loss and global_step
    assert isinstance(avg_loss, float)
    assert avg_loss > 0.0
    assert isinstance(global_step, int)
    assert global_step == 5  # 10 samples / batch_size 2 = 5 batches

    # Checking that weights have been updated (meaning backward/step worked)
    assert not torch.all(model.linear.weight == initial_weights)


def test_bc_validation_loop() -> None:
    # Setting up mock data and dataloaders
    # Background: 10 segments, 2 channels, 100 sequence length
    X_back = torch.randn(10, 2, 100)
    y_back = torch.zeros(10)
    back_loader = DataLoader(TensorDataset(X_back, y_back), batch_size=2)

    # Foreground: 10 segments, 2 channels, 100 sequence length
    X_fore = torch.randn(10, 2, 100)
    y_fore = torch.ones(10)
    fore_loader = DataLoader(TensorDataset(X_fore, y_fore), batch_size=2)

    # Setting up mock model
    class MockModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.flatten = nn.Flatten()
            # 2 channels * 100 length = 200 input features
            self.linear = nn.Linear(200, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.linear(self.flatten(x))

    model = MockModel()

    # Copying the initial weights to verify they DO NOT update during validation
    initial_weights = model.linear.weight.clone()

    # Setting up loss and device
    loss_fn = nn.BCEWithLogitsLoss()
    device = "cpu"

    # Ensuring model is in train mode initially to verify the loop switches it to eval
    model.train()

    # Executing the validation loop
    avg_loss, auc_score, probs_list = bc_validation_loop(
        back_loader, fore_loader, model, loss_fn, device
    )

    # Checking that the model was correctly set to evaluation mode
    assert model.training is False

    # Checking that it returns a valid float loss
    assert isinstance(avg_loss, float)
    assert avg_loss > 0.0

    # Checking AUC score
    assert isinstance(auc_score, float)
    assert 0.0 <= auc_score <= 1.0

    # Checking probabilities list
    # We passed 10 background + 10 foreground items = 20 total items
    assert isinstance(probs_list, list)
    assert len(probs_list) == 20
    assert all(0.0 <= p <= 1.0 for p in probs_list)

    # Checking that weights have NOT been updated (no backward/step occurred)
    assert torch.all(model.linear.weight == initial_weights)


def test_train_gw_resnet152(tmp_path: pathlib.Path) -> None:
    # Setting up mock datasets and loaders for train and validation
    X_back = torch.randn(8, 2, 100)
    y_back = torch.zeros(8)
    back_train_loader = DataLoader(TensorDataset(X_back, y_back), batch_size=4)
    back_val_loader = DataLoader(TensorDataset(X_back, y_back), batch_size=4)

    X_fore = torch.randn(8, 2, 100)
    snr = torch.rand(8)

    fore_train_dset = Foreground_Dataset(X_fore, snr = snr)
    fore_val_dset = Foreground_Dataset(X_fore, snr = snr)

    fore_train_loader = DataLoader(fore_train_dset, batch_size=4)
    fore_val_loader = DataLoader(fore_val_dset, batch_size=4)

    # Defining the model and checkpoint path
    model = GW_ResNet152()
    save_path = str(tmp_path / "best_model.pth")

    # Running training for just 3 epochs with curriculum learning enabled
    train_loss, val_loss, val_auc = train_gw_resnet152(
        model=model,
        back_train_loader=back_train_loader,
        fore_train_loader=fore_train_loader,
        back_val_loader=back_val_loader,
        fore_val_loader=fore_val_loader,
        epochs=3,
        cl_epochs=1,
        path=save_path,
        verbose=False
    )

    # Checking lists lengths match the number of executed epochs
    assert len(train_loss) == 3
    assert len(val_loss) == 3
    assert len(val_auc) == 3

    # Checking types and value bounds
    assert all(isinstance(a, float) for a in train_loss)
    assert all(isinstance(a, float) for a in val_loss)
    assert all(isinstance(a, float) for a in val_auc)

    # Check that the checkpoint file was successfully saved by EarlyStopping
    assert pathlib.Path(save_path).exists()
