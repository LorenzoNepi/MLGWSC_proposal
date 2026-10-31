import torch
from torch import nn


class DAIN_Layer(nn.Module):
    """Adaptive normalization layer.

    This module was first implemented by N. Passalis et al. in the following paper:
    https://arxiv.org/pdf/1902.07892v1. You can find the original source code
    (which was modified to address some deprecation issues) in the following
    GitHub repository: https://github.com/passalis/dain.

    Parameters
    ----------
    mode : str | None, optional
        Specifies the normalization scheme to apply. Supported values are:
        `None` (no normalization), `'avg'` (simple average subtraction),
        `'adaptive_avg'` (adaptive average), `'adaptive_scale'` (adaptive average
        and scaling), and `'full'` (adaptive average, scaling, and gating),
        by default 'adaptive_avg'.
    mean_lr : float, optional
        Suggested learning rate for the `mean_layer`. Note: This value is only
        stored as an attribute; to take effect, it must be explicitly passed
        to the optimizer via parameter groups during training setup,
        by default 0.00001.
    gate_lr : float, optional
        Suggested learning rate for the `gating_layer` (see note on `mean_lr`),
        by default 0.001.
    scale_lr : float, optional
        Suggested learning rate for the `scaling_layer` (see note on `mean_lr`),
        by default 0.00001.
    input_dim : int, optional
        Number of input features (channels). This corresponds to the second
        dimension of the expected input tensor in the forward pass, which
        must have the shape `(batch_size, input_dim, sequence_length)`,
        by default 144.
    """

    def __init__(self, mode: str | None = 'adaptive_avg',
                 mean_lr: float = 0.00001,
                 gate_lr: float = 0.001,
                 scale_lr: float = 0.00001,
                 input_dim: int = 144) -> None:
        super().__init__()

        self.mode = mode
        self.mean_lr = mean_lr
        self.gate_lr = gate_lr
        self.scale_lr = scale_lr

        # Parameters for adaptive average
        self.mean_layer = nn.Linear(input_dim, input_dim, bias=False)
        # * Replaced deprecated .data assignment with nn.init.eye_
        nn.init.eye_(self.mean_layer.weight)

        # Parameters for adaptive std
        self.scaling_layer = nn.Linear(input_dim, input_dim, bias=False)
        # * Replaced deprecated .data assignment with nn.init.eye_
        nn.init.eye_(self.scaling_layer.weight)

        # Parameters for adaptive scaling
        self.gating_layer = nn.Linear(input_dim, input_dim)

        self.eps = 1e-8

    def forward(self, x):
        # Expecting  (n_samples, dim,  n_feature_vectors)

        # Nothing to normalize
        if self.mode is None:
            pass

        # Do simple average normalization
        elif self.mode == 'avg':
            # * Added dim=2 for explicit dimension declaration
            avg = torch.mean(x, dim=2)
            # * Replaced deprecated .resize with .view
            avg = avg.view(avg.size(0), avg.size(1), 1)
            x = x - avg

        # Perform only the first step (adaptive averaging)
        elif self.mode == 'adaptive_avg':
            # * Added dim=2 for explicit dimension declaration
            avg = torch.mean(x, dim=2)
            adaptive_avg = self.mean_layer(avg)
            # * Replaced deprecated .resize with .view
            adaptive_avg = adaptive_avg.view(adaptive_avg.size(0), adaptive_avg.size(1), 1)
            x = x - adaptive_avg

        # Perform the first + second step (adaptive averaging + adaptive scaling )
        elif self.mode == 'adaptive_scale':

            # Step 1:
            # * Added dim=2 for explicit dimension declaration
            avg = torch.mean(x, dim=2)
            adaptive_avg = self.mean_layer(avg)
            # * Replaced deprecated .resize with .view
            adaptive_avg = adaptive_avg.view(adaptive_avg.size(0), adaptive_avg.size(1), 1)
            x = x - adaptive_avg

            # Step 2:
            # * Added dim=2 for explicit dimension declaration
            std = torch.mean(x ** 2, dim=2)
            std = torch.sqrt(std + self.eps)
            adaptive_std = self.scaling_layer(std)
            # * Replaced in-place modification with torch.where to avoid Autograd errors
            adaptive_std = torch.where(adaptive_std <= self.eps,
                                       torch.ones_like(adaptive_std),
                                       adaptive_std)

            # * Replaced deprecated .resize with .view
            adaptive_std = adaptive_std.view(adaptive_std.size(0), adaptive_std.size(1), 1)
            x = x / (adaptive_std)

        elif self.mode == 'full':

            # Step 1:
            # * Added dim=2 for explicit dimension declaration
            avg = torch.mean(x, dim=2)
            adaptive_avg = self.mean_layer(avg)
            # * Replaced deprecated .resize with .view
            adaptive_avg = adaptive_avg.view(adaptive_avg.size(0), adaptive_avg.size(1), 1)
            x = x - adaptive_avg

            # # Step 2:
            # * Added dim=2 for explicit dimension declaration
            std = torch.mean(x ** 2, dim=2)
            std = torch.sqrt(std + self.eps)
            adaptive_std = self.scaling_layer(std)
            # * Replaced in-place modification with torch.where to avoid Autograd errors
            adaptive_std = torch.where(adaptive_std <= self.eps,
                                       torch.ones_like(adaptive_std),
                                       adaptive_std)

            # * Replaced deprecated .resize with .view
            adaptive_std = adaptive_std.view(adaptive_std.size(0), adaptive_std.size(1), 1)
            x = x / adaptive_std

            # Step 3:
            # * Added dim=2 for explicit dimension declaration
            avg = torch.mean(x, dim=2)
            # * Replaced deprecated F.sigmoid with torch.sigmoid
            gate = torch.sigmoid(self.gating_layer(avg))
            # * Replaced deprecated .resize with .view
            gate = gate.view(gate.size(0), gate.size(1), 1)
            x = x * gate

        else:
            raise AssertionError()

        return x
