import torch
import torch.nn as nn

from MLGWSC_proposal.dain import DAIN_Layer


class Block(nn.Module):
    """
    A bottleneck building block for a 1D Residual Network (ResNet).

    The block consists of:
    - A 1D convolutional layer with kernel size 1 for dimensionality reduction.
    - A 1D convolutional layer with kernel size 3, using the specified `stride` (bottleneck).
    - A 1D convolutional layer with kernel size 1 for dimensionality expansion.
    - A 1D Batch Normalization layer after each convolution.
    - A ReLU activation function.

    The first convolutional layer accepts an input with `in_channels_1` channels
    and outputs a reduced number of activation maps equal to `out_channels_1`.
    The final output of the block expands to `out_channels_1 * dim_expansion` channels.
    A residual skip connection is applied before the final ReLU activation by summing
    the convolutional output with the original input, which is automatically projected
    via a strided convolution if spatial or channel dimensions need to be aligned.

    Parameters
    ----------
    in_channels_1 : int
        Number of input channels to the block.
    out_channels_1 : int
        Number of internal channels (activation maps) output by the first convolutional layer.
    stride : int, optional
        Stride of the second convolutional layer (kernel size 3) and the projection
        skip connection, by default 1.
    dim_expansion : int, optional
        Expansion factor for the final convolutional layer, by default 4.
        The total number of output channels of the block will be `out_channels_1 * dim_expansion`.
    bn_momentum : float, optional
        `momentum` value for the `BatchNorm1d` running statistics, by default 0.05.
    """
    def __init__(
        self,
        in_channels_1 : int,
        out_channels_1 :int ,
        stride : int = 1,
        dim_expansion : int = 4,
        bn_momentum : float = 0.05 # Reduced from the default 0.1 value
        ) -> None:
        super().__init__()

        self.block_conv1 = nn.Conv1d(
            # Number of activation maps outputted by the previous layer
            in_channels = in_channels_1,
            # Number of activation maps outputted by this layer
            out_channels = out_channels_1,
            kernel_size = 1,
            stride = 1,
            padding = 0,
            bias = False # Bias will be taken care of by BatchNorm1d
            )

        self.block_norm1 = nn.BatchNorm1d(out_channels_1, momentum = bn_momentum)

        self.block_conv2 = nn.Conv1d(
            in_channels = out_channels_1,
            out_channels = out_channels_1,
            kernel_size = 3,
            # Stride is parameterized because the first block of a new stage downsamples
            # the temporal dimension using stride = 2, while subsequent blocks use stride = 1
            stride = stride,
            # With kernel_size=3, padding=1 is required to preserve the sequence length
            # when stride = 1 so that the convolution itself does not shrink the activation
            # maps at the edges:
            # L_out = [L_in + 2p - 2k]/s + 1 => P = [s(L_out -1) - L_in + k]/2
            # and for L_in = L_out, k = 3 and s = 1 we get p = 1
            padding = 1,
            bias = False
        )

        self.block_norm2 = nn.BatchNorm1d(out_channels_1, momentum = bn_momentum)

        self.block_conv3 = nn.Conv1d(
            in_channels = out_channels_1,
            out_channels = out_channels_1 * dim_expansion,
            kernel_size = 1,
            stride = 1,
            padding = 0,
            bias = False
        )

        self.block_norm3 = nn.BatchNorm1d(out_channels_1 * dim_expansion, momentum = bn_momentum)

        if stride !=1 or in_channels_1 != out_channels_1 * dim_expansion:
            self.identity_downsample = nn.Sequential(
                nn.Conv1d(
                    in_channels = in_channels_1,
                    out_channels = out_channels_1 * dim_expansion,
                    kernel_size = 1,
                    stride = stride,
                    bias = False
                    ),
                nn.BatchNorm1d(out_channels_1 * dim_expansion, momentum = bn_momentum)
                )
        else:
            self.identity_downsample = None

        self.relu = nn.ReLU()

    def forward(self, x : torch.Tensor) -> torch.Tensor:

        identity = x

        # 1x1 conv stride = 1 -> out_channels_1 activation maps
        x = self.block_conv1(x)
        x = self.block_norm1(x)
        x = self.relu(x)

        # 3x3 conv stride = 1 or 2 -> out_channels_1 activation maps
        x = self.block_conv2(x)
        x = self.block_norm2(x)
        x = self.relu(x)

        # 1x1 conv stride = 1 -> out_channels_1 * 4 activation maps
        x = self.block_conv3(x)
        x = self.block_norm3(x)

        # Residual skip connection
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        x = x + identity
        x = self.relu(x)

        return x

class BasicBlock(nn.Module):
    """
    A standard residual building block for a 1D Residual Network (ResNet-18 / ResNet-34).

    The block consists of:
    - A 1D convolutional layer with kernel size 3, using the specified `stride`.
    - A 1D convolutional layer with kernel size 3 and stride 1.
    - A 1D Batch Normalization layer after each convolution.
    - A ReLU activation function.

    The first convolutional layer accepts an input with `in_channels_1` channels
    and outputs `out_channels_1` activation maps, while the second convolution
    preserves `out_channels_1` channels without bottleneck expansion. A residual
    skip connection is applied before the final ReLU activation by summing the
    convolutional output with the original input, which is automatically projected
    via a strided 1x1 convolution if spatial or channel dimensions need to be aligned.

    Parameters
    ----------
    in_channels_1 : int
        Number of input channels to the block.
    out_channels_1 : int
        Number of output channels (activation maps) produced by the block.
    stride : int, optional
        Stride of the first convolutional layer (kernel size 3) and the projection
        skip connection, by default 1.
    bn_momentum : float, optional
        `momentum` value for the `BatchNorm1d` running statistics, by default 0.05.
    """
    def __init__(
        self,
        in_channels_1 : int,
        out_channels_1 :int ,
        stride : int = 1,
        bn_momentum : float = 0.05
        ) -> None:
        super().__init__()

        self.block_conv1 = nn.Conv1d(
            in_channels = in_channels_1,
            out_channels = out_channels_1,
            kernel_size = 3,
            stride = stride,
            padding = 1,
            bias = False
            )

        self.block_norm1 = nn.BatchNorm1d(out_channels_1, momentum = bn_momentum)

        self.block_conv2 = nn.Conv1d(
            in_channels = out_channels_1,
            out_channels = out_channels_1,
            kernel_size = 3,
            stride = 1,
            padding = 1,
            bias = False
        )

        self.block_norm2 = nn.BatchNorm1d(out_channels_1, momentum = bn_momentum)

        if stride !=1 or in_channels_1 != out_channels_1:
            self.identity_downsample = nn.Sequential(
                nn.Conv1d(
                    in_channels = in_channels_1,
                    out_channels = out_channels_1,
                    kernel_size = 1,
                    stride = stride,
                    bias = False
                    ),
                nn.BatchNorm1d(out_channels_1, momentum = bn_momentum)
                )
        else:
            self.identity_downsample = None

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        # 3x3 conv stride = 1 or 2 -> out_channels_1 activation maps
        x = self.block_conv1(x)
        x = self.block_norm1(x)
        x = self.relu(x)

        # 3x3 conv stride = 1  -> out_channels_1 activation maps
        x = self.block_conv2(x)
        x = self.block_norm2(x)

        # Residual skip connection
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        x = x + identity
        x = self.relu(x)

        return x

class ResNet(nn.Module): # For ResNet152: stages = [3, 8, 36, 3]
    """A 1D Residual Network (ResNet) backbone for time-series classification.

    This class implements the ResNet architecture (adapted for 1D signals) described by
    He et al. (https://arxiv.org/abs/1512.03385). It supports both standard two-layer
    residual blocks (`BasicBlock`, used in ResNet-18 and ResNet-34) and three-layer
    bottleneck blocks (`Block`, used in ResNet-50, ResNet-101, and ResNet-152) stacked
    across four main stages, ending with a concatenated global average and max pooling
    layer, dropout regularization, and a fully connected linear classifier.

    Parameters
    ----------
    stages : list[int]
        Number of residual blocks per stage. Common configurations include:
        - ResNet-18: `stages = [2, 2, 2, 2]` with `basic_block = True`;
        - ResNet-34: `stages = [3, 4, 6, 3]` with `basic_block = True`;
        - ResNet-50: `stages = [3, 4, 6, 3]` with `basic_block = False`;
        - ResNet-101: `stages = [3, 4, 23, 3]` with `basic_block = False`;
        - ResNet-152: `stages = [3, 8, 36, 3]` with `basic_block = False`.
    input_channels : int
        Number of channels of the input tensor with expected shape
        `(batch_size, input_channels, data_len)`.
    num_classes : int
        Number of output logits for the classification task.
    base_width : int, optional
        Base number of channels in the first stage. Subsequent stages scale this
        value by factors of 2, 4, and 8, by default 32.
    dropout : float, optional
        Dropout probability applied before the final linear classifier, by default 0.5.
    bn_momentum : float, optional
        Momentum value for all `BatchNorm1d` layers to stabilize running statistics
        across deep residual stages, by default 0.05.
    basic_block : bool, optional
        If True, builds stages using `BasicBlock` (expansion factor 1); if False, uses
        bottleneck `Block` (expansion factor 4), by default False.
    """
    def __init__(
        self,
        stages: list[int],
        input_channels : int,
        num_classes : int,
        base_width : int = 32,
        dropout : float = 0.5,
        bn_momentum : float = 0.05,
        basic_block : bool = False
        ) -> None:
        super().__init__()

        self.stages = stages
        self.num_classes = num_classes
        self.bn_momentum = bn_momentum
        self.basic_block = basic_block

        # ---- 1st stage (1 layer) ----
        # Preserve full 2048 Hz temporal resolution in early layers (as in Table I of Nousi et al.)
        self.conv1 = nn.Conv1d(
            in_channels = input_channels,
            out_channels = base_width,
            kernel_size = 3,
            stride = 1,
            # Remeber: P = [s(L_out -1) - L_in + k]/2
            # In the original architecture L_out = L_in/2, k = 7 and s = 2, so p = 3
            padding = 1,
            bias = False
        )

        self.bn1 = nn.BatchNorm1d(base_width, momentum = bn_momentum)
        self.relu = nn.ReLU()

        self.stage1 = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu
        )

        self.stage_in_channels = base_width

        # ---- 2nd stage ----
        # For ResNet152: 3 blocks -> 9 layers
        # With base_width = 32: Input 32 channels -> Output 128 channels
        # no downsampling = > stride = 1

        self.stage2 = self._make_stage(
            stages[0],
            out_channels_1 = base_width,
            first_block_stride = 1
            )

        # ---- 3rd stage ----
        # For ResNet152: 8 blocks -> 24 layers
        # With base_width = 32: Input 128 channels -> Output 256 channels
        # downsampling with stride = 2 in the first block

        self.stage3 = self._make_stage(
            stages[1],
            out_channels_1 = base_width * 2,
            first_block_stride = 2
            )

        # ---- 4th stage ---
        # For ResNet152: 36 blocks -> 108 layers
        # With base_width = 32: Input 256 channels -> Output 512 channels
        # downsampling with stride = 2 in the first block

        self.stage4 = self._make_stage(
            stages[2],
            out_channels_1 = base_width * 4,
            first_block_stride = 2
            )

        # ---- 5th stage ---
        # For ResNet152: 3 blocks -> 9 layers
        # With base_width=32: Input 512 channels -> Output 1024 channels
        # downsampling with stride = 2 in the first block

        self.stage5 = self._make_stage(
            stages[3],
            out_channels_1 = base_width * 8,
            first_block_stride = 2
            )

        # Pooling (Concat Avg + Max), Dropout and Linear
        # Max pooling and average pooling not subsequent but concatened (see forward)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.maxpool_out = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(p = dropout)
                            # base_width * num_stages * dim_expansion * num_pooling_operations
        dim_expansion = 1 if basic_block else 4
        self.output = nn.Linear(base_width * 8 * dim_expansion * 2, self.num_classes)

        #*Initialize convolutional weights using Kaiming normal distribution
        # to maintain variance stability across deep ReLU layers
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode = "fan_out", nonlinearity = "relu")

        #*Zero-initialize the last BatchNorm layer in each residual block.
        # This makes each residual block behave like an identity function at the start
        # of training (x + 0 = x), drastically improving convergence in very deep networks
        # (Goyal et al., 2018).
        for m in self.modules():
            if isinstance(m, Block):
                nn.init.zeros_(m.block_norm3.weight)
            elif isinstance(m, BasicBlock):
                nn.init.zeros_(m.block_norm2.weight)

    def _make_stage(
        self,
        stage : int,
        out_channels_1 : int,
        first_block_stride : int
        ) -> nn.Sequential:

        if self.basic_block:
            first_block = BasicBlock(
                in_channels_1 = self.stage_in_channels,
                out_channels_1 = out_channels_1,
                stride = first_block_stride,
                bn_momentum = self.bn_momentum
            )

            layers = [first_block]

            for _i in range(stage - 1):
                block = BasicBlock(
                    in_channels_1 = out_channels_1,
                    out_channels_1 = out_channels_1,
                    stride = 1,
                    bn_momentum = self.bn_momentum
                )
                layers.append(block)

            self.stage_in_channels = out_channels_1
        else:
            first_block = Block(
                in_channels_1 = self.stage_in_channels,
                out_channels_1 = out_channels_1,
                stride = first_block_stride,
                bn_momentum = self.bn_momentum
            )

            layers = [first_block]
            for _i in range(stage - 1):
                block = Block(
                    in_channels_1 = out_channels_1 * 4,
                    out_channels_1 = out_channels_1,
                    stride = 1,
                    dim_expansion = 4,
                    bn_momentum = self.bn_momentum
                )
                layers.append(block)

            self.stage_in_channels = out_channels_1 * 4

        return nn.Sequential(*layers)

    def forward(self, x : torch.Tensor) -> torch.Tensor:

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        # shape of x : (batch_size, base_width * 32, reduced_strain_data) =
        # (batch_size, num_channels, reduced_strain_data)

        x_avg = torch.flatten(self.avgpool(x), 1)
        x_max = torch.flatten(self.maxpool_out(x), 1)
        # shape of x : (batch_size, base_width * 32) = (batch_size, num_channels)
        x = torch.cat([x_avg, x_max], dim=1)
        # shape of x : (batch_size, base_width * 32 * 2)

        x = self.dropout(x)
        x = self.output(x)
        # shape of x : (batch_size, num_classes)
        return x

class GW_ResNet(nn.Module):
    """Configurable 1D ResNet architecture with optional Deep Adaptive Input Normalization (DAIN).

    Combines an optional adaptive preprocessing layer (`DAIN_Layer`) with a configurable
    1D `ResNet` backbone for binary classification of two-channel (`H1`, `L1`)
    gravitational-wave strain time-series across different network depths
    (e.g., ResNet-34, ResNet-50, ResNet-101, and ResNet-152).

    Parameters
    ----------
    base_width : int, optional
        Base channel width of the ResNet backbone, by default 32. Using 32 instead of 64
        reduces parameter count by ~4x to mitigate overfitting on 1D strain data while
        preserving network depth.
    dropout : float, optional
        Dropout probability before the final classification layer, by default 0.5.
    stages : tuple[int, ...] | list[int], optional
        Number of residual blocks per stage, by default `(3, 8, 36, 3)` (ResNet-152).
    basic_block : bool, optional
        If True, uses `BasicBlock` (for ResNet-18 and ResNet-34); if False, uses
        bottleneck `Block` (for ResNet-50, ResNet-101, and ResNet-152), by default False.
    use_dain : bool, optional
        If True, applies `DAIN_Layer(mode="full")` before the ResNet backbone;
        if False, bypasses adaptive normalization and feeds the whitened strain directly
        into the ResNet, by default True.
    """
    def __init__(
        self,
        base_width: int = 32,
        dropout: float = 0.5,
        stages = (3, 8, 36, 3),
        basic_block: bool = False,
        use_dain: bool = True
    ) -> None:
        super().__init__()
        self.dain = DAIN_Layer(mode = "full" if use_dain else None, input_dim = 2)
        self.resnet = ResNet(
            stages = list(stages),
            input_channels = 2,
            num_classes = 1,
            base_width = base_width,
            dropout = dropout,
            basic_block = basic_block
        )

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = self.dain(x)
        x = self.resnet(x)
        return x
