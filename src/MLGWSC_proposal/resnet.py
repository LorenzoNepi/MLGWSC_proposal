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
    """
    def __init__(
        self,
        in_channels_1 : int, out_channels_1 :int ,
        stride : int = 1, dim_expansion : int = 4
        ) -> None:
        super().__init__()

        self.block_conv1 = nn.Conv1d(
            # Number of activation maps outputted by the previous layer
            in_channels = in_channels_1,
            # Number of activation maps outputted by this layer
            out_channels = out_channels_1,
            kernel_size = 1,
            stride = 1,
            padding = 0
            )

        self.block_norm1 = nn.BatchNorm1d(out_channels_1)

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
            padding = 1
        )

        self.block_norm2 = nn.BatchNorm1d(out_channels_1)

        self.block_conv3 = nn.Conv1d(
            in_channels = out_channels_1,
            out_channels = out_channels_1 * dim_expansion,
            kernel_size = 1,
            stride = 1,
            padding = 0
        )

        self.block_norm3 = nn.BatchNorm1d(out_channels_1 * dim_expansion)

        if stride !=1 or in_channels_1 != out_channels_1 * dim_expansion:
            self.identity_downsample = nn.Sequential(
                nn.Conv1d(
                    in_channels = in_channels_1,
                    out_channels = out_channels_1 * dim_expansion,
                    kernel_size = 1,
                    stride = stride
                    ),
                nn.BatchNorm1d(out_channels_1 * dim_expansion)
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
        x = self.relu(x)

        # Residual skip connection
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        x += identity
        x = self.relu(x)

        return x

class ResNet(nn.Module): # For ResNet152: stages = [3, 8, 36, 3]
    """A 1D Residual Net architecture (ResNet) for classification.

    This class implements the ResNet model (adapted for 1D signals) described in the
    following paper by He et al.: https://arxiv.org/abs/1512.03385. The original design
    was adapted for 1D input data and it features a deep stack of bottleneck blocks divided
    into four main stages, ending with a global average pooling layer and a fully
    connected linear layer.

    Parameters
    ----------
    stages : list[int]
        Number of blocks of type `Block` per stage. For example:
        - ResNet50: `stages = [3, 4, 6, 3]`;
        - ResNet101: `stages = [2, 4, 23, 3]`;
        - ResNet152: `stages = [3, 8, 36, 3]`.
    input_channels : int
        Number of channels of the input tensor with expected
        shape `(batch_size, num_channels, data_len)`.
    num_classes : int
        Number of output classes for the classification task.
    """
    def __init__(
        self,
        stages: list[int], input_channels : int, num_classes : int
        ) -> None:
        super().__init__()

        self.stages = stages
        self.num_classes = num_classes

        # ---- 1st stage (1 layer) ----
        # Big conv with stride 2 to half the timeseries resolution
        self.conv1 = nn.Conv1d(
            in_channels = input_channels,
            out_channels = 64,
            kernel_size = 7,
            stride = 2,
            # P = [s(L_out -1) - L_in + k]/2
            # We want L_out = L_in/2, k = 7 and s = 2, so p = 3
            padding = 3,
        )

        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()

        # Max pooling with stride 2 to half it again
        self.maxpool = nn.MaxPool1d(
            kernel_size = 3, stride = 2, padding = 1
        )

        self.stage1 = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.maxpool
        )

        self.stage_in_channels = 64

        # ---- 2nd stage ----
        # For ResNet152: 3 blocks -> 9 layers
        # Input: 64 channels, Output: 256 channels
        # no downsampling = > stride = 1

        self.stage2 = self._make_stage(
            stages[0],
            out_channels_1 = 64,
            first_block_stride = 1
            )

        # ---- 3rd stage ----
        # For ResNet152: 8 blocks -> 24 layers
        # Input: 256 channels, Output: 512
        # downsampling with stride = 2 in the first block

        self.stage3 = self._make_stage(
            stages[1],
            out_channels_1 = 128,
            first_block_stride = 2
            )

        # ---- 4th stage ---
        # For ResNet152: 36 blocks -> 108 layers
        # Input: 512 channels, Output: 1024 channels
        # downsampling with stride = 2 in the first block

        self.stage4 = self._make_stage(
            stages[2],
            out_channels_1 = 256,
            first_block_stride = 2
            )

        # ---- 5th stage ---
        # For ResNet152: 3 blocks -> 9 layers
        # Input: 1024 channels, Output: 2048 channels
        # downsampling with stride = 2 in the first block

        self.stage5 = self._make_stage(
            stages[3],
            out_channels_1 = 512,
            first_block_stride = 2
            )

        # Pooling and Linear
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.output = nn.Linear(2048, self.num_classes)

    def _make_stage(
        self,
        stage : int, out_channels_1 : int, first_block_stride : int
        ) -> nn.Sequential:

        first_block = Block(
            in_channels_1 = self.stage_in_channels,
            out_channels_1 = out_channels_1,
            stride = first_block_stride,
            dim_expansion = 4
        )

        layers = [first_block]
        for _i in range(stage - 1):
            block = Block(
                in_channels_1 = out_channels_1 * 4,
                out_channels_1 = out_channels_1,
                stride = 1,
                dim_expansion = 4
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
        # shape of x : (batch_size, 2048, 64) = (batch_size, num_channels, reduced_strain_data)

        x = self.avgpool(x)
        # shape of x : (batch_size, 2048, 1)
        x = torch.flatten(x, 1)
        # shape of x : (batch_size, 2048) = (batch_size, num_channels)
        x = self.output(x)
        # shape of x : (batch_size, num_classes)
        return x

class GW_ResNet152(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dain = DAIN_Layer(mode = "full", input_dim = 2)
        # DAIN_layer does not modify the shape of the input tensor
        self.resnet = ResNet(stages = [3, 8, 36, 3], input_channels = 2, num_classes = 1)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = self.dain(x)
        x = self.resnet(x)
        return x
