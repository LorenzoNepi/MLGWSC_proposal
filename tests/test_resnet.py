# Lunedì 09: Completa test_resnet
#            Scrivi funzioni per il training
#            Esegui training su Kaggle
# Martedì 10: Completa totalmente il progetto provando a scaricare dataset più grossi
# Mercoledì 11: Buffer progetto. Invia progetto ai professori e chiedi se va bene
# Giovedì 12: Inizia ripassone
# Venerdì 13 - Venerdì 20:

import torch

from MLGWSC_proposal.resnet import Block, ResNet


def test_block_identity() -> None:
    # Testing a block where no downsampling or channel expansion is needed
    batch_size = 2
    seq_len = 100
    in_channels = 256
    out_channels_1 = 64
    dim_expansion = 4

    # in_channels == out_channels_1 * dim_expansion (256 == 64 * 4) and stride == 1
    # This means the identity_downsample should be None
    block = Block(
        in_channels_1 = in_channels,
        out_channels_1 = out_channels_1,
        stride = 1,
        dim_expansion = dim_expansion
    )

    # Tensor with shape (2, 256, 100) for testing
    x = torch.randn(batch_size, in_channels, seq_len)
    out = block(x)

    assert block.identity_downsample is None
    assert isinstance(out, torch.Tensor)

    # Checking expected shape: (2, 64 * 4, 100)
    assert out.shape == (batch_size, out_channels_1 * dim_expansion, seq_len)
    assert not torch.isnan(out).any()


def test_block_downsample() -> None:
    # Testing a block where the timeseries dimension is halved and channels are expanded
    batch_size = 2
    seq_len = 100
    in_channels = 128
    out_channels_1 = 64
    dim_expansion = 4

    # Stride = 2 forces downsampling.
    # Also in_channels (128) != out_channels_1 * dim_expansion (256)
    # The identity_downsample must be activated
    block = Block(
        in_channels_1 = in_channels,
        out_channels_1 = out_channels_1,
        stride = 2,
        dim_expansion = dim_expansion
    )

    x = torch.randn(batch_size, in_channels, seq_len)
    out = block(x)

    assert block.identity_downsample is not None
    assert isinstance(out, torch.Tensor)
    # Timeseries shape must be halved (seq_len // 2)
    assert out.shape == (batch_size, out_channels_1 * dim_expansion, seq_len // 2)


def test_resnet152_forward() -> None:
    # Testing the complete ResNet152 architecture with mock data similar to
    # those that we are actually going to work on
    stages = [3, 8, 36, 3]
    input_channels = 2  # H1 and L1
    num_classes = 2     # Signal vs Noise

    batch_size = 4
    seq_len = 2048      # 1 second of data at 2048 Hz

    model = ResNet(
        stages = stages,
        input_channels = input_channels,
        num_classes = num_classes
    )

    x = torch.randn(batch_size, input_channels, seq_len)

    # We use torch.no_grad() during tests to avoid tracking gradients,
    # which saves memory and speeds up the test significantly.
    with torch.no_grad():
        out = model(x)

    assert isinstance(out, torch.Tensor)
    # The network must collapse the 2048 timeseries and output exactly 2 class logits
    assert out.shape == (batch_size, num_classes)
    assert not torch.isnan(out).any()
