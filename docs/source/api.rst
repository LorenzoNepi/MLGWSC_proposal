.. _api:

API Reference
=============

Data Processing & Loading
-------------------------
These modules handle the extraction, **whitening**, and **segmentation** of raw HDF5
strain data, as well as the creation of PyTorch ``DataLoaders`` with all the necessary
parameters to implement **curriculum learning** during training.

.. automodule:: MLGWSC_proposal.data_manipulation
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: MLGWSC_proposal.dataloader
   :members:
   :undoc-members:
   :show-inheritance:

Neural Network Architectures
----------------------------
These modules implement the ``GW_ResNet`` architecture, which combines a **1D ResNet**
[He16]_ backbone (supporting both basic and bottleneck blocks for customizable depths)
with an optional **Deep Adaptive Input Normalization (DAIN)** [Pa19]_ layer. The code for the
latter was adapted from [DainRepo]_ to be compatible with newer PyTorch versions.

.. automodule:: MLGWSC_proposal.resnet
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: MLGWSC_proposal.dain
   :members:
   :undoc-members:
   :show-inheritance:

Training Pipeline
-----------------
This module contains training and validation loops, implements an early stopping class, and
provides a parameter grouping function for the ``GW_ResNet`` architecture.

.. automodule:: MLGWSC_proposal.training
   :members:
   :undoc-members:
   :show-inheritance: