.. _api:

API Reference
=============

Data Processing & Loading
-------------------------
This module handles the extraction, **whitening**, and **segmentation** of raw HDF5
strain data, as well as the creation of PyTorch `DataLoaders` with all the necessary
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
This module gives birth to `GW_ResNet152` model, which is the union of a **1D ResNet152**
[He2016]_ backbone and a **Deep Adaptive Normalization Layer (DAIN)** [Pas2019]_. The code for the
latter was just udated from [DainRepo]_ to be compatible with the newer PyTorch versions.

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
This module contains training and validation loops, implements an early stopping class and
a parameter grouping function for the `GW_ResNet152` architecture.

.. automodule:: MLGWSC_proposal.training
   :members:
   :undoc-members:
   :show-inheritance:
