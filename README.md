# MLGWSC_proposal

Repository for the Machine Learning project developed for the *Computing Methods for Experimental Physics and Data Analysis* exam (University of Pisa — UNIPI) by **Lorenzo Nepi**.

## Overview

**MLGWSC_proposal** was designes for the data and the evaluation metrics of the **First Machine Learning Gravitational-Wave Search Mock Data Challenge (MLGWSC-1)** (Schäfer et al.). The goal of the challenge was to determine
the coalescence times of compact objects from simulated GW strain timeseries.

Based on one of the original proposals for MLGWSC-1 (Nousi et al.), the package implements GPU-accelerated spectral whitening,  *Curriculum Learning* based on the SNR values of the strain segments, a customizable 1D **ResNet** (`GW_ResNet`) with optional **Deep Adaptive Input Normalization (DAIN)** (N. Passalis), and inference pipeline compliant with the official MLGWSC-1 submission format.

---

## Repository Structure

```text
MLGWSC_proposal/
├── docs/                           # Sphinx documentation source and configuration
│   └── source/
│       ├── _static/
│       │   └── custom.css
│       ├── api.rst                 # Auto-generated API reference
│       ├── bib.rst                 # Bibliography
│       ├── conf.py
│       ├── index.rst
│       └── installation.rst        # Installation
├── src/
│   └── MLGWSC_proposal/
│       ├── dain.py                 # Deep Adaptive Input Normalization (DAIN) layer
│       ├── data_manipulation.py    # Welch PSD estimation, whitening, and HDF5 segmentation
│       ├── dataloader.py           # PyTorch Datasets and Curriculum Learning DataLoaders
│       ├── inference.py            # Sliding-window inference, clustering, and MLGWSC-1 export
│       ├── resnet.py               # 1D ResNet backbone (Basic & Bottleneck blocks) and GW_ResNet
│       └── training.py             # Training/validation loops, EarlyStopping, and test evaluation
├── tests/                          # Pytest unit and integration test suite
│   ├── test_data_manipulation.py
│   ├── test_dataloader.py
│   ├── test_inference.py
│   ├── test_resnet.py
│   └── test_training.py
├── training_pipeline.ipynb         # Full training pipeline of a ResNet model
├── inference_pipeline.ipynb        # Full pipeline for time of coalescence estimation from raw data
├── .gitignore
├── LICENSE
├── README.md
├── noxfile.py                      # Automated sessions for testing, linting, and type-checking
└── pyproject.toml                  # Package metadata, dependencies, and tool configurations
```

---

## Installation

Clone the repository and install the package in editable mode (requires **Python >= 3.12**):

```bash
git clone [https://github.com/LorenzoNepi/MLGWSC_proposal.git](https://github.com/LorenzoNepi/MLGWSC_proposal.git)
cd MLGWSC_proposal
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

To install optional dependencies for development (`pytest`, `ruff`, `pylint`, `mypy`, `nox`) and documentation (`sphinx`, `pydata-sphinx-theme`):

```bash
pip install -e ".[dev, docs]"
```

---

## Pipeline Workflow

The `training_pipeline.ipynb` and `inference_pipeline.ipynb` contain an example of a full pipeline, from raw HDF5 file to a trained ResNet-like model able to infer coalesce times.

### 1. Data Preprocessing & Whitening
Raw `H1` and `L1` strain timeseries (sampled at `2048 Hz`) are processed on GPU/CPU via `MLGWSC_proposal.data_manipulation`:
* **Power Spectral Density (PSD):** Estimated per continuous block ($\ge 2\text{ h}$) using Welch's method with 50% overlapping Hann windows (`welch_psd`).
* **Frequency-Domain Whitening:** Signals are transformed via PyTorch's `rfft`, high-pass filtered below the dataset `low_frequency_cutoff`, normalized by the Amplitude Spectral Density ($\sqrt{\text{PSD}}$), transformed back via `irfft`, and center-cropped to remove edge artifacts (`whiten`).
* **Background & Foreground Extraction:** `torch_segmentation` extracts noise windows across continuous segments, while `foreground_segmentation` extracts windows around injected coalescence times ($t_c$) with randomized temporal offsets (`get_tc`) to avoid positional bias.

### 2. Curriculum Learning & Training
`MLGWSC_proposal.dataloader` and `MLGWSC_proposal.training` manage model training:
* **Curriculum Learning:** `Foreground_Dataset` orders injected signals by descending approximated SNR and progressively expands the training fraction (`snr_split`) over the first `cl_epochs` epochs.
* **Architecture:** `GW_ResNet` supports configurable 1D ResNet depths (e.g., ResNet-34 with `BasicBlock` or ResNet-152 with bottleneck `Block`) and an optional `DAIN` normalization layer with differential learning rates.
* **Optimization:** Trained with `BCEWithLogitsLoss`, `ReduceLROnPlateau` scheduling, and `EarlyStopping` checkpointing through the `train_gw_resnet` function, followed by ROC-AUC and loss evaluation via `gw_resnet_test`.

### 3. Continuous Inference & MLGWSC-1 ubmission standard
`MLGWSC_proposal.inference` evaluates continuous strain files using a sliding window with step `stride // 4` (`0.25 s` for `1.0 s` windows):
* Candidate triggers above `prob_threshold` are grouped via `cluster_detections`, retaining the peak probability timestamp per event cluster.
* `mlgwsc1_output` exports the final triggers to an HDF5 file containing the three required 1D datasets of equal length: `time` (GPS coalescence timestamps), `stat` (ranking statistic / probability), and `var` (timing tolerance window in seconds).

---

## References

* M.B. Schäfer, O. Zelenka et al. (2022), *MLGWSC-1: The first Machine Learning Gravitational-Wave Search Mock Data Challenge*, arXiv:2209.11146.
* GitHub page for [https://github.com/gwastro/ml-mock-data-challenge-1](MLGWSC-1).
* P. Nousi et al. (2023), *Deep Residual Networks for Gravitational Wave Detection*, arXiv:2211.01520.
* K. He, X. Zhang, S. Ren, & J. Sun (2016), *Deep Residual Learning for Image Recognition*, arXiv:1512.03385.
* N. Passalis, et al. (2019), *Deep Adaptive Input Normalization for Time Series Forecasting*, arXiv:1902.07892.
* GitHub page for the original [https://github.com/passalis/dain](DAIN) code.

---

## License and documentation

The full documentation can be found at the following page: [https://lorenzonepi.github.io/MLGWSC_proposal/](Docs).

This project is released under the terms specified in the [LICENSE](LICENSE) file.