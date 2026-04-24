# MIMICBIO_LS2N

## 📌 Description

Understanding and isolating alpha and beta cell contributions to pancreatic islets electrical activity is crucial for studying endocrine function and developing real-time bioelectrical monitoring systems. However, electrophysiological signals recorded from islet tissues are typically noisy and convoluted, making reliable signal decomposition challenging.

In this work, we develop and evaluate model-based signal decomposition methods inspired by neuromuscular iEMG analysis. Specifically, we adapt Bayesian sparse deconvolution approaches to separate spike trains associated with alpha and beta cells using only one channel. Our methodology is compatible with real-time processing and low-latency hardware constraints.

The performance of the proposed pipeline is assessed from MEA (Multi-Electrode Array) signals, showing that alpha/beta patterns can be reliably distinguished even in low-SNR conditions. This approach opens the door to embedded islet monitoring devices and closed-loop control strategies in bioelectronic medicine.

> **Context:** This project started as a Master's thesis/internship (Paul Martinet) and is now continued as part of a PhD research program in the context of Type 1 Diabetes. The long-term goal is to improve automated insulin delivery devices by enabling real-time monitoring of alpha/beta cell activity.

---

## 🧠 Algorithm Overview

The pipeline consists of two main phases:

### 1. Offline Phase (Initialization)

- **Spike detection** using thresholding (based on estimated noise level)
- **MUAP extraction and clustering** to identify individual motor unit action potentials
- **Weibull parameter estimation** (t₀ and β) for each source using ISI distributions
- **Initialization of Kalman filter** state and covariance matrices

### 2. Online Phase (Real-time Estimation)

- **Sequential processing** of incoming samples
- **Beam search** over possible spike combinations (`n_s` sequences)
- **Weibull hazard function** (`r(t)`) for spike probability
- **Kalman filter update** (or LMS alternative) for MUAP estimation
- **Parameter update** for t₀ and β using gradient descent

The algorithm is fully configurable via `config.yaml`, supporting grid search over key parameters.

---

## 📂 Project Structure

```
MIMICBIO_LS2N/
│
├── main.py                          # Main entry point
├── offline.py                       # Offline initialization phase
├── online.py                        # Online estimation phase
├── config.yaml                      # Configuration file
│
├── Functions/
│   ├── Algo2.py                     # Core algorithm (beam search, Kalman)
│   ├── Metrics.py                   # Evaluation metrics
│   ├── Utils.py                     # Utility functions (data loading, preprocessing)
│   │
│   └── Plots/
│       ├── save_offline_figure.py   # Offline visualizations
│       ├── save_spike_online_figure.py  # Online spike raster plots
│       ├── save_spectral_analysis.py    # Spectrogram and ACF
│       ├── save_weibull_fit_figure.py   # ISI distribution + Weibull fit
│       └── save_theta_history.py        # Parameter evolution (t₀, β)
│
├── Report/
│   ├── builder.py                   # HTML report generator
│   └── template.html                # Report template
│
└── experiments/
    └── test_run/                    # Experiment outputs
```

---

## ⚙️ Installation & Dependencies

### Requirements

- Python 3.10+
- Dependencies:

| Package | Purpose |
|---|---|
| `numpy` | Numerical computing |
| `scipy` | Signal processing, optimization |
| `matplotlib` | Static plotting |
| `plotly` | Interactive visualizations |
| `pyyaml` | Configuration parsing |
| `h5py` | HDF5 data loading |
| `gc`, `time`, `sys`, `os`, `json` | Standard library |

---

## 🚀 Usage

### Configuration (`config.yaml`)

All parameters are defined in `config.yaml`. Parameters that support grid search accept a list; single values are used as-is.

### Running the Pipeline

```bash
python main.py
```

The script will:
1. Load data (real `.h5`/`.bin` or synthetic)
2. Run offline initialization
3. Run online estimation
4. Generate an interactive HTML report with all figures and metrics

### Synthetic Signal Generation

A standalone script is provided to generate synthetic two-source signals with:
- Biphasic or monophasic MUAPs
- Adjustable refractory periods and firing rates
- Controllable noise level
- Downsampling support

There is already an option to activate it directly from the yaml. Example usage:

```python
from generate_synthetic_signal import generate_synthetic_signal

Y, U_true, H_true, Theta_true, isi_true, spike_idx_true, fs, ell_RI, offset_info, OFFSET_peaks = generate_synthetic_signal(
    config=config,
    duration_sec=150,
    amp1=40, amp2=30,
    noise_std=4,
    biphasic=True
)
```

### Data Format

- **`.h5` files** contain metadata and multi-channel recordings. Refer to `Test_h5file.ipynb` for structure details.
- **`.bin` files** are raw binary multi-channel recordings (no metadata).

---

## 📊 Visualizations

All figures are generated as interactive Plotly HTML files and aggregated into a single report.

### Offline Figures

| Figure | Description |
|---|---|
| Spike detection | Filtered signal with thresholds and detected spikes |
| MUAPs | Individual waveforms + average + ±1σ band per source |
| ISI distribution | Histogram with theoretical Weibull fit (t₀, β, t_R) |

### Online Figures

| Figure | Description |
|---|---|
| Signal reconstruction | Real vs. reconstructed signal overlay |
| Raster plot | Estimated spike trains (U) per source |
| Updated MUAPs | Extracted waveforms and average per source |
| Spectral analysis | Spectrogram of real vs. reconstructed signal and ACF|
| Residual analysis | Residual signal + histogram for normality check |
| Weibull fit (online) | ISI distribution with estimated Weibull parameters |

### Report

`Report/builder.py` generates a single HTML report containing:
- All figures (interactive Plotly)
- Model parameters used
- Performance metrics (RMSE, correlation, CPA)
- Source separation quality metrics (for synthetic data)

---

## 📏 Evaluation Metrics

### For Real Data

| Metric | Description |
|---|---|
| RMSE | Root mean square error between real and reconstructed signal |
| CPA | Correlation coefficient with peak alignment |

### For Synthetic Data (with Ground Truth)

| Metric | Description |
|---|---|
| TP / FP / FN | True positives, false positives, false negatives |
| Precision | TP / (TP + FP) |
| Recall | TP / (TP + FN) |
| F1-score | Harmonic mean of precision and recall |
| Alignment error | Difference (in samples) between detected and true spike peaks |
| Mode error | Most frequent alignment error (in samples) |

---

## ⚡ Optimizations Implemented

| Optimization | Description |
|---|---|
| Analytic 2×2 matrix inversion | Replaces `np.linalg.inv` for faster Weibull parameter updates |
| `update_theta_fast` | Avoids `np.outer` by manual computation |
| `kalman_update_fast` | Exploits ψ sparsity (only non-zero indices) |
| Precomputed valid transitions | Reduces combinatorial checks in beam search |
| Configurable Kalman / LMS | Allows switching to faster LMS when appropriate |
