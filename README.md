# EEG Filtering & Denoising

Within this repository you will find:
- `benchmark_metrics.py`: Includes several implementations of quality assessment metrics used for benchmarking our different EEG denoising approaches.
- `compute_benchmarks.py`: Placed within the `results` folder of `EEGdenoiseNet`, computes the benchmark metrics (global and across runs) for the provided model simulation folder (as a command-line argument).
- `generate-tES.py`: This script generates tES signals compatible with the EEGdenoiseNet framework. Must be placed withing the `data` folder. **Please note that the resulting signal is stored in $\mu V$, not in $V$.**
- `Classic (Shallow) Algorithms`: This folder contains three *shallow* approaches to EEG denoising under tES:
    * `FastICA`: Using **Independent Component Analysis** (ICA) to extract the denoised signal (includes a novel approach to solving the inverted signal problem).
    * `Wavelets`: Provides an example of training a regression model which uses Wavelets extracted from the noisy signal as input features.
    * `Empirical Mode Decomposition (EMD)`: Provides an implementation of a simple algorithm which uses **Empirical Mode Decomposition** (EMD) and **Mutual Information/Information Gain** as a signal reconstruction criterion. 