# EEG Filtering & Denoising

Within this repository you will find:
- `benchmark_metrics.py`: Includes several implementations of quality assessment metrics used for benchmarking our different EEG denoising approaches.
- `compute_benchmarks.py`: Placed within the `results` folder of `EEGdenoiseNet`, computes the benchmark metrics (global and across runs) for the provided model simulation folder (as a command-line argument).