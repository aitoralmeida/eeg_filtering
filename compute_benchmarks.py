import os, sys
from benchmark_metrics import *

if len(sys.argv) < 2:
    print("Please provide path to simulation results as argument...")
    sys.exit(1)

sim_path = sys.argv[1]
runs = max([int(run) for run in next(os.walk(sim_path))[1]]) # Number of runs executed

total_avg_rrmse_temporal = []
total_avg_rrmse_spectral = []
total_avg_cc = []

for run in range(1, runs + 1):
    nn_output_path = os.path.join(sim_path, str(run), "nn_output/")

    denoised = np.load(nn_output_path + "Denoiseoutput_test.npy")
    ground_truth = np.load(nn_output_path + "EEG_test.npy")

    avg_rrmse_temporal = np.mean([rrmse_temporal(denoised[sample,:], ground_truth[sample,:]) for sample in range(denoised.shape[0])])
    avg_rrmse_spectral = np.mean([rrmse_spectral(denoised[sample,:], ground_truth[sample,:]) for sample in range(denoised.shape[0])])
    avg_cc = np.mean([cc(denoised[sample,:], ground_truth[sample,:]) for sample in range(denoised.shape[0])])

    total_avg_rrmse_temporal.append(avg_rrmse_temporal)
    total_avg_rrmse_spectral.append(avg_rrmse_spectral)
    total_avg_cc.append(avg_cc)

    with open(nn_output_path + "benchmarks.txt", "w") as f:
        f.write("BENCHMARKS:\n")
        f.write(f"\t- Average RRMSE Temporal: {avg_rrmse_temporal:.3f}\n")
        f.write(f"\t- Average RRMSE Spectral: {avg_rrmse_spectral:.3f}\n")
        f.write(f"\t- Average CC: {avg_cc:.3f}\n")

with open(sim_path + "/benchmarks.txt", "w") as f:
        f.write("BENCHMARKS:\n")
        f.write(f"\t- Average RRMSE Temporal: {np.mean(total_avg_rrmse_temporal):.3f}\n")
        f.write(f"\t- Average RRMSE Spectral: {np.mean(total_avg_rrmse_spectral):.3f}\n")
        f.write(f"\t- Average CC: {np.mean(total_avg_cc):.3f}\n")

print("Finished writing benchmarks to file!")
sys.exit(0)
