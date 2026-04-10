##main.py

import numpy as np
from offline import run_offline
from online import run_online
import matplotlib.pyplot as plt
import os
import plotly.graph_objects as go
from tqdm import tqdm

# 🔥 TUS FUNCIONES
import Functions.Utils as Utils  #save_weibull_fit_figure, read_multichannel_h5_data, read_multichannel_bin_data, load_config, create_experiment_folder, save_config, save_online_results, save_interactive_online_figure
import Functions.Metrics as Metrics
import Functions.Plots.save_spectral_analysis as save_spectral_analysis


import Functions.Plots.save_offline_figure as offline_plot
import Functions.Plots.save_spike_online_figure as online_plot
import Functions.Plots.save_residual_analysis as residual_plot
import Functions.Plots.save_weibull_fit_figure as weibull_plot
from Report.builder import build_report

def load_data(config):

    path = config["data_path"]
    #path = r"{}".format(path)
    print(path)
    if path.endswith(".h5"):
        signal, fs, labels = Utils.read_multichannel_h5_data(
            path,
            ch=config["channel"],
            return_fs_labels=True
        )

    elif path.endswith(".raw"):
        data = Utils.read_multichannel_bin_data(path, ch=config["channel"])
        signal = data # data[:, config["channel"]]
        fs = config["sampling_rate"]

    else:
        raise ValueError("Unsupported format")

    return signal, fs


def main():
    config = Utils.load_config()

    exp_path = Utils.create_experiment_folder(config["output_dir"])
    Utils.save_config(config, exp_path)

    print("📁 Experiment:", exp_path)

    # ---- LOAD ----
    signal, fs = load_data(config)

    print(f"Signal length: {len(signal)} | Fs: {fs}")

    # ---- Pre-treatment ----

    signal_filtered = Utils.preprocess_signal(signal, config) # Filter de whole signal, we should probably do this by chunks having the whole length in memory, not good


    # ---- OFFLINE RESULTS AND PLOTS ----
    offline_results = run_offline(signal_filtered[:config["init_data"]], fs, config) ## Give online de first N samples for offline training

    print(f"OFFLINE RESULTS:\n")
    print(f"Spikes: {len(offline_results['spike_idx'])}")
    print(f"Sigma: {offline_results['sigma']:.2f}")
    print(f"Threshold: {offline_results['threshold']:.2f}")

    plot_spikes, plot_muaps_offline, plot_isi = offline_plot.save_offline_figure(offline_results, signal_filtered[:config["init_data"]], config, exp_path) ## Its added after with the online plots in the figures dic

    # ---- ONLINE RESULTS AND PLOTS ----
    online_results = run_online(signal_filtered[config["init_data"]:config["init_data"]+5*fs], offline_results, config) ## por ahora puse hasta 10 segundos de analisis par que no dure tanto
    #breakpoint()
    corr, rmse, Y_pred = Metrics.compute_cpa_score(online_results["Y"], online_results["U_est"], online_results["H_est"]) ## al coger la senalque devuelve el online, es la misma filtered
    breakpoint()
    t0 = online_results["Theta_est"][0][0]
    beta = online_results["Theta_est"][0][1]
    isi_per_mu = Metrics.extract_isi_per_mu(online_results["U_est"])

    
    
    LogL = Metrics.log_likelihood_weibull_refractory(isi_per_mu, online_results["Theta_est"], config["t_R"])
    #print("Log-likelihood:", online_results["log_likelihood"])

    # ---- SAVE ----
    np.save(f"{exp_path}/spikes.npy", offline_results["spike_idx"])
    if offline_results["waveforms"] is not None:
        np.save(f"{exp_path}/waveforms.npy", offline_results["waveforms"])
    if offline_results["mean_waveform"] is not None:
        np.save(f"{exp_path}/H0.npy", offline_results["mean_waveform"])
     # 🔹 ISI
    if offline_results["isi"] is not None:
        np.save(f"{exp_path}/isi.npy", offline_results["isi"])


    print("✅ Save done for Offline")

    #Utils.save_online_results(online_results, exp_path)
    plot_online, plot_MUAP = online_plot.save_spike_online_figure(online_results, online_results["Y"], config, exp_path)


    plot_residual, plot_residual_hist = residual_plot.save_residual_analysis(online_results["Y"], Y_pred, config, exp_path)

    plot_spectral = save_spectral_analysis.save_spectral_analysis(online_results["Y"], Y_pred, config, exp_path)

    plot_weibull = weibull_plot.save_weibull_fit_figure(online_results, config, exp_path)
    
    

    #plot_spikes, plot_muaps_offline, plot_isi
    figures = {

    "spikes": plot_spikes,
    "muaps_offline": plot_muaps_offline,
    "isi": plot_isi,

    "online": plot_online,
    "online_MUAP": plot_MUAP,
    "residual": plot_residual,
    "residual_hist": plot_residual_hist,   # ← new
    "spectral": plot_spectral,
    "weibull": plot_weibull
    }
    metrics = {
    "rmse": rmse,
    "corr": corr,
    "LogL (log_likelihood_total, log_likelihood_mean, log_likelihood_per_each_mu)": LogL
    }


    build_report(figures, metrics, config, exp_path)


    print("✅ Save done for Online")


if __name__ == "__main__":
    main()