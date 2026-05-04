import cProfile
import os
import numpy as np
import sys
from datetime import datetime


from offline import run_offline
from online import run_online

import Functions.Utils as Utils  
import Functions.Metrics as Metrics
import Functions.Plots.save_spectral_analysis as save_spectral_analysis


import Functions.Plots.save_offline_figure as offline_plot
import Functions.Plots.save_spike_online_figure as online_plot
import Functions.Plots.save_residual_analysis as residual_plot
import Functions.Plots.save_weibull_fit_figure as weibull_plot
import Functions.Plots.save_theta_history as save_theta_history
from Functions.update_experiment_summary import update_experiment_summary

import Signal_simulation as sig_sim
from Report.builder import build_report
from Functions.params_grid_search import get_all_configs
import matplotlib.pyplot as plt

class Logger:
    """Clase para redirigir print a un archivo de log y mantener la salida en consola"""
    def __init__(self, log_file_path):
        self.terminal = sys.stdout
        self.log_file = open(log_file_path, 'w', encoding='utf-8')
        
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()  # Asegura que se escriba inmediatamente
        
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

def load_data(config, canal):

    path = config["data_path"]
    #path = r"{}".format(path)
    print(path)
    if path.endswith(".h5"):
        signal, fs, labels = Utils.read_multichannel_h5_data(
            path,
            ch=canal,
            return_fs_labels=True
        )

    elif path.endswith(".bin"):
        data = Utils.read_multichannel_bin_data(path, ch=canal, skip_s= config["skip_s"], length_s= config["length_s"])
        signal = data # data[:, config["channel"]]
        
        fs = config["sampling_rate"]

    else:
        raise ValueError("Unsupported format")

    return signal, fs


def main():


    #config = Utils.load_config()

    configs = get_all_configs('config.yaml')
    #Iterar sobre cada configuración
    for idx, config in enumerate(configs):
        print(f"\n{'='*60}")
        print(f"📊 EXPERIMENTO {idx+1}/{len(configs)}")
        print(f"Parametros son {config}")
        print(f"{'='*60}")
        
        canales = config["channel"]
        for canal in canales:
            exp_path = Utils.create_experiment_folder(config["output_dir"])
            

            # Configurar el logger para guardar todo lo que se imprime
            log_file_path = os.path.join(exp_path, f"execution_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
            logger = Logger(log_file_path)
            sys.stdout = logger
            
            print(f"{'='*80}")
            print(f"EXPERIMENT LOG")
            print(f"{'='*80}")
            print(f"📁 Experiment folder: {exp_path}")
            print(f"📝 Log file: {log_file_path}")
            print(f"⏰ Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*80}\n")

            # ---- LOAD ----
            if config["Sintetic"]:
                Y, _, _, _, _, spike_idx_true, fs_final, _, _, OFFSET_peaks = sig_sim.generate_synthetic_signal(config = config, 
                                                                                                                amp1 = config["amp1"], amp2 = config["amp2"], 
                                                                                                                h1_s1_std_div = config["h1_s1_std_div"], h1_s2_std_div = config["h1_s2_std_div"]
                                                                                                                , h2_s1_std_div = config["h2_s1_std_div"], h2_s2_std_div = config["h2_s2_std_div"]
                                                                                                                , noise_std = config["noise_std"]) #Senal simulada con los parametros groundtruth para validacion
                #Y, U_true, H_true, Theta_true, isi_true, spike_idx_true, fs_final, ell_final, offset_info, OFFSET_peaks
                
                #el tR y el init data lo cambio dentro de la funcion
                fs = fs_final
                signal = Y
                
            else:
                signal, fs = load_data(config, canal)
                print("Signal reading")
                print(f"Signal length: {len(signal)} | Fs: {fs}")

            # ---- Pre-treatment ----
            if not config["Sintetic"]:
                signal_filtered = Utils.preprocess_signal(signal, config) # Filter the whole signal, we should probably do this by chunks having the whole length in memory, not good
            else:
                signal_filtered = signal # Bc of variable name change
            
            # ---- Down Sampling ---#
            if not config["Sintetic"] and config["sampling_rate_DS"] != config["sampling_rate"]:
                signal_filtered, fs, t_R, decimation_factor= Utils.preprocess_with_downsampling(signal_filtered, fs, config)
            
                config["sampling_rate"] = fs 
                config["t_R"] = t_R
                config["init_data"] = int(config["init_data"]/decimation_factor)
            
            ## Correction of logic start of spike event versus spike peak
            #spike_idx_true = spike_idx_true + OFFSET_peaks
            # ---- OFFLINE RESULTS AND PLOTS ----
            print("---RUN OFFLINE---")
            
            
            offline_results = run_offline(signal_filtered[:config["init_data"]], fs, config) ## Give online de first N samples for offline training

            print(f"OFFLINE RESULTS:\n")
            print(f"Spikes: {len(offline_results['spike_idx'])}")
            print(f"Sigma: {offline_results['sigma']:.2f}")
            print(f"Threshold: {offline_results['threshold']:.2f}")

            ###
            print("---OFFLINE FIGURES GENERATION---")
            if config["Sintetic"]:
                plot_spikes, plot_muaps_offline, plot_isi = offline_plot.save_offline_figure(offline_results, 
                                                                                            signal_filtered[:config["init_data"]], config, exp_path, 
                                                                                            spike_idx_true= spike_idx_true, 
                                                                                            OFFSET_peaks=OFFSET_peaks) ## Its added after with the online plots in the figures dic
            else:
                plot_spikes, plot_muaps_offline, plot_isi = offline_plot.save_offline_figure(offline_results, 
                                                                                        signal_filtered[:config["init_data"]], config, exp_path, 
                                                                                        ) ## Its added after with the online plots in the figures dic


            # Ejecutar profiling
            profiler = cProfile.Profile()
            profiler.enable()

            # ---- ONLINE RESULTS AND PLOTS ----
            print("---ONLINE ESTIMATION---")
            
            online_results = run_online(signal_filtered[config["init_data"]:config["init_data"]+int(config["online_s"]*fs)], offline_results, config) ## por ahora puse hasta 10 segundos de analisis par que no dure tanto
            

            profiler.disable()

            stats_data, categories = Utils.generate_profile_report(profiler, exp_path, top_n=20)



            corr, rmse, Y_pred = Metrics.compute_cpa_score(online_results["Y"], online_results["U_est"], online_results["H_est"]) ## al coger la senalque devuelve el online, es la misma filtered
            #corr, rmse, Y_pred = [1,2,3]
            #t0 = online_results["Theta_est"][0][0]
            #beta = online_results["Theta_est"][0][1]
            #isi_per_mu = Metrics.extract_isi_per_mu(online_results["U_est"])
            #Metrics.log_likelihood_weibull_refractory(isi_per_mu, online_results["Theta_est"], config["t_R"]) # arreglarla para el nuevo cambio de MUs
            #print("Log-likelihood:", online_results["log_likelihood"])

            # ==================================================
            # 🔹 GUARDAR RESULTADOS OFFLINE SEPARADOS POR MU
            # ==================================================

            # Obtener número de MUs desde config
            n_mus = config["n_sources"]  # 2 en tu caso
            params_per_mu = 40  # Longitud fija de cada MUAP

            # 1. Guardar waveforms individuales (ya separados en lista)
            if offline_results["waveforms_per_mu"] is not None:
                for mu_idx in range(n_mus):
                    waveforms = offline_results["waveforms_per_mu"][mu_idx]
                    np.save(f"{exp_path}/waveforms_MU{mu_idx+1}.npy", waveforms)
                print(f"   ✅ Guardados waveforms para {n_mus} MUs")

            # 2. Guardar mean waveforms (H0) - Separar el H concatenado
            if offline_results["mean_waveforms_per_mu"] is not None:
                # Verificar si es lista o array concatenado
                if isinstance(offline_results["mean_waveforms_per_mu"], (list, tuple)):
                    # Ya está en formato lista
                    for mu_idx in range(n_mus):
                        mean_wf = offline_results["mean_waveforms_per_mu"][mu_idx]
                        np.save(f"{exp_path}/mean_waveform_MU{mu_idx+1}.npy", mean_wf)
                else:
                    # Está concatenado (array de n_mus * params_per_mu)
                    H_concatenated = offline_results["mean_waveforms_per_mu"]
                    for mu_idx in range(n_mus):
                        start = mu_idx * params_per_mu
                        end = start + params_per_mu
                        mean_wf = H_concatenated[start:end]
                        np.save(f"{exp_path}/mean_waveform_MU{mu_idx+1}.npy", mean_wf)
                print(f"   ✅ Guardados H0 (mean waveforms) para {n_mus} MUs")

            # 3. Guardar ISI separado por MU
            if offline_results["isi_per_mu"] is not None:
                if isinstance(offline_results["isi_per_mu"], (list, tuple)):
                    for mu_idx in range(n_mus):
                        isi = offline_results["isi_per_mu"][mu_idx]
                        if len(isi) > 0:
                            np.save(f"{exp_path}/isi_MU{mu_idx+1}.npy", isi)
                else:
                    # Caso de una sola MU
                    np.save(f"{exp_path}/isi_MU1.npy", offline_results["isi_per_mu"])
                print(f"   ✅ Guardados ISI para {n_mus} MUs")

            # 4. Guardar información de spikes (común para todas las MUs)
            if offline_results["spike_idx"] is not None:
                np.save(f"{exp_path}/spike_idx.npy", offline_results["spike_idx"])
                print(f"   ✅ Guardados índices de spikes ({len(offline_results['spike_idx'])} spikes)")


            print(f"\n📁 Resultados offline guardados en: {exp_path}")
            print(f"   • {n_mus} unidades motoras (config['n_sources'] = {n_mus})")
            for mu_idx in range(n_mus):
                n_spikes = len(offline_results["waveforms_per_mu"][mu_idx]) if offline_results["waveforms_per_mu"] else 0
                print(f"   • MU{mu_idx+1}: {n_spikes} spikes")


            print("✅ Save done for Offline")

            #Utils.save_online_results(online_results, exp_path)
            if config["Sintetic"]:
                plot_online, plot_MUAP = online_plot.save_spike_online_figure(online_results, online_results["Y"], 
                                                                            config, exp_path, spike_idx_true= spike_idx_true, 
                                                                            OFFSET_peaks=OFFSET_peaks) ## [config["init_data"]:config["init_data"]+int(config["online_s"]*fs)]
            else:
                plot_online, plot_MUAP = online_plot.save_spike_online_figure(online_results, online_results["Y"], 
                                                                            config, exp_path) ## [config["init_data"]:config["init_data"]+int(config["online_s"]*fs)]


            plot_residual, plot_residual_hist, residuals = residual_plot.save_residual_analysis(online_results["Y"], Y_pred, config, exp_path)

            qq_plot = residual_plot.save_qq_plot(residuals, config["sampling_rate"],exp_path)

            plot_spectral, plot_acf = save_spectral_analysis.save_spectral_analysis(online_results["Y"], Y_pred, config, exp_path)

            plot_weibull = weibull_plot.save_weibull_fit_figure(online_results, config, exp_path)

            ### graficar theta history
            theta_history = online_results["theta_history"]
            
            save_theta_history.plot_theta_individual_mus(theta_history, exp_path)

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
            "acf": plot_acf,
            "weibull": plot_weibull
            }
            metrics = {
            "rmse": rmse,
            "corr": corr
            
            }


            build_report(figures, metrics, config, exp_path)


            print("✅ Save done for Online")
            
            # 🔥 Cerrar el log y restaurar stdout
            print(f"\n{'='*80}")
            print(f"⏰ End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"📝 Log saved to: {log_file_path}")
            print(f"{'='*80}")
            
            sys.stdout = logger.terminal
            logger.log_file.close()
            Utils.save_config(config, exp_path)
            update_experiment_summary(
            exp_path=exp_path,
            config=config,
            base_output_dir=config["output_dir"],
            excel_name="experiments_summary.xlsx")


if __name__ == "__main__":
    main()