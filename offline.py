##Offline
import numpy as np
from Functions.Utils import remove_close_spikes
import matplotlib.pyplot as plt
import Functions.weibull_params_init as init_tetha

def detect_spikes(signal, fs, config):
    
    
    if config["threshold_method"]== "std":
        sigma = np.std(signal) # cambiar por Median standard deviation ?
        threshold = config["threshold_sigma"] * sigma
    elif config["threshold_method"] == "percentil":
        threshold = np.percentile(np.abs(signal), 99.5)
    elif config["threshold_method"] == "median":
        med = np.median(signal)
        
        sigma = np.median(np.abs(signal - med)) / 0.6745

        #A = np.percentile(np.abs(signal), 99.9)

        #snr = A / sigma
        #snr_db = 20*np.log10(snr)
        
        threshold = config["threshold_sigma"] * sigma
       
    print(f"For method {config["threshold_method"]} the threshold is {threshold}")
    
    spike_idx = np.where(np.abs(signal) > threshold)[0]
    
    #  Eliminar spikes cercanos dado el periodo refractario; esto no es biologicamente del todo correcto en realidad.
    min_dist = int(config["spike_min_d_ms"] * fs / 1000) # convertirlo en muestras
    spike_idx = remove_close_spikes(spike_idx, min_dist)
    print(spike_idx)
    
    return spike_idx, sigma, threshold


def extract_waveforms(signal, spike_idx, fs, config):
    pre = int(config["window_pre_ms"] * fs / 1000)
    post = int(config["window_post_ms"] * fs / 1000)
    
    forms = []
    
    for t in spike_idx:
        if t-pre >= 0 and t+post < len(signal):
            snippet = signal[t-pre:t+post]
            forms.append(snippet)
    
    if len(forms) == 0:
        return None, None
    
    waveforms = np.array(forms)
    mean_waveform = np.mean(waveforms, axis=0)
    
    return waveforms, mean_waveform


def extract_waveforms_separate_muaps(signal, spike_idx, fs, config, threshold):
    """
    Extrae waveforms separando por polaridad (positivas y negativas)
    cuando se asume que hay 2 MUAPs (n_MU = 2).
    
    Parámetros:
    - signal: señal filtrada
    - spike_idx: índices de los spikes detectados
    - fs: sampling rate (Hz)
    - config: diccionario con parámetros:
        - window_pre_ms, window_post_ms: ventana en ms
        - threshold_uv: umbral en µV para separar positivo/negativo
        - n_MU: número de unidades motoras (2 para esta separación)
    
    Retorna:
    - waveforms_per_mu: lista con [waveforms_pos, waveforms_neg] (solo los que existen)
    - mean_waveforms_per_mu: lista con [mean_pos, mean_neg] (solo los que existen)
    - mu_assignment: array con asignación de cada spike a MU (0=positiva, 1=negativa)
    """
    
    # Verificar que n_MU sea 2
    if config["n_sources"] != 2:
        print("⚠️ extract_waveforms_separate_muaps solo se usa con n_MU=2")
        # Fallback a extract_waveforms normal
        waveforms, mean_waveform = extract_waveforms(signal, spike_idx, fs, config)
        return [waveforms], [mean_waveform], None
    
    # Parámetros de ventana
    pre = int(config["window_pre_ms"] * fs / 1000)
    post = int(config["window_post_ms"] * fs / 1000)
    threshold_uv = threshold  # umbral en µV
    
    forms_pos = []
    forms_neg = []
    mu_assignment = []  # 0 para positiva, 1 para negativa
    
    for t in spike_idx:
        if t-pre >= 0 and t+post < len(signal):

            snippet = signal[t-pre:t+post]

            max_val = np.max(snippet)
            min_val = np.min(snippet)

            # verificar que supera umbral
            if max(abs(max_val), abs(min_val)) < threshold_uv:
                continue

            # polaridad dominante
            if abs(max_val) >= abs(min_val):
                forms_pos.append(snippet)
                mu_assignment.append(0)

            else:
                forms_neg.append(snippet)
                mu_assignment.append(1)
    
    # Convertir a arrays - solo agregar los que existen
    waveforms_per_mu = []
    mean_waveforms_per_mu = []
    
    if forms_pos:
        waveforms_pos = np.array(forms_pos)
        mean_pos = np.mean(waveforms_pos, axis=0)
        waveforms_per_mu.append(waveforms_pos)
        mean_waveforms_per_mu.append(mean_pos)
        print(f"✅ MU positiva: {len(forms_pos)} spikes")
    
    if forms_neg:
        waveforms_neg = np.array(forms_neg)
        mean_neg = np.mean(waveforms_neg, axis=0)
        waveforms_per_mu.append(waveforms_neg)
        mean_waveforms_per_mu.append(mean_neg)
        print(f"✅ MU negativa: {len(forms_neg)} spikes")
    
    if not forms_pos and not forms_neg:
        print("⚠️ No se detectaron spikes ni positivos ni negativos")
        waveforms_per_mu = []
        mean_waveforms_per_mu = []
    
    mu_assignment = np.array(mu_assignment) if mu_assignment else None
    
    #breakpoint()
    return waveforms_per_mu, mean_waveforms_per_mu, mu_assignment


def compute_isi(spike_idx, fs):
    times = spike_idx / fs
    isi = np.diff(times)
    return isi


def compute_isi_per_mu(spike_idx, mu_assignment, fs):
    """
    Calcula ISI por separado para cada MU.
    
    Parámetros:
    - spike_idx: índices de todos los spikes
    - mu_assignment: asignación de cada spike a MU (0, 1, ...)
    - fs: sampling rate
    
    Retorna:
    - isi_per_mu: lista de arrays con ISIs por MU
    """
    if mu_assignment is None:
        return [compute_isi(spike_idx, fs)]
    
    n_mu = len(np.unique(mu_assignment))
    spikes_per_mu = [[] for _ in range(n_mu)]
    
    for idx, mu in zip(spike_idx, mu_assignment):
        spikes_per_mu[mu].append(idx)
        
    isi_per_mu = []
    for spikes in spikes_per_mu:
        if len(spikes) > 1:
            spikes_array = np.array(spikes)
            times = spikes_array / fs
            isi = np.diff(times) ## da el isi en msegundos
            isi_per_mu.append(isi)
        else:
            isi_per_mu.append(np.array([]))
    
    return isi_per_mu



def run_offline(signal_filtered, fs, config):
    """
    Offline estimations for the algortihm initialisation
    
    Parámetros:
    - signal_filtered: señal ya filtrada
    - fs: sampling rate
    - config: diccionario con parámetros:
        - threshold_sigma: umbral para detección
        - t_R: período refractario (ms) - lo pase a muestras
        - window_pre_ms, window_post_ms: ventana para waveforms
        - n_MU: número de unidades (1 o 2)
        - threshold_uv: umbral para separar polaridades (si n_MU=2)
    """
    
    # Detección de spikes
    spike_idx, sigma, threshold = detect_spikes(signal_filtered, fs, config) # already min dist spikes
    print(f"🔍 Spikes detectados: {len(spike_idx)}")
    
    # Extracción de waveforms según n_MU
    n_MU = config["n_sources"]
    #breakpoint()
    if n_MU == 2:
        
        # Separar por polaridad (positiva/negativa)
        waveforms_per_mu, mean_waveforms_per_mu, mu_assignment = extract_waveforms_separate_muaps(
            signal_filtered, spike_idx, fs, config, threshold
        )
        
        # Calcular ISI por MU
        #breakpoint()
        isi_per_mu = compute_isi_per_mu(spike_idx, mu_assignment, fs)
        # Intercambiar los t_R de acuerdo a cual es la waveform con con mas waveforms. Primera implementacion rustica peroe efectiva y rapida, la que tenga mas, tiene el t_R menor

    else:
        # Una sola MU (todas las waveforms juntas)
        waveforms, mean_waveform = extract_waveforms(signal_filtered, spike_idx, fs, config)
        waveforms_per_mu = [waveforms]
        mean_waveforms_per_mu = [mean_waveform]
        mu_assignment = None
        isi_per_mu = [compute_isi(spike_idx, fs)]

    ## Chequeo de si se encontraron mas MU

    if len(isi_per_mu) == 1:
        print(str(config["n_sources"]) + "SOURCES WERE SELECTED, however as only 1 type of waveform was found in the initialization, algorithm goes back to 1 source." )
        
        config["n_sources"] = 1


    ## Estimación parámetros Weibull iniciales - FOllowing initialisation that Paul said, he didn't provide it, so i added it.

    if config["weibull_init_method"] == "LBFGS":
        t0_init, beta_init = init_tetha.estimate_weibull_LBFGS(
            isi_per_mu,
            fs,
            config
        )
    elif config["weibull_init_method"] == "Bayesian":
        t0_init, beta_init = init_tetha.estimate_weibull_bayesian(
            isi_per_mu,
            fs,
            config
        )

    elif config["weibull_init_method"] == "Moments":
        t0_init, beta_init = init_tetha.estimate_weibull_moments(
            isi_per_mu,
            fs,
            config
        )

    elif config["weibull_init_method"] == "GridSearch":
        t0_init, beta_init = init_tetha.estimate_weibull_grid_search(
            isi_per_mu,
            fs,
            config
        )
    elif config["weibull_init_method"] == "Manual":
        t0_init= config["t0"]
        beta_init =config["beta"]
    elif config["weibull_init_method"] == "None":
        t0_init= np.array([np.random.uniform(2000, 5000)],
                          np.random.uniform(2000, 5000))
        beta_init = np.array([np.random.uniform(1, 5)],
                          np.random.uniform(1, 5))

    
    ## Estimacion de la varianza del ruido v
    
    ## Es una variante del estimador robusto basado en la MAD (Median Absolute Deviation).
    v = np.median((signal_filtered - np.median(signal_filtered))**2) #/ (0.6745**2)
    print("Varianza del ruido estimado usando variante del estimador robusto basado en la MAD (Median Absolute Deviation) : ")
    print("v = " + str(v) + "Sin nromalizacion ? o con normalizacion, que es mejor ?")
    print(f"Sigma, desvicion estandar pero sin mask de picos, pq no los sabemos. Es el que se usa para el umbral de picos : {sigma}")

    mask = np.abs(signal_filtered) < config["threshold_sigma"]*sigma
    noise_segment = signal_filtered[mask]  # realmente no tiene mucho sentido hacer esto para el MAD ya que de por si la mediana es robusta a outliers. (lo dejo en el std para comparar)
    print(f"Sigma, desvicion estandar pero con mask de picos, pq no los sabemos. Lo caluclo ahora para comparar : {np.std(noise_segment)}")
    
    return {
    "signal": signal_filtered,
    "spike_idx": spike_idx,
    "sigma": sigma, # es el ruido pero teniendo en cuenta
    "threshold": threshold,
    "waveforms_per_mu": waveforms_per_mu,
    "mean_waveforms_per_mu": mean_waveforms_per_mu,
    "mu_assignment": mu_assignment,
    "isi_per_mu": isi_per_mu,
    "t0_init": t0_init,
    "beta_init": beta_init,
    "n_MU": n_MU,
    "noise_variance": v
}


def plot_waveforms_by_mu(offline_results, fs, config, save_path=None):
    """
    Visualiza las waveforms separadas por MU (útil para n_MU=2).
    """
    mean_waveforms = offline_results["mean_waveforms_per_mu"]
    n_MU = offline_results["n_MU"]
    
    pre_ms = config["window_pre_ms"]
    post_ms = config["window_post_ms"]
    time_axis_ms = np.linspace(-pre_ms, post_ms, len(mean_waveforms[0]))
    
    plt.figure(figsize=(12, 4))
    
    colors = ['blue', 'red', 'green', 'purple']
    
    for i, mean_wf in enumerate(mean_waveforms):
        if mean_wf is not None:
            label = f"MU {i+1} (n={offline_results['waveforms_per_mu'][i].shape[0] if offline_results['waveforms_per_mu'][i] is not None else 0})"
            plt.plot(time_axis_ms, mean_wf, label=label, color=colors[i % len(colors)], linewidth=2)
    
    plt.xlabel("Tiempo (ms)")
    plt.ylabel("Amplitud (µV)")
    plt.title(f"MUAPs estimados — {n_MU} unidades")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"✅ Figura guardada: {save_path}")
    
    plt.show()