##Offline
import numpy as np
from Functions.Utils import remove_close_spikes
import matplotlib.pyplot as plt

def detect_spikes(signal, fs, config):
    sigma = np.std(signal)
    threshold = config["threshold_sigma"] * sigma
    
    spike_idx = np.where(np.abs(signal) > threshold)[0]
    
    #  Eliminar spikes cercanos dado el periodo refractario; esto no es biologicamente del todo correcto en realidad.
    min_dist = int(config["t_R"] * fs / 1000)
    spike_idx = remove_close_spikes(spike_idx, min_dist)
    
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


def extract_waveforms_separate_muaps(signal, spike_idx, fs, config):
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
    - waveforms_per_mu: lista con [waveforms_pos, waveforms_neg]
    - mean_waveforms_per_mu: lista con [mean_pos, mean_neg]
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
    threshold_uv = config["threshold_uv"]  # umbral en µV
    
    forms_pos = []
    forms_neg = []
    mu_assignment = []  # 0 para positiva, 1 para negativa
    
    for t in spike_idx:
        if t-pre >= 0 and t+post < len(signal):
            snippet = signal[t-pre:t+post]
            
            # Clasificar por polaridad
            
            if np.max(snippet) > threshold_uv:
                forms_pos.append(snippet)
                mu_assignment.append(0)  # MU positiva
            elif np.min(snippet) < -threshold_uv:
                forms_neg.append(snippet)
                mu_assignment.append(1)  # MU negativa
            # Nota: spikes que no superan el umbral se ignoran
    
    # Convertir a arrays
    waveforms_per_mu = []
    mean_waveforms_per_mu = []
    
    if forms_pos:
        waveforms_pos = np.array(forms_pos)
        mean_pos = np.mean(waveforms_pos, axis=0)
        waveforms_per_mu.append(waveforms_pos)
        mean_waveforms_per_mu.append(mean_pos)
        print(f"✅ MU positiva: {len(forms_pos)} spikes")
    else:
        waveforms_per_mu.append(None)
        mean_waveforms_per_mu.append(None)
        print("⚠️ No se detectaron spikes positivos")
    
    if forms_neg:
        waveforms_neg = np.array(forms_neg)
        mean_neg = np.mean(waveforms_neg, axis=0)
        waveforms_per_mu.append(waveforms_neg)
        mean_waveforms_per_mu.append(mean_neg)
        print(f"✅ MU negativa: {len(forms_neg)} spikes")
    else:
        waveforms_per_mu.append(None)
        mean_waveforms_per_mu.append(None)
        print("⚠️ No se detectaron spikes negativos")
    
    
    mu_assignment = np.array(mu_assignment) if mu_assignment else None
    # solo ahre la concatenacion dentro del algo 2, es mejor
    #mean_waveforms_per_mu = np.concatenate(mean_waveforms_per_mu) ## So that kalman con handle matrix operations
    
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
            isi = np.diff(times)
            isi_per_mu.append(isi)
        else:
            isi_per_mu.append(np.array([]))
    
    return isi_per_mu


def run_offline(signal_filtered, fs, config):
    """
    Ejecuta el pipeline offline completo.
    
    Parámetros:
    - signal_filtered: señal ya filtrada
    - fs: sampling rate
    - config: diccionario con parámetros:
        - threshold_sigma: umbral para detección
        - t_R: período refractario (ms)
        - window_pre_ms, window_post_ms: ventana para waveforms
        - n_MU: número de unidades (1 o 2)
        - threshold_uv: umbral para separar polaridades (si n_MU=2)
    """
    
    # Detección de spikes
    spike_idx, sigma, threshold = detect_spikes(signal_filtered, fs, config)
    print(f"🔍 Spikes detectados: {len(spike_idx)}")
    
    # Extracción de waveforms según n_MU
    n_MU = config["n_sources"]
    
    if n_MU == 2:
        
        # Separar por polaridad (positiva/negativa)
        waveforms_per_mu, mean_waveforms_per_mu, mu_assignment = extract_waveforms_separate_muaps(
            signal_filtered, spike_idx, fs, config
        )
        
        # Calcular ISI por MU
        isi_per_mu = compute_isi_per_mu(spike_idx, mu_assignment, fs)
    else:
        # Una sola MU (todas las waveforms juntas)
        waveforms, mean_waveform = extract_waveforms(signal_filtered, spike_idx, fs, config)
        waveforms_per_mu = [waveforms]
        mean_waveforms_per_mu = [mean_waveform]
        mu_assignment = None
        isi_per_mu = [compute_isi(spike_idx, fs)]
    
    return {
        "signal": signal_filtered,
        "spike_idx": spike_idx,
        "sigma": sigma,
        "threshold": threshold,
        "waveforms_per_mu": waveforms_per_mu,
        "mean_waveforms_per_mu": mean_waveforms_per_mu,
        "mu_assignment": mu_assignment,
        "isi_per_mu": isi_per_mu,
        "n_MU": n_MU
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