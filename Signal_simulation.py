## Generate artificial signal
import numpy as np
from Functions import Algo2


# ==========================================================
# Biphasic waveform
# ==========================================================
def generate_biphasic_muap(length=40, c1_center=0.275, c2_center=0.55, 
                           s1_std_div=12, s2_std_div=8, g2_amp=0.45, polarity=1):
    """
    Genera MUAP bifásico.
    
    Parámetros:
    -----------
    length : int
        Número de muestras (para fs=10000, length=40 son 4ms)
    c1_center : float
        Posición del pico principal (0.275 = 11/40)
    c2_center : float
        Posición del segundo pico
    s1_std_div, s2_std_div : int
        Control de anchura
    g2_amp : float
        Amplitud del segundo pico
    polarity : +1 o -1
        Polaridad del pico principal
    """
    x = np.arange(length)
    c1 = int(length * c1_center)
    c2 = int(length * c2_center)
    s1 = length / s1_std_div
    s2 = length / s2_std_div

    g1 = np.exp(-0.5 * ((x - c1) / s1) ** 2)
    g2 = g2_amp * np.exp(-0.5 * ((x - c2) / s2) ** 2)

    h = g1 - g2

    if polarity == -1:
        h = -h

    h = h / np.max(np.abs(h))
    OFFSET_peak = c1
    
    return h, OFFSET_peak


# ==========================================================
# Monophasic waveform
# ==========================================================
def generate_monophasic_muap(length=40, polarity=1):
    """Genera MUAP monofásico."""
    x = np.arange(length)
    c1 = int(length * 0.30)
    s1 = length / 10
    h = np.exp(-0.5 * ((x - c1) / s1) ** 2)
    
    tail_start = c1
    tail = np.zeros(length)
    idx = x >= tail_start
    tail[idx] = 0.15 * np.exp(-(x[idx] - tail_start) / (length / 8))
    h = h + tail
    
    if polarity == -1:
        h = -h
    
    h = h / np.max(np.abs(h))
    
    return h


# ==========================================================
# Simular UNA fuente tipo MATLAB
# ==========================================================
def simulate_one_source(N, H, theta, t_R):
    """Simula una fuente de spikes."""
    ell = len(H)
    Tn = 0
    phi = np.zeros(ell)
    U = np.zeros(N, dtype=np.int8)
    Y = np.zeros(N)
    spike_times = []

    for n in range(N):
        U[n] = 1 if Tn == 0 else 0
        if U[n] == 1:
            spike_times.append(n)
        
        phi[1:] = phi[:-1]
        phi[0] = U[n]
        Y[n] = np.dot(phi, H)
        Pn = Algo2.r(Tn + 1, theta, t_R)
        
        if np.random.rand() < Pn:
            Tn = 0
        else:
            Tn += 1

    spike_times = np.array(spike_times)
    if len(spike_times) >= 2:
        isi = np.diff(spike_times)
    else:
        isi = np.array([])

    return U, Y, isi, spike_times


# ==========================================================
# Medir offset teórico de una MUAP (Inicio de evento vs pico del evento determinado por c1 de generate muap)
# ==========================================================
def measure_theoretical_offset(muap, fs=10000):
    """Mide el offset desde el inicio hasta el pico máximo."""
    peak_idx = np.argmax(np.abs(muap))
    peak_value = muap[peak_idx]
    offset_samples = peak_idx
    offset_ms = (peak_idx / fs) * 1000
    
    return {
        'offset_samples': offset_samples,
        'offset_ms': offset_ms,
        'peak_idx': peak_idx,
        'peak_value': peak_value,
        'muap_length': len(muap)
    }


# ==========================================================
# Calcular length basado en fs
# ==========================================================
def get_muap_length(fs, duration_ms=4):
    """
    Calcula la longitud del MUAP en muestras.
    
    Parámetros:
    -----------
    fs : int
        Frecuencia de muestreo (Hz)
    duration_ms : float
        Duración del MUAP en milisegundos (default: 4ms para fs=10000 → 40 muestras)
    """
    return int(duration_ms * fs / 1000)


# ==========================================================
# Generate signal with 2 sources
# ==========================================================
def generate_synthetic_signal(config, duration_sec=150, 
                              amp1=40, amp2=30, 
                              h1_s1_std_div=12, h1_s2_std_div=8,
                              h2_s1_std_div=10, h2_s2_std_div=6,
                              noise_std=4, theta_list=None, 
                              biphasic=True):
    """
    Genera señal sintética.
    
    IMPORTANTE: Ahora el downsampling se hace ANTES de generar el MUAP,
    para evitar problemas con la longitud del waveform.
    
    Parámetros:
    -----------
    config : dict
        Debe contener 'sampling_rate' (original) y 'sampling_rate_DS' (target)
    """
    
    # --------------------------------------------------
    # Obtener frecuencias
    # --------------------------------------------------
    fs_original = config.get('sampling_rate', 10000)
    fs_target = config.get('sampling_rate_DS', fs_original)
    theta_list = [[config["t0"][0], config["beta"][0]],[config["t0"][1], config["beta"][1]]]
    t_R_list = config["t_R"]
    
    # Determinar si hay downsampling
    do_downsample = fs_target < fs_original
    
    if do_downsample:
        ratio = fs_original / fs_target
        fs_final = fs_target
        print(f"🔽 Downsampling: {fs_original}Hz → {fs_target}Hz (ratio={ratio:.1f})")
    else:
        ratio = 1
        fs_final = fs_original
        print(f"📡 Sin downsampling: {fs_final}Hz")
    
    # --------------------------------------------------
    # Ajustar parámetros que dependen de fs
    # --------------------------------------------------
    # Longitud del MUAP (4ms en la frecuencia final)
    ell_RI = get_muap_length(fs_final, duration_ms=4)
    
    # t_R en muestras (15ms en la frecuencia final)
    t_R_ms = 15  # 15ms de período refractario
    t_R_samples = int(t_R_ms * fs_final / 1000)
    
    # Valores por defecto si no se proporcionan
    if theta_list is None:
        # t0 en muestras (basado en fs_final)
        t0_1 = int(0.27 * fs_final)   # ~270ms en muestras
        t0_2 = int(1.2 * fs_final)    # ~1200ms en muestras
        theta_list = [
            [t0_1, 3.0],
            [t0_2, 2.0],
        ]
    
    if t_R_list is None:
        t_R_list = [t_R_samples, t_R_samples]
    
    # --------------------------------------------------
    # Generar MUAPs con la frecuencia FINAL
    # --------------------------------------------------
    n_MU = 2
    N = int(fs_final * duration_sec)
    
    print(f"\n📊 Generando MUAPs con fs={fs_final}Hz, length={ell_RI} muestras ({ell_RI*1000/fs_final:.1f}ms)")
    
    if biphasic:
        h1, offset1 = generate_biphasic_muap(ell_RI, s1_std_div=h1_s1_std_div, 
                                              s2_std_div=h1_s2_std_div, polarity=+1)
        h1 = amp1 * h1
        h2, offset2 = generate_biphasic_muap(ell_RI, s1_std_div=h2_s1_std_div, 
                                              s2_std_div=h2_s2_std_div, polarity=-1)
        h2 = amp2 * h2
    else:
        h1 = amp1 * generate_monophasic_muap(ell_RI, polarity=+1)
        h2 = amp2 * generate_monophasic_muap(ell_RI, polarity=-1)
        offset1 = np.argmax(np.abs(h1))
        offset2 = np.argmax(np.abs(h2))
    
    H_true = np.vstack([h1, h2])
    OFFSET_peaks = [offset1, offset2]
    
    # --------------------------------------------------
    # Medir offset teórico
    # --------------------------------------------------
    offset_info = {}
    for i, muap in enumerate([h1, h2]):
        info = measure_theoretical_offset(muap, fs_final)
        offset_info[f'MU{i}'] = info
        print(f"   MU{i}: pico en muestra {info['peak_idx']} (offset={info['offset_samples']} muestras)")
    
    # --------------------------------------------------
    # Simular fuentes
    # --------------------------------------------------
    print(f"\n🔄 Simulando {duration_sec}s de señal...")
    
    U1, Y1, isi1, spike_idx1 = simulate_one_source(
        N=N, H=h1, theta=theta_list[0], t_R=t_R_list[0]
    )
    
    U2, Y2, isi2, spike_idx2 = simulate_one_source(
        N=N, H=h2, theta=theta_list[1], t_R=t_R_list[1]
    )
    
    # --------------------------------------------------
    # Mezcla + ruido
    # --------------------------------------------------
    Y = Y1 + Y2 + np.random.normal(0, noise_std, N)
    
    U_true = np.vstack([U1, U2])
    isi_true = [isi1, isi2]
    spike_idx_true = [spike_idx1, spike_idx2]
    Theta_true = np.array(theta_list, dtype=float)
    
    # --------------------------------------------------
    # Actualizar configuración
    # --------------------------------------------------
    config['sampling_rate'] = fs_final
    config['t_R'] = t_R_list
    config['sampling_rate_DS'] = fs_final
    if 'init_data' in config:
        config['init_data'] = int(config['init_data'] / ratio) if ratio > 1 else config['init_data']
    
    print(f"\n✅ Señal generada: {len(Y)} muestras, {len(Y)/fs_final:.1f}s")
    print(f"   MU0: {len(spike_idx1)} spikes")
    print(f"   MU1: {len(spike_idx2)} spikes")
    
    return Y, U_true, H_true, Theta_true, isi_true, spike_idx_true, fs_final, ell_RI, offset_info, OFFSET_peaks