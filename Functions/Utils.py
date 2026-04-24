##Functions to format, read and remove spikes
import os
import numpy as np
import h5py
from scipy import signal
import yaml
import os
from datetime import datetime
import pstats



def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def create_experiment_folder(base_path):
    now = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    path = os.path.join(base_path, f"exp_{now}")
    os.makedirs(path, exist_ok=True)
    return path


def save_config(config, path):
    with open(os.path.join(path, "config.yaml"), "w") as f:
        yaml.dump(config, f)


def read_multichannel_bin_data(filepath, ch=None, quantum=0.0050863, skip_s=0, length_s=0, save=False):
    """
    Reads multichannel extracellular recordings from a .bin MEA file.

    Parameters
    ----------
    filepath : str
        Path to the .bin file
    quantum : float, optional
        Acquisition quantum -- scaling factor that converts the raw binary values 
        (signed 16-bit integers, >i2) into physical units (likely volts or microvolts).
        The file contains 2-byte signed integers representing the raw ADC counts.
    skip_s : float, optional
        Time (in seconds) to skip at the beginning of the file. Default is 0.
    length_s : float, optional
        Time (in seconds) of data to read. Leave 0 to read whole file. Default is 0.
    ch : int or None, optional
        If specified, load only this channel (0-indexed). Otherwise, load all channels.
    save : bool, optional
        If True, saves each channel as a separate .npy file. Default is False.

    Returns
    -------
    data_scaled : ndarray
        Array of signals converted to physical units (quantum * raw).
        - Shape: (length_chunks, N_channels) if ch is None
        - Shape: (length_chunks,) if ch is specified

    Notes
    -----
    File structure:
        At every sampling time (1/Fs), 64 samples (1 for every channel) are recorded
        in a 128B chunk (2B per channel):
            @0x0000 <Sample0/Ch0><Sample0/Ch1>...<Sample0/Ch63> (@t=0 s)
            @0x0080 <Sample1/Ch0><Sample1/Ch1>...<Sample1/Ch63> (@t=0.1 ms)
            ...
    """
    
    Fs = 10000  # Hz
    N_channels = 64  # Channels -- electrodes
    data_width_bytes = 2  # 2 bytes per sample (>i2: signed 2-byte integers, big-endian)
    filesize = os.path.getsize(filepath)
    
    # Number of chunks in requested data
    skip_chunks = int(skip_s * Fs)  # Number of 128B chunks to skip
    length_chunks = int(length_s * Fs)  # Number of 128B chunks to read
    
    # Handle cases where skip_s is unspecified(0) and/or length_s is unspecified(0)
    if (length_chunks == 0) and (skip_chunks == 0):
        # If no length or skip specified, read the whole file
        length_chunks = int(filesize / (N_channels * data_width_bytes))
        
    if (length_chunks == 0) and (skip_chunks > 0):
        # If no length specified but non-zero skip, read until end of file
        length_chunks = int(filesize / (N_channels * data_width_bytes) - skip_chunks)
    
    # Number of bytes in requested data
    skip_bytes = int(skip_chunks * N_channels * data_width_bytes)
    length_bytes = int(length_chunks * N_channels * data_width_bytes)
    
    print(f"Reading {length_bytes}B from offset {skip_bytes} (0x{skip_bytes:X})")
    
    # Read file
    with open(filepath, 'rb') as fid:
        if skip_bytes > 0:
            fid.seek(skip_bytes)
        data = np.fromfile(fid, dtype='>i2', count=length_bytes // data_width_bytes)
    
    print("20 first points of data (binary values): ")
    for x in data[:20]:
        print(x, end=',')
    print()
    
    # Reshape to (time_points, channels)
    data = data.reshape(length_chunks, N_channels)
    print("All done.")
    
    # Select channel if specified
    if ch is not None:
        if ch < 0 or ch >= N_channels:
            raise ValueError(f"Channel must be between 0 and {N_channels-1}. Got {ch}")
        data = data[:, ch]  # Shape: (length_chunks,)
    else:
        # Keep all channels, shape: (length_chunks, N_channels)
        pass
    
    # Save if requested
    if save and ch is None:
        dir = r"Data\Recordings_per_channel"
        os.makedirs(dir, exist_ok=True)  # Create directory if it doesn't exist
        for i in range(N_channels):
            np.save(file=(os.path.join(dir, f"channel_{i:02d}.npy")), arr=data[:, i])
    elif save and ch is not None:
        dir = r"Data\Recordings_per_channel"
        os.makedirs(dir, exist_ok=True)
        np.save(file=(os.path.join(dir, f"channel_{ch:02d}.npy")), arr=data)
    
    # Scale to physical units
    return data * quantum


def read_multichannel_h5_data(filepath, skip_s=0, length_s=0, ch=None, return_fs_labels=False):
    """
    Reads multichannel extracellular recordings from an MCS-converted HDF5 (.h5) MEA file.

    The function assumes the following HDF5 structure (as in
    'Haut glucose (16.7mM)_1min.h5'):

        Data
        └── Recording_0
            └── AnalogStream
                └── Stream_0
                    ├── ChannelData          (shape: n_channels × n_samples)
                    ├── ChannelDataTimeStamps (1 × 3)
                    └── InfoChannel          (structured array with channel metadata)

    Parameters
    ----------
    filepath : str
        Path to the .h5 file to read.
    skip_s : float, optional
        Number of seconds to skip from the start of the recording. Default is 0.
    length_s : float, optional
        Duration in seconds to load. If 0 or None, loads until the end. Default is 0.
    ch : int or None, optional
        If specified, load only this channel (0-indexed). Otherwise, load all channels.
    return_fs_labels : bool, optional
        If True, also return sampling frequency (Hz) and channel labels.

    Returns
    -------
    signals_uv : ndarray
        Array of signals converted to **microvolts**.
        - Shape: (n_channels, n_samples) if ch is None
        - Shape: (n_samples,) if ch is specified
    fs : int, optional
        Sampling frequency in Hz (returned if return_fs_labels=True)
    labels : list of str, optional
        List of channel labels (returned if return_fs_labels=True)

    Notes
    -----
    - Conversion from raw ADC counts to Volts:
        signals_volt = (signals_raw - ADZero) * ConversionFactor * 10^Exponent
      Then multiplied by 1e6 to convert to µV.
    - ADZero, ConversionFactor, and Exponent are per-channel and obtained from InfoChannel.
    - Sampling frequency (Hz) is derived from the 'Tick' attribute (usually in µs):
        fs = 1e6 / Tick
    - Supports partial loading using `skip_s` and `length_s`.
    
    Example
    -------
    >>> signals = read_multichannel_h5_data("Haut glucose.h5", skip_s=10, length_s=20)
    >>> ch5_signal = read_multichannel_h5_data("Haut glucose.h5", ch=5)
    >>> signals, fs, labels = read_multichannel_h5_data("Haut glucose.h5", return_fs_labels=True)
    """

    # Load data and metadata
    with h5py.File(filepath, "r") as f:
        data = f["Data/Recording_0/AnalogStream/Stream_0/ChannelData"]
        info = f["Data/Recording_0/AnalogStream/Stream_0/InfoChannel"]

        signals_raw = data[:]
        info_np = info[:]

    # Get per-channel ADC conversion parameters
    ADZero = info_np["ADZero"][:, None]            # ADC offset
    factor = info_np["ConversionFactor"][:, None]  # Gain factor
    exponent = info_np["Exponent"][:, None]        # Scaling exponent

    # Convert raw counts to Volts
    signals_volt = (signals_raw - ADZero) * factor * (10.0 ** exponent)

    # Sampling frequency
    tick_us = info_np[0]["Tick"]                   # in microseconds
    fs = int(1e6 / tick_us)                        # Hz

    # Handle optional skip and length
    n_samples = signals_volt.shape[1]
    start_idx = int(skip_s * fs) if skip_s > 0 else 0
    if length_s > 0:
        end_idx = start_idx + int(length_s * fs)
        end_idx = min(end_idx, n_samples)  # don't exceed total samples
    else:
        end_idx = n_samples  # load until the end

    # Select channels if specified
    if ch is not None:
        signals_uv = signals_volt[ch, start_idx:end_idx] * 1e6  # convert to µV
    else:
        signals_uv = signals_volt[:, start_idx:end_idx] * 1e6

    # Optional: return labels and sampling frequency
    if return_fs_labels:
        labels = [x.decode() if isinstance(x, bytes) else str(x) for x in info_np["Label"]]
        return signals_uv, fs, labels

    return signals_uv
     

def read_one_channel(filepath, quantum=0.0050863, skip_s=0, length_s=0, Fs = 10000): 


    # Only to use for .npy files that have already when saved using the read_multichannel_bin_data by channel
    
    start_idx = Fs*skip_s
    End_idx = Fs*length_s

    if start_idx and length_s == 0: 
        data = np.load(filepath)
    else:
        data = np.load(filepath)[start_idx:start_idx + End_idx]
    
    
    return data * quantum


def preprocess_signal(signal_raw, config):
    nyq = 0.5 * config["sampling_rate"]

    # High-pass
    b_high, a_high = signal.butter(4, config["highpass"]/nyq, btype='high')
    sig = signal.filtfilt(b_high, a_high, signal_raw)

    # Notch
    b_notch, a_notch = signal.iirnotch(config["notch"]/nyq, 30)
    sig = signal.filtfilt(b_notch, a_notch, sig)

    return sig


def preprocess_with_downsampling(signal, fs_original, config):
    """
    Downsample la señal y ajusta parámetros.
    """
    from scipy.signal import decimate
    
    # Calcular factor de downsampling
    decimation_factor = int(fs_original / config["sampling_rate_DS"])
    
    if decimation_factor == 1:
        signal_downsampled = signal
    else :
        # Downsampling de la señal
        signal_downsampled = decimate(signal, decimation_factor, ftype='fir')
    
    # Nuevos parámetros
    fs_new = fs_original / decimation_factor
    
    # Ajustar t_R y ell_RI
    t_R_new = [t_R // decimation_factor for t_R in config["t_R"]] 
    
    
    return signal_downsampled, fs_new, t_R_new, decimation_factor

# Fonction pour enlever les spikes trop proches (< min_distance samples) 10ms utilisé par Paul
def remove_close_spikes(indices, min_distance): # Dado el conocimiento del periodo refractario, 
                                                # se pueden eliminar picos muy cercanos que no son realmente picos
    if len(indices) < 2:
        return indices
    
    indices = np.sort(indices)
    to_keep = [True] * len(indices)
    
    i = 0
    while i < len(indices):
        j = i + 1
        while j < len(indices) and indices[j] - indices[i] < min_distance:
            to_keep[j] = False
            j += 1
        i = j
    
    return indices[to_keep]


def save_online_results(online_results, path):


    # 🔹 Arrays grandes
    np.save(os.path.join(path, "U_est.npy"), online_results["U_est"])
    np.save(os.path.join(path, "Y_est.npy"), online_results["Y_est"])
    np.save(os.path.join(path, "H_est.npy"), online_results["H_est"])
    np.save(os.path.join(path, "Theta_est"), online_results["Theta_est"])

    """ # 🔹 Parámetros (Theta)
    theta = online_results["Theta_est"]

    # Convertir a lista si es numpy
    theta_serializable = (
        theta.tolist() if isinstance(theta, np.ndarray) else theta
    )

    with open(os.path.join(path, "Theta_est.json"), "w") as f:
        json.dump(theta_serializable, f, indent=4) """
#Profiling paara evaluar velocidad del algoritmo

def generate_profile_report(profiler, exp_path, top_n=15):
    """
    Genera un reporte legible del profiling.
    
    Parámetros:
    - profiler: objeto cProfile.Profile
    - exp_path: ruta para guardar el reporte
    - top_n: número de funciones a mostrar
    """
    
    # Guardar stats
    profiler.dump_stats(os.path.join(exp_path, 'online_profile.prof'))
    
    # Cargar stats
    stats = pstats.Stats(os.path.join(exp_path, 'online_profile.prof'))
    
    # Obtener tiempo total
    total_time = stats.total_tt
    
    # ==================================================
    # 🔹 REPORTE POR FUNCIÓN (tiempo acumulado)
    # ==================================================
    
    print("\n" + "="*80)
    print(f"📊 PROFILING REPORT - {total_time:.2f} seconds total")
    print("="*80)
    
    # Extraer información de stats
    stats_data = []
    for func, (cc, nc, tt, ct, callers) in stats.stats.items():
        file_name = func[0].split('/')[-1]  # Solo nombre del archivo
        func_name = func[2]
        line_no = func[1]
        
        # Filtrar funciones internas de Python/NumPy si se desea
        if 'site-packages' in func[0] and 'numpy' not in func[0]:
            continue
            
        stats_data.append({
            'name': f"{func_name} ({file_name}:{line_no})",
            'ncalls': nc,
            'tottime': tt,
            'tottime_pct': 100 * tt / total_time if total_time > 0 else 0,
            'cumtime': ct,
            'cumtime_pct': 100 * ct / total_time if total_time > 0 else 0
        })
    
    # Ordenar por tiempo acumulado
    stats_data.sort(key=lambda x: x['cumtime'], reverse=True)
    
    # Mostrar top N
    print(f"\n🔝 TOP {top_n} FUNCTIONS (by cumulative time):")
    print("-"*80)
    print(f"{'Function':<50} {'Calls':>8} {'Tottime':>10} {'%':>6} {'Cumtime':>10} {'%':>6}")
    print("-"*80)
    
    for i, func in enumerate(stats_data[:top_n]):
        print(f"{func['name'][:50]:<50} "
              f"{func['ncalls']:>8} "
              f"{func['tottime']:>8.2f}s "
              f"{func['tottime_pct']:>5.1f}% "
              f"{func['cumtime']:>8.2f}s "
              f"{func['cumtime_pct']:>5.1f}%")
    
    # ==================================================
    # 🔹 RESUMEN POR TIPO DE OPERACIÓN
    # ==================================================
    
    print("\n" + "="*80)
    print("📈 SUMMARY BY OPERATION TYPE:")
    print("-"*80)
    
    categories = {
        'update_theta': {'pattern': 'update_theta', 'time': 0, 'calls': 0},
        'kalman_update': {'pattern': 'kalman_update', 'time': 0, 'calls': 0},
        'gradient_Q': {'pattern': 'gradient_Q', 'time': 0, 'calls': 0},
        'weibull_r': {'pattern': 'weibull', 'time': 0, 'calls': 0},
        'numpy_outer': {'pattern': 'outer', 'time': 0, 'calls': 0},
        'numpy_inv': {'pattern': 'inv', 'time': 0, 'calls': 0},
        'tqdm/threading': {'pattern': 'threading|tqdm', 'time': 0, 'calls': 0},
        'other': {'pattern': None, 'time': 0, 'calls': 0}
    }
    
    for func in stats_data:
        assigned = False
        for cat, info in categories.items():
            if info['pattern'] and info['pattern'] in func['name'].lower():
                categories[cat]['time'] += func['cumtime']
                categories[cat]['calls'] += func['ncalls']
                assigned = True
                break
        if not assigned and func['name'] != 'other':
            categories['other']['time'] += func['cumtime']
            categories['other']['calls'] += func['ncalls']
    
    for cat, info in categories.items():
        if info['time'] > 0:
            pct = 100 * info['time'] / total_time
            print(f"  {cat:<20}: {info['time']:>8.2f}s ({pct:>5.1f}%) - {info['calls']:>10} calls")
    
    # ==================================================
    # 🔹 GUARDAR REPORTE COMPLETO
    # ==================================================
    
    report_path = os.path.join(exp_path, 'profile_report.txt')
    with open(report_path, 'w') as f:
        f.write(f"Profiling Report - {total_time:.2f} seconds\n")
        f.write("="*80 + "\n\n")
        
        # Guardar top N
        f.write(f"TOP {top_n} FUNCTIONS (by cumulative time):\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Function':<50} {'Calls':>8} {'Tottime':>10} {'%':>6} {'Cumtime':>10} {'%':>6}\n")
        f.write("-"*80 + "\n")
        
        for func in stats_data[:top_n]:
            f.write(f"{func['name'][:50]:<50} "
                    f"{func['ncalls']:>8} "
                    f"{func['tottime']:>8.2f}s "
                    f"{func['tottime_pct']:>5.1f}% "
                    f"{func['cumtime']:>8.2f}s "
                    f"{func['cumtime_pct']:>5.1f}%\n")
        
        # Guardar resumen por categoría
        f.write("\n" + "="*80 + "\n")
        f.write("SUMMARY BY OPERATION TYPE:\n")
        f.write("-"*80 + "\n")
        for cat, info in categories.items():
            if info['time'] > 0:
                pct = 100 * info['time'] / total_time
                f.write(f"  {cat:<20}: {info['time']:>8.2f}s ({pct:>5.1f}%) - {info['calls']:>10} calls\n")
    
    print(f"\n📁 Full report saved to: {report_path}")
    
    # ==================================================
    # 🔹 RECOMENDACIONES
    # ==================================================
    
    print("\n" + "="*80)
    print("💡 RECOMMENDATIONS:")
    print("-"*80)
    
    # Identificar bottlenecks
    for cat, info in categories.items():
        pct = 100 * info['time'] / total_time if total_time > 0 else 0
        if cat == 'tqdm/threading' and pct > 10:
            print(f"  ⚠️ tqdm/threading takes {pct:.1f}% of time → increase mininterval or disable")
        if cat == 'numpy_inv' and pct > 5:
            print(f"  ⚠️ Matrix inversion takes {pct:.1f}% of time → use analytic 2x2 inverse")
        if cat == 'numpy_outer' and pct > 10:
            print(f"  ⚠️ np.outer takes {pct:.1f}% of time → use manual calculation")
        if cat == 'kalman_update' and pct > 20:
            print(f"  ⚠️ Kalman update takes {pct:.1f}% of time → consider EnKF")
        if cat == 'update_theta' and pct > 20:
            print(f"  ⚠️ Theta update takes {pct:.1f}% of time → check gradient calculation")
    
    print("\n✅ Profiling complete!")
    
    return stats_data, categories


