##Metrics

import numpy as np
from scipy.spatial import cKDTree
from statistics import multimode

def weibull_discrete_pmf(t, t0, beta, t_R):
    return np.where(
        t > t_R,
        np.exp(-((t - 1 - t_R) / (t0 - t_R))**beta)
        - np.exp(-((t - t_R) / (t0 - t_R))**beta),
        0
    )

def extract_isi_per_mu(U_est):
    isi_per_mu = []
    for mu in range(U_est.shape[0]):
        spike_times = np.where(U_est[mu] == 1)[0]
        if len(spike_times) > 1:
            isi = np.diff(spike_times)
        else:
            isi = np.array([])
        isi_per_mu.append(isi)
    return isi_per_mu

def compute_cpa_score(Y, U_est, H_est):
    """
    Devuelve:
    - corr (shape similarity)
    - rmse (error real)
    - Y_pred (reconstrucción)
    """

    n_MU, N = U_est.shape

    # 🔹 reconstrucción
    Y_pred = np.zeros(N)
    for i in range(n_MU):
        recon = np.convolve(U_est[i], H_est[i], mode='full')[:N]
        Y_pred += recon

    # 🔹 correlación
    Yc = Y - np.mean(Y)
    Ypc = Y_pred - np.mean(Y_pred)

    denom = np.linalg.norm(Yc) * np.linalg.norm(Ypc)
    corr = np.dot(Yc, Ypc) / denom if denom != 0 else 0.0

    # 🔥 🔹 NUEVO: RMSE (CRÍTICO)
    rmse = np.sqrt(np.mean((Y - Y_pred)**2))

    return corr, rmse, Y_pred

def evaluate_spike_detection(spikes_abs, gt_peaks, tolerance=3, verbose=True):
    """
    Evalúa la detección de spikes.
    
    Parámetros:
    -----------
    spikes_abs : list or array
        Spikes detectados (coordenadas absolutas)
    gt_peaks : list or array
        Ground truth (coordenadas absolutas, ya con offset aplicado)
    tolerance : int
        Tolerancia en muestras para considerar un acierto
    verbose : bool
        Si imprimir resultados
    
    Retorna:
    --------
    dict con métricas
    """
    
    #Convertir a numpy arrays si es necesario
    spikes_abs = np.asarray(spikes_abs)
    gt_peaks = np.asarray(gt_peaks)
    
    if len(gt_peaks) == 0:
        tp, fp, fn = 0, len(spikes_abs), 0
        errors = []
    elif len(spikes_abs) == 0:
        tp, fp, fn = 0, 0, len(gt_peaks)
        errors = []
    else:
        #reshape apra el cKDTree
        tree = cKDTree(gt_peaks.reshape(-1, 1))
        distances, indices = tree.query(spikes_abs.reshape(-1, 1), k=1)
        
        used_gt = set()
        tp = 0
        errors = []
        
        for i_spike, (dist, idx_gt) in enumerate(zip(distances, indices)):
            if dist <= tolerance and idx_gt not in used_gt:
                tp += 1
                used_gt.add(idx_gt)
                errors.append(spikes_abs[i_spike] - gt_peaks[idx_gt])
        
        fp = len(spikes_abs) - tp
        fn = len(gt_peaks) - tp
    
    # Métricas
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    results = {
        'TP': tp,
        'FP': fp,
        'FN': fn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'errors': np.array(errors),
        'mean_error': np.mean(errors) if errors else 0,
        'std_error': np.std(errors) if errors else 0,
    }
    
    # Calcular moda (requiere lista de enteros)
    if errors:
        errors_int = [int(round(e)) for e in errors]
        results['mode_error'] = multimode(errors_int) if errors_int else []
    else:
        results['mode_error'] = []
    
    if verbose:
        print(f"\n  📊 Spike Detection Metrics:")
        print(f"     TP: {tp} | FP: {fp} | FN: {fn}")
        print(f"     Precision: {precision:.3f}")
        print(f"     Recall: {recall:.3f}")
        print(f"     F1-score: {f1:.3f}")
        if errors:
            print(f"     Mean error: {results['mean_error']:.2f} samples")
            print(f"     Std error: {results['std_error']:.2f}")
            print(f"     Mode error: {results['mode_error']}")
    
    return results
