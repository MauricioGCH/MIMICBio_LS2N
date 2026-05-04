## Gradient_descent_weibull_params
import numpy as np
import Functions.Metrics as Metrics
from scipy.optimize import minimize
from scipy.special import gamma

def reorder_tR_by_activity(isi_per_mu, t_R_config):
    """
    Reordena t_R para que el valor más pequeño se asigne a la MU más activa (más ISI).
    
    Parámetros:
    - isi_per_mu: lista de arrays con ISIs por MU
    - t_R_config: lista de t_R (2 valores) o escalar
    
    Retorna:
    - t_R_por_mu: lista reordenada de t_R
    - indices_ordenados: índices de MUs ordenados por actividad (mayor a menor)
    - n_isi_por_mu: número de ISI por MU
    """
    n_isi_por_mu = [len(isi) for isi in isi_per_mu]
    indices_ordenados = np.argsort(n_isi_por_mu)[::-1]  # De mayor a menor actividad
    
    if isinstance(t_R_config, (list, np.ndarray)) and len(t_R_config) == 2:
        t_R_ordenados = sorted(t_R_config)  # [t_R_pequeño, t_R_grande]
        t_R_por_mu = [0, 0]
        # Más activo (primer índice) → t_R pequeño
        t_R_por_mu[indices_ordenados[0]] = t_R_ordenados[0]
        # Menos activo (segundo índice) → t_R grande
        t_R_por_mu[indices_ordenados[1]] = t_R_ordenados[1]
    else:
        # Si es escalar o no es lista de 2, mantener igual
        t_R_por_mu = t_R_config if isinstance(t_R_config, list) else [t_R_config, t_R_config]
    
    print("\n📊 Asignación de t_R según actividad (más activo → t_R más pequeño):")
    for i in range(len(isi_per_mu)):
        actividad = "MÁS activo" if i == indices_ordenados[0] else "MENOS activo"
        print(f"   MU{i}: {n_isi_por_mu[i]} ISIs → t_R = {t_R_por_mu[i]} muestras ({actividad})")
    
    return t_R_por_mu, indices_ordenados, n_isi_por_mu


def estimate_weibull_LBFGS(isi_per_mu, fs, config):
    """
    Estima t0 y beta de la distribución Weibull discreta con periodo refractario
    usando gradient descent (minimización de NLL).
    
    Parámetros:
    - isi_per_mu: lista de arrays con ISIs por MU (en segundos)
    - fs: sampling rate (Hz)
    - config: dict con t_R (ms o muestras)
    
    Retorna:
    - t0_init: array con t0 estimado por MU (en muestras)
    - beta_init: array con beta estimado por MU
    """

    t_R_config = config["t_R"]
    
    # ==================================================
    # 🔹 REORDENAR t_R SEGÚN ACTIVIDAD
    # ==================================================
    t_R_por_mu, indices_ordenados, n_isi_por_mu = reorder_tR_by_activity(isi_per_mu, t_R_config)

    t0_init = []
    beta_init = []

    for mu_idx, isi in enumerate(isi_per_mu):

        # 🔹 t_R para esta MU específica (ya reordenado)
        if np.isscalar(t_R_por_mu):
            t_R = t_R_por_mu
        else:
            t_R = t_R_por_mu[mu_idx]
        
        if len(isi) < 2:
            raise ValueError(
                f"MU {mu_idx+1}: ISIs insuficientes para estimar Weibull ({len(isi)} ISIs). "
                f"Revisa la detección de spikes o el umbral."
            )

        # 🔹 ISIs segundos → muestras discretas
        t = np.round(isi * fs).astype(int)

        # ==================================================
        # 🔹 NEGATIVE LOG-LIKELIHOOD
        # ==================================================
        def nll(params):
            t0, beta = params

            if t0 <= t_R + 1e-6 or beta <= 0:
                return 1e10

            pmf = Metrics.weibull_discrete_pmf(t, t0, beta, t_R)
            pmf = np.clip(pmf, 1e-12, None)

            return -np.sum(np.log(pmf))

        # ==================================================
        # 🔹 INICIALIZACIÓN
        # ==================================================
        t0_guess = np.mean(t)
        beta_guess = 2.0

        # ==================================================
        # 🔹 GRADIENT DESCENT (L-BFGS-B)
        # ==================================================
        result = minimize(
            nll,
            x0=[t0_guess, beta_guess],
            method='L-BFGS-B',
            bounds=[
                (t_R + 1e-3, None),  # t0 > t_R
                (1e-3, None)         # beta > 0
            ],
            options={
                'maxiter': 1000,
                'ftol': 1e-12,
                'gtol': 1e-8
            }
        )

        t0_est, beta_est = result.x

        print("---GRADIENT DESCENT - WEIBULL THETA INITIALISATION---")
        if result.success:
            print(f"✅ MU {mu_idx+1}: t0={t0_est:.2f} muestras, beta={beta_est:.3f} | NLL={result.fun:.4f} | ISIs={len(t)}")
            print(f"   t_R asignado: {t_R} muestras")
        else:
            print(f"⚠️ MU {mu_idx+1}: convergencia no alcanzada — {result.message}")
            print(f"   Usando mejor resultado: t0={t0_est:.2f}, beta={beta_est:.3f}")
        
        t0_init.append(t0_est)
        beta_init.append(beta_est)

    return np.array(t0_init), np.array(beta_init)


def estimate_weibull_bayesian(isi_per_mu, fs, config):
    """
    Estima t0 y beta usando inferencia Bayesiana con priors informativos.
    Recomendado para pocos eventos (< 15 ISI por MU).
    
    Parámetros:
    - isi_per_mu: lista de arrays con ISIs por MU (en segundos)
    - fs: sampling rate (Hz)
    - config: dict con t_R (muestras o lista)
    
    Retorna:
    - t0_init: array con t0 estimado por MU (en muestras)
    - beta_init: array con beta estimado por MU
    """

    t_R_config = config["t_R"]
    
    # ==================================================
    # 🔹 REORDENAR t_R SEGÚN ACTIVIDAD
    # ==================================================
    t_R_por_mu, indices_ordenados, n_isi_por_mu = reorder_tR_by_activity(isi_per_mu, t_R_config)

    t0_init = []
    beta_init = []

    for mu_idx, isi in enumerate(isi_per_mu):
        # t_R para esta MU (ya reordenado)
        if np.isscalar(t_R_por_mu):
            t_R = t_R_por_mu
        else:
            t_R = t_R_por_mu[mu_idx]
        
        if len(isi) < 2:
            print(f"⚠️ MU {mu_idx+1}: solo {len(isi)} ISIs, usando valores por defecto")
            t0_init.append(1.2 * fs)
            beta_init.append(2.0)
            continue

        # ISI segundos → muestras
        t = np.round(isi * fs).astype(int)
        
        # ===== NEGATIVE LOG-POSTERIOR (likelihood + prior) =====
        def nlp(params):
            t0, beta = params
            if t0 <= t_R + 1e-6 or beta <= 0:
                return 1e10
            
            # Likelihood
            pmf = Metrics.weibull_discrete_pmf(t, t0, beta, t_R)
            pmf = np.clip(pmf, 1e-12, None)
            nll = -np.sum(np.log(pmf))
            
            # Priors
            mean_t = np.mean(t)
            std_t = max(np.std(t), 1.0)
            prior_t0 = 0.5 * ((t0 - mean_t) / std_t) ** 2
            prior_beta = 0.5 * ((beta - 2.0) / 1.0) ** 2
            
            return nll + prior_t0 + prior_beta
        
        # Inicialización
        t0_guess = np.mean(t)
        beta_guess = 2.0
        
        result = minimize(
            nlp,
            x0=[t0_guess, beta_guess],
            method='L-BFGS-B',
            bounds=[(t_R + 1e-3, None), (1e-3, None)],
            options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-6}
        )
        
        t0_est, beta_est = result.x
        
        print("---BAYESIAN ESTIMATION---")
        if result.success:
            print(f"✅ MU {mu_idx+1}: t0={t0_est:.2f} samples, beta={beta_est:.3f} | NLL={result.fun:.4f} | ISIs={len(t)}")
            print(f"   t_R asignado: {t_R} muestras")
        else:
            print(f"⚠️ MU {mu_idx+1}: {result.message} | t0={t0_est:.2f}, beta={beta_est:.3f}")
        
        t0_init.append(t0_est)
        beta_init.append(beta_est)

    return np.array(t0_init), np.array(beta_init)


def estimate_weibull_moments(isi_per_mu, fs, config):
    """
    Estima t0 y beta usando el método de momentos con tabla completa
    de Rinne (2008) e interpolación lineal.
    """
    from scipy.special import gamma
    from scipy.interpolate import interp1d

    # ==================================================
    # 🔹 TABLA COMPLETA DE RINNE (2008) Table 12/3
    # c (beta) → CV = σ/μ
    # Cubre c ∈ [0.1, 11.0]
    # ==================================================
    RINNE_TABLE = np.array([
        (0.1, 429.8),  (0.2, 15.84),  (0.3, 5.408),  (0.4, 3.141),
        (0.5, 2.236),  (0.6, 1.758),  (0.7, 1.462),  (0.8, 1.261),
        (0.9, 1.113),  (1.0, 1.000),  (1.2, 0.8369), (1.4, 0.7238),
        (1.6, 0.6399), (1.8, 0.5749), (2.0, 0.5227), (2.2, 0.4798),
        (2.4, 0.4438), (2.6, 0.4131), (2.8, 0.3866), (3.0, 0.3634),
        (3.2, 0.3430), (3.4, 0.3248), (3.6, 0.3085), (3.8, 0.2939),
        (4.0, 0.2805), (4.2, 0.2684), (4.4, 0.2573), (4.6, 0.2471),
        (4.8, 0.2377), (5.0, 0.2291), (5.2, 0.2210), (5.4, 0.2135),
        (5.6, 0.2065), (5.8, 0.1999), (6.0, 0.1938), (6.2, 0.1880),
        (6.4, 0.1826), (6.6, 0.1774), (6.8, 0.1726), (7.0, 0.1680),
        (7.2, 0.1637), (7.4, 0.1596), (7.6, 0.1556), (7.8, 0.1519),
        (8.0, 0.1484), (8.2, 0.1450), (8.4, 0.1417), (8.6, 0.1387),
        (8.8, 0.1357), (9.0, 0.1329), (9.2, 0.1301), (9.4, 0.1275),
        (9.6, 0.1250), (9.8, 0.1226), (10.0, 0.1203),(10.2, 0.1181),
        (10.4, 0.1159),(10.6, 0.1139),(10.8, 0.1119),(11.0, 0.1099),
    ])

    c_values  = RINNE_TABLE[:, 0]   # beta
    cv_values = RINNE_TABLE[:, 1]   # CV correspondiente

    # Interpolador CV → beta (CV es decreciente, invertimos para interp1d)
    # fill_value="extrapolate" maneja CVs fuera del rango de la tabla
    cv_to_beta = interp1d(
        cv_values[::-1],   # CV creciente (requerido por interp1d)
        c_values[::-1],    # beta correspondiente
        kind='linear',
        bounds_error=False,
        fill_value=(c_values[-1], c_values[0])  # clamp a [0.1, 11.0]
    )

    # ==================================================
    # 🔹 REORDENAR t_R SEGÚN ACTIVIDAD
    # ==================================================
    t_R_config = config["t_R"]
    if not len(isi_per_mu) == 1:
        t_R_por_mu, indices_ordenados, n_isi_por_mu = reorder_tR_by_activity(isi_per_mu, t_R_config)
    else:
        t_R_por_mu = t_R_config

    t0_init  = []
    beta_init = []

    for mu_idx, isi in enumerate(isi_per_mu):

        t_R = t_R_por_mu if np.isscalar(t_R_por_mu) else t_R_por_mu[mu_idx]

        if len(isi) < 2:
            print(f"⚠️ MU {mu_idx+1}: solo {len(isi)} ISIs, usando valores por defecto")
            t0_init.append(1.2 * fs)
            beta_init.append(2.0)
            continue

        # ISI segundos → muestras y quitar t_R
        t         = np.round(isi * fs).astype(int)
        t_shifted = t - t_R
        t_shifted = t_shifted[t_shifted > 0]

        if len(t_shifted) < 2:
            print(f"⚠️ MU {mu_idx+1}: insuficientes ISIs tras t_R, valores por defecto")
            t0_init.append(1.2 * fs)
            beta_init.append(2.0)
            continue

        # ==================================================
        # 🔹 MOMENTOS EMPÍRICOS
        # ==================================================
        mean_t = np.mean(t_shifted)
        std_t  = np.std(t_shifted)
        cv     = std_t / mean_t if mean_t > 0 else 1.0

        # ==================================================
        # 🔹 BETA — interpolación en tabla de Rinne
        # ==================================================
        beta_est = float(cv_to_beta(cv))

        # Clamp de seguridad — beta fuera de rango fisiológico
        beta_est = np.clip(beta_est, 0.5, 11.0)

        # ==================================================
        # 🔹 T0 — a partir de la media de la Weibull desplazada
        # E[t_shifted] = t0_shifted × Γ(1 + 1/β)
        # ==================================================
        t0_shifted = mean_t / gamma(1 + 1 / beta_est)
        t0_est     = t0_shifted + t_R

        print("---METHOD OF MOMENTS (Rinne 2008)---")
        print(f"✅ MU {mu_idx+1}: t0={t0_est:.2f} samples, beta={beta_est:.4f} | "
              f"CV={cv:.4f} | ISIs={len(t_shifted)} | t_R={t_R}")

        t0_init.append(t0_est)
        beta_init.append(beta_est)

    return np.array(t0_init), np.array(beta_init)


def estimate_weibull_grid_search(isi_per_mu, fs, config):
    """
    Estima t0 y beta usando búsqueda en grilla + fine-tuning con L-BFGS-B.
    Robusto para pocos datos, encuentra mínimo global.
    
    Parámetros:
    - isi_per_mu: lista de arrays con ISIs por MU (en segundos)
    - fs: sampling rate (Hz)
    - config: dict con t_R (muestras o lista)
    
    Retorna:
    - t0_init: array con t0 estimado por MU (en muestras)
    - beta_init: array con beta estimado por MU
    """

    t_R_config = config["t_R"]
    
    # ==================================================
    # 🔹 REORDENAR t_R SEGÚN ACTIVIDAD
    # ==================================================
    t_R_por_mu, indices_ordenados, n_isi_por_mu = reorder_tR_by_activity(isi_per_mu, t_R_config)

    t0_init = []
    beta_init = []

    for mu_idx, isi in enumerate(isi_per_mu):
        # t_R para esta MU (ya reordenado)
        if np.isscalar(t_R_por_mu):
            t_R = t_R_por_mu
        else:
            t_R = t_R_por_mu[mu_idx]
        
        if len(isi) < 2:
            print(f"⚠️ MU {mu_idx+1}: solo {len(isi)} ISIs, usando valores por defecto")
            t0_init.append(1.2 * fs)
            beta_init.append(2.0)
            continue

        # ISI segundos → muestras
        t = np.round(isi * fs).astype(int)
        
        # ===== NEGATIVE LOG-LIKELIHOOD =====
        def nll(params):
            t0, beta = params
            if t0 <= t_R + 1e-6 or beta <= 0:
                return 1e10
            pmf = Metrics.weibull_discrete_pmf(t, t0, beta, t_R)
            pmf = np.clip(pmf, 1e-12, None)
            return -np.sum(np.log(pmf))
        
        # ===== GRID SEARCH =====
        best_nll = np.inf
        best_params = None
        
        mean_t = np.mean(t)
        
        # t0 candidates: autour de la moyenne
        t0_min = max(t_R + 10, mean_t * 0.5)
        t0_max = mean_t * 2.0
        t0_candidates = np.linspace(t0_min, t0_max, 15)
        
        # beta candidates
        beta_candidates = np.linspace(1.2, 4.0, 12)
        
        for t0_test in t0_candidates:
            for beta_test in beta_candidates:
                pmf = Metrics.weibull_discrete_pmf(t, t0_test, beta_test, t_R)
                pmf = np.clip(pmf, 1e-12, None)
                nll_val = -np.sum(np.log(pmf))
                
                if nll_val < best_nll:
                    best_nll = nll_val
                    best_params = (t0_test, beta_test)
        
        print(f"  Grid search: meilleur t0={best_params[0]:.1f}, beta={best_params[1]:.3f}, NLL={best_nll:.2f}")
        
        # ===== FINE-TUNING avec L-BFGS-B =====
        result = minimize(
            nll,
            x0=best_params,
            method='L-BFGS-B',
            bounds=[(t_R + 1e-3, None), (1e-3, None)],
            options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6}
        )
        
        t0_est, beta_est = result.x
        
        print("---GRID SEARCH + FINE-TUNING---")
        if result.success:
            print(f"✅ MU {mu_idx+1}: t0={t0_est:.2f} samples, beta={beta_est:.3f} | NLL={result.fun:.4f} | ISIs={len(t)}")
            print(f"   t_R asignado: {t_R} muestras")
        else:
            print(f"⚠️ MU {mu_idx+1}: fine-tuning non convergé, utilisation grille | t0={best_params[0]:.2f}, beta={best_params[1]:.3f}")
            t0_est, beta_est = best_params
        
        t0_init.append(t0_est)
        beta_init.append(beta_est)

    return np.array(t0_init), np.array(beta_init)