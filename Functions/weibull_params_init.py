## Gradient_descent_weibull_params
import numpy as np
import Functions.Metrics as Metrics
from scipy.optimize import minimize
from scipy.special import gamma

def estimate_weibull_LBFGS(isi_per_mu, fs, config):
    """
    Estima t0 y beta de la distribución Weibull discreta con periodo refractario
    usando gradient descent (minimización de NLL).
    
    Parámetros:
    - isi_per_mu: lista de arrays con ISIs por MU (en segundos)
    - fs: sampling rate (Hz)
    - config: dict con t_R (ms)
    
    Retorna:
    - t0_init: array con t0 estimado por MU (en muestras)
    - beta_init: array con beta estimado por MU
    """

    t_R_config = config["t_R"]  # Puede ser escalar o lista [t_R_MU1, t_R_MU2, ...] # ms → muestras config["t_R"] * fs / 1000 

    t0_init = []
    beta_init = []

    for mu_idx, isi in enumerate(isi_per_mu):

        # 🔹 t_R para esta MU específica
        if np.isscalar(t_R_config):
            t_R = t_R_config
        else:
            t_R = t_R_config[mu_idx]
        
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
        t0_guess = np.mean(t)   # Media de ISIs: estimación natural de la escala
        beta_guess = 2.0        # Forma típica para ISIs fisiológicos

        # ==================================================
        # 🔹 GRADIENT DESCENT (L-BFGS-B)
        # ==================================================
        result = minimize(
            nll,
            x0=[t0_guess, beta_guess],
            method='L-BFGS-B',
            bounds=[
                (t_R + 1e-3, None),  # t0 > t_R
                (1e-3, None)          # beta > 0
            ],
            options={
                'maxiter': 1000,
                'ftol': 1e-12,
                'gtol': 1e-8
            }
        )

        t0_est, beta_est = result.x
        

        # if t0_est > int(1.2*fs):
        #         print(" t_0 muy grande, usando inicializacion de Paul")
        #         print(f"✅ MU {mu_idx+1}: t0={int(1.2*fs):.2f} muestras, beta={2:.3f}")
        #         t0_init.append(int(1.2*fs))
        #         beta_init.append(2)
        #     else:
        #         t0_init.append(t0_est)
        #         beta_init.append(beta_est)

        print("---GRADIENT DESCENT - WEIBULL THETA INITIALISATION---")
        if result.success:
            print(f"✅ MU {mu_idx+1}: t0={t0_est:.2f} muestras, beta={beta_est:.3f} | NLL={result.fun:.4f} | ISIs={len(t)}")

            
            
            t0_init.append(t0_est)
            beta_init.append(beta_est)
        else:
            print(f"⚠️ MU {mu_idx+1}: convergencia no alcanzada — {result.message}")
            #print(f"   Usando mejor resultado: t0={t0_est:.2f}, beta={beta_est:.3f}")
            #print("Usando inicializacion de Paul")
            #t0_init.append(1.2*fs)
            #beta_init.append(2)

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
    t0_init = []
    beta_init = []

    for mu_idx, isi in enumerate(isi_per_mu):
        # t_R para esta MU
        if np.isscalar(t_R_config):
            t_R = t_R_config
        else:
            t_R = t_R_config[mu_idx]
        
        if len(isi) < 2: ## En caso de que haya una sola isi
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
            
            # Priors (informatifs)
            # Prior para t0: centré autour de la moyenne des ISIs
            mean_t = np.mean(t)
            std_t = max(np.std(t), 1.0) # Cambiar decia que 0.7pero es mejor que sea adaptativo, luego se pueden probar mas cosas
            prior_t0 = 0.5 * ((t0 - mean_t) / std_t) ** 2 # ya esta en forma de negative log, 0.5 suaviza la penaliacion del prior
            
            # Prior para beta: centré autour de 2.0 avec σ=1.0
            prior_beta = 0.5 * ((beta - 2.0) / 1.0) ** 2 #  ya esta en forma de negative log
            
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
        else:
            print(f"⚠️ MU {mu_idx+1}: {result.message} | t0={t0_est:.2f}, beta={beta_est:.3f}")
        
        t0_init.append(t0_est)
        beta_init.append(beta_est)

    return np.array(t0_init), np.array(beta_init)



def estimate_weibull_moments(isi_per_mu, fs, config):
    """
    Estima t0 y beta usando el método de momentos (sin optimización iterativa).
    Muy rápido, robusto con pocos datos.
    
    Parámetros:
    - isi_per_mu: lista de arrays con ISIs por MU (en segundos)
    - fs: sampling rate (Hz)
    - config: dict con t_R (muestras o lista)
    
    Retorna:
    - t0_init: array con t0 estimado por MU (en muestras)
    - beta_init: array con beta estimado por MU
    """


    t_R_config = config["t_R"]
    t0_init = []
    beta_init = []

    for mu_idx, isi in enumerate(isi_per_mu):
        # t_R para esta MU
        if np.isscalar(t_R_config):
            t_R = t_R_config
        else:
            t_R = t_R_config[mu_idx]
        
        if len(isi) < 2:
            print(f"⚠️ MU {mu_idx+1}: solo {len(isi)} ISIs, usando valores por defecto")
            t0_init.append(1.2 * fs)
            beta_init.append(2.0)
            continue

        # ISI segundos → muestras
        t = np.round(isi * fs).astype(int)
        
        # Enlever la période réfractaire
        t_shifted = t - t_R
        t_shifted = t_shifted[t_shifted > 0]
        
        if len(t_shifted) < 2:
            print(f"⚠️ MU {mu_idx+1}: pas assez d'ISIs après t_R, valeurs par défaut")
            t0_init.append(1.2 * fs)
            beta_init.append(2.0)
            continue
        
        # Moments empiriques
        mean_t = np.mean(t_shifted)
        var_t = np.var(t_shifted)
        cv = np.sqrt(var_t) / mean_t if mean_t > 0 else 1.0
        
        # Déterminer β à partir du coefficient de variation
        # Valeurs typiques pour Weibull:
        # CV = 1.0 → β = 1.0 (exponentielle)
        # CV = 0.5 → β = 2.0 (Rayleigh)
        # CV = 0.3 → β = 3.0
        if cv < 0.4:
            beta_est = 3.5
        elif cv < 0.6:
            beta_est = 2.5
        elif cv < 0.8:
            beta_est = 2.0
        elif cv < 1.0:
            beta_est = 1.5
        else:
            beta_est = 1.2
        
        # Estimer t0 à partir de la moyenne: E[t] = t0 × Γ(1 + 1/β)
        t0_est = mean_t / gamma(1 + 1/beta_est)
        
        # Ajuster t0 pour inclure t_R
        t0_est = t0_est + t_R
        
        print("---METHOD OF MOMENTS---")
        print(f"✅ MU {mu_idx+1}: t0={t0_est:.2f} samples, beta={beta_est:.3f} | CV={cv:.3f} | ISIs={len(t)}")
        
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
    t0_init = []
    beta_init = []

    for mu_idx, isi in enumerate(isi_per_mu):
        # t_R para esta MU
        if np.isscalar(t_R_config):
            t_R = t_R_config
        else:
            t_R = t_R_config[mu_idx]
        
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
        
        # Grille de valeurs (plus fine si peu de données)
        mean_t = np.mean(t)
        std_t = np.std(t)
        
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
        else:
            print(f"⚠️ MU {mu_idx+1}: fine-tuning non convergé, utilisation grille | t0={best_params[0]:.2f}, beta={best_params[1]:.3f}")
            t0_est, beta_est = best_params
        
        t0_init.append(t0_est)
        beta_init.append(beta_est)

    return np.array(t0_init), np.array(beta_init)