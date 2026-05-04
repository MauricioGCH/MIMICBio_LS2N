##Modelisation functions
import numpy as np
from scipy.linalg import inv
import gc
import time
import sys

# ------ ALGO 2 --------
# t_R = 150 refractory period
def r(t, theta, t_R):
    t0, beta = theta
    if t <= t_R:
        return 0
    
    return 1 - np.exp(((t - 1 - t_R) / (t0 - t_R)) ** beta - ((t - t_R) / (t0 - t_R)) ** beta)

def calculate_gradient_Q(t_prev, t_curr, theta, t_R, epsilon=1e-2):
    t0, beta = theta
    assert  np.isfinite(t0) and np.isfinite(beta) , f" theta = {theta}"

    if t_curr == 0:
        x1 = (t_prev - t_R) / (t0 - t_R)
        x2 = (t_prev + 1 - t_R) / (t0 - t_R)
        if abs(x1 - x2) < epsilon:
            assert x2 > 0 and np.isfinite(x2), f"x2 problématique : {x2} avec test abs(x1 - x2),x1 = {x1}, x2 = {x2}, t_current = {t_curr}, t_prev = {t_prev}, theta = {theta}"
            assert beta > 0 and np.isfinite(beta), f"beta problématique : {beta} avec test abs(x1 - x2),x1 = {x1}, x2 = {x2}, t_current = {t_curr}, t_prev = {t_prev}, theta = {theta}"
            dQ_dt0 = (beta / (t0 - t_R))   #degenere
            dQ_dbeta = 1/beta + np.log(x2)  #degenere
            return np.array([dQ_dt0, dQ_dbeta])
        assert x2 > 0 and np.isfinite(x2), f"x2 problématique : {x2} sans test abs(x1 - x2),x1 = {x1}, x2 = {x2}, t_current = {t_curr}, t_prev = {t_prev}, theta = {theta}"
        assert beta > 0 and np.isfinite(beta), f"beta problématique : {beta} sans test abs(x1 - x2), x1 = {x1}, x2 = {x2}, t_current = {t_curr}, t_prev = {t_prev}, theta = {theta}"
        exp_term = np.exp(-(x1**beta) + (x2**beta))
        denom = 1 - exp_term
        dQ_dt0 = (beta / (t0 - t_R)) * (x1**beta - x2**beta) / denom 
        dQ_dbeta = (np.log(x1) * x1**beta - np.log(x2) * x2**beta) / denom 
    elif t_curr > t_R:
        x1 = max((t_prev - t_R) / (t0 - t_R), epsilon)
        x2 = max((t_prev + 1 - t_R) / (t0 - t_R), epsilon)
        assert x2 > 0 and np.isfinite(x2), f"x2 problématique : {x2} dans elif,x1 = {x1}, x2 = {x2}, t_current = {t_curr}, t_prev = {t_prev}, theta = {theta}"
        assert beta > 0 and np.isfinite(beta), f"beta problématique : {beta} dans elif, x1 = {x1}, x2 = {x2}, t_current = {t_curr}, t_prev = {t_prev}, theta = {theta}"
        dQ_dt0 = (beta / (t0 - t_R)) * (x1**beta - x2**beta)  
        dQ_dbeta = np.log(x1) * x1**beta - np.log(x2) * x2**beta  
    else:
        return np.zeros(2)
    return np.array([dQ_dt0, dQ_dbeta])

def update_theta(theta, G, t_prev, t_curr, ell, t_R, config, epsilon=1e-6):
    if config["Gradient_Q"] == "Fast":
        grad_Q = calculate_gradient_Q_fast(t_prev, t_curr, theta, t_R)
    elif config["Gradient_Q"] == "Regular":
        grad_Q = calculate_gradient_Q(t_prev, t_curr, theta, t_R)
    ell = np.array(ell)
    
    G = (1 - 1 / ell) * G + (1 / ell) * np.outer(grad_Q, grad_Q)
    G_reg = G + epsilon * np.eye(2)  # régularisation pour assurer l'inversibilité
    theta = theta - (1 / ell) * np.linalg.inv(G_reg) @ grad_Q
    assert  np.isfinite(theta[0]) or np.isfinite(theta[1]) , f" theta = {theta}, G_reg = {G_reg}"
    return theta, G


def calculate_gradient_Q_fast(t_prev, t_curr, theta, t_R, epsilon=1e-2, debug=False):
    """
    Versión optimizada de calculate_gradient_Q con asserts opcionales.

    Parámetros
    ----------
    t_prev : int/float
    t_curr : int/float
    theta : array-like -> [t0, beta]
    t_R : int/float
    epsilon : float
    debug : bool (default=False)
        Si True activa los mismos asserts diagnósticos
        de la versión original.
    """

    t0, beta = theta

    # --------------------------------------------------
    # ASSERT GLOBAL
    # --------------------------------------------------
    if debug:
        assert np.isfinite(t0) and np.isfinite(beta), f"theta = {theta}"

    # --------------------------------------------------
    # PRECALCULOS
    # --------------------------------------------------
    t0_minus_tR = t0 - t_R
    inv_t0_minus_tR = 1.0 / t0_minus_tR

    # ==================================================
    # CASO 1: spike (reset)
    # ==================================================
    if t_curr == 0:

        t_prev_minus_tR = t_prev - t_R
        t_prev_plus1_minus_tR = t_prev + 1 - t_R

        x1 = t_prev_minus_tR * inv_t0_minus_tR
        x2 = t_prev_plus1_minus_tR * inv_t0_minus_tR

        # ----------------------------------------------
        # Caso degenerado
        # ----------------------------------------------
        if abs(x1 - x2) < epsilon:

            if debug:
                assert x2 > 0 and np.isfinite(x2), (
                    f"x2 problématique : {x2} avec test abs(x1-x2), "
                    f"x1={x1}, x2={x2}, t_curr={t_curr}, "
                    f"t_prev={t_prev}, theta={theta}"
                )

                assert beta > 0 and np.isfinite(beta), (
                    f"beta problématique : {beta} avec test abs(x1-x2), "
                    f"x1={x1}, x2={x2}, t_curr={t_curr}, "
                    f"t_prev={t_prev}, theta={theta}"
                )

            dQ_dt0 = beta * inv_t0_minus_tR
            dQ_dbeta = 1.0 / beta + np.log(x2)

            return np.array([dQ_dt0, dQ_dbeta])

        # ----------------------------------------------
        # Caso normal
        # ----------------------------------------------
        if debug:
            assert x2 > 0 and np.isfinite(x2), (
                f"x2 problématique : {x2} sans test abs(x1-x2), "
                f"x1={x1}, x2={x2}, t_curr={t_curr}, "
                f"t_prev={t_prev}, theta={theta}"
            )

            assert beta > 0 and np.isfinite(beta), (
                f"beta problématique : {beta} sans test abs(x1-x2), "
                f"x1={x1}, x2={x2}, t_curr={t_curr}, "
                f"t_prev={t_prev}, theta={theta}"
            )

        x1_beta = x1 ** beta
        x2_beta = x2 ** beta

        exp_term = np.exp(-x1_beta + x2_beta)
        denom = 1.0 - exp_term
        inv_denom = 1.0 / denom

        dQ_dt0 = beta * inv_t0_minus_tR * (x1_beta - x2_beta) * inv_denom
        dQ_dbeta = (
            np.log(x1) * x1_beta
            - np.log(x2) * x2_beta
        ) * inv_denom

    # ==================================================
    # CASO 2: no spike, supervivencia
    # ==================================================
    elif t_curr > t_R:

        x1 = max((t_prev - t_R) * inv_t0_minus_tR, epsilon)
        x2 = max((t_prev + 1 - t_R) * inv_t0_minus_tR, epsilon)

        if debug:
            assert x2 > 0 and np.isfinite(x2), (
                f"x2 problématique : {x2} dans elif, "
                f"x1={x1}, x2={x2}, t_curr={t_curr}, "
                f"t_prev={t_prev}, theta={theta}"
            )

            assert beta > 0 and np.isfinite(beta), (
                f"beta problématique : {beta} dans elif, "
                f"x1={x1}, x2={x2}, t_curr={t_curr}, "
                f"t_prev={t_prev}, theta={theta}"
            )

        x1_beta = x1 ** beta
        x2_beta = x2 ** beta

        dQ_dt0 = beta * inv_t0_minus_tR * (x1_beta - x2_beta)
        dQ_dbeta = (
            np.log(x1) * x1_beta
            - np.log(x2) * x2_beta
        )

    # ==================================================
    # CASO 3: refractario absoluto
    # ==================================================
    else:
        return np.zeros(2)

    return np.array([dQ_dt0, dQ_dbeta])

def update_theta_fast(theta, G, t_prev, t_curr, ell, t_R, config, eps=1e-6):
    if config["Gradient_Q"] == "Fast":
        grad_Q = calculate_gradient_Q_fast(t_prev, t_curr, theta, t_R)
    elif config["Gradient_Q"] == "Regular":
        grad_Q = calculate_gradient_Q(t_prev, t_curr, theta, t_R)
    
    ell = np.array(ell)
    # Optimizado: sin np.outer
    
    f1 = 1 - 1/ell
    
    f2 = 1/ell
    g0, g1 = grad_Q[0], grad_Q[1]
    G = f1 * G + f2 * np.array([[g0*g0, g0*g1], [g1*g0, g1*g1]])
    
    # Optimizado: inversión analítica
    G_reg = G + eps * np.eye(2)
    theta = theta - f2 * (inv_2x2_safe(G_reg) @ grad_Q)
    
    return theta, G



def calculate_psi(T_column, ell_RI, n_MU):
    psi = np.zeros(ell_RI * n_MU)
    for i in range(n_MU):
        t_i = T_column[i]
        if 0 <= t_i < ell_RI:
            psi[i * ell_RI + t_i] = 1
    return psi
# Replace calculate_psi with this inline version inside the loop:
def fill_psi(psi_buffer, T_column, ell_RI, n_MU):
    """Fill existing buffer without reallocation"""
    psi_buffer.fill(0)  # Reset
    for i in range(n_MU):
        t_i = T_column[i]
        if 0 <= t_i < ell_RI:
            psi_buffer[i * ell_RI + t_i] = 1
    return psi_buffer


def calculate_K(P, psi, nu):
    return P @ psi.T / nu


def kalman_update(y, psi, H_prev, P_prev, v):
    
    y_pred = float(psi @ H_prev) #80 vs 80
    nu = float(psi @ P_prev @ psi.T) + v
    K = P_prev @ psi.T / nu
    H_new = H_prev + K * (y - y_pred) # No hay Correccion de H ? 
    P_new = P_prev - np.outer(K, K) * nu
    assert np.all(np.isfinite(P_prev)) or np.all(np.isfinite(K)) or np.all(np.isfinite(P_new)), f"K invalide :\n{K}"
    return H_new, P_new, y_pred, nu, K


def kalman_update_fast(y, psi, H, P, v):
    # Aprovechar sparsity de psi
    idx = np.where(psi > 0)[0]
    
    y_pred = np.sum(H[idx])
    nu = v + np.sum(P[idx][:, idx])
    
    K = np.zeros_like(H)
    K[idx] = P[idx][:, idx] @ np.ones(len(idx)) / nu
    
    error = y - y_pred
    H_new = H + K * error
    P_new = P - np.outer(K, K) * nu
    
    return H_new, P_new, y_pred, nu, K
def inv_2x2(M, eps=1e-12):
    a,b,c,d = M[0,0], M[0,1], M[1,0], M[1,1]
    det = a*d - b*c
    return np.array([[d,-b],[-c,a]])/det if abs(det)>eps else np.eye(2)

def inv_2x2_safe(M, eps=1e-12):
    """Inversión 2×2 rápida pero con fallback seguro."""
    a, b, c, d = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
    det = a * d - b * c
    
    if abs(det) < eps:
        print("Matriz casi singular, se usa inv the scipy")
        # Si es casi singular, usar scipy (más robusto)
        from scipy.linalg import inv
        return inv(M)
    
    return np.array([[d, -b], [-c, a]]) / det




# An alterantive tothe Kalman FIlter is the Least Means Square or LMS, Eric said it was an option they evaluated and determined it was possible.

def lms_update(y, psi, H, mu):
    """
    LMS update para reemplazar el filtro de Kalman.
    
    Parámetros:
    - y: muestra actual
    - psi: vector de observación (sparse, un 1 por MU)
    - H: vector de filtro actual (n_MU * ell_RI)
    - mu: paso de aprendizaje (learning rate)
    
    Retorna:
    - H_new: filtro actualizado
    - y_pred: predicción
    """
    y_pred = float(psi @ H)
    error = y - y_pred
    H_new = H + mu * error * psi
    return H_new, y_pred

def algorithm_2(Y, n_MU, t_R_vec, ell_RI, n_s, ell_infinity, H0, P0, t_0, beta, v, config):
    n_samples = len(Y)

    # ==================================================
    # 🔹 CONFIGURACIÓN
    # ==================================================
    h_estimator   = config["H_update"]
    lms_step_size = config["lms_step_size"]

    if h_estimator not in ("Kalman", "LMS", "Kalman Fast"):
        raise ValueError(f"H_estimator desconocido: '{h_estimator}'")

    print(f"🔧 H_estimator: {h_estimator}" +
          (f" | lms_step_size={lms_step_size}" if h_estimator == "LMS" else ""))

    # ==================================================
    # 🔹 INITIALISATION
    # ==================================================
    sequences = {}
    for j in range(2 ** n_MU):
        t_vec = [np.random.randint(t_R_vec[i] + 1, 3 * t_R_vec[i]) for i in range(n_MU)]

        seq = {
            't': t_vec,
            'theta': [np.array([t_0[i], beta[i]]) for i in range(n_MU)],
            'G': [np.eye(2) for _ in range(n_MU)],
            'H': np.concatenate(H0.copy()),
            'prob': 1.0,
            'spike_history': []
        }

        if h_estimator in ("Kalman", "Kalman Fast"):
            seq['P'] = P0.copy()

        sequences[j] = seq

    U_est = np.zeros((n_MU, n_samples))
    Y_est = np.zeros(n_samples)
    theta_history = []
    psi_buffer = np.zeros(ell_RI * n_MU)

    # ==================================================
    # 🔹 TIMER
    # ==================================================
    update_every = max(500, n_samples // 100)
    bar_width = 40
    t0_loop = time.perf_counter()

    # ==================================================
    # 🔹 BUCLE TEMPORAL
    # ==================================================
    for n in range(n_samples):

        # Progress bar
        if (n % update_every == 0) or (n == n_samples - 1):
            frac   = (n + 1) / n_samples
            filled = int(bar_width * frac)
            bar    = "█" * filled + "-" * (bar_width - filled)
            elapsed = time.perf_counter() - t0_loop
            rate    = (n + 1) / elapsed
            remain  = (n_samples - n - 1) / rate

            sys.stdout.write(
                f"\r[{bar}] {100*frac:6.2f}% | {n+1}/{n_samples} | ETA {remain:6.1f}s"
            )
            sys.stdout.flush()

        if n % 1000 == 0 and n > 1:
            gc.collect()

        y = Y[n]
        new_sequences = {}
        seq_id = 0

        # ==================================================
        # 🔹 LOOP SOBRE SECUENCIAS
        # ==================================================
        for seq in sequences.values():

            t_old      = seq['t']
            theta_old  = seq['theta']
            G_old      = seq['G']
            H_old      = seq['H']
            prob_old   = seq['prob']
            spike_hist = seq['spike_history']
            P_old      = seq.get('P', None)

            # ==================================================
            # 🔹 GENERACIÓN DIRECTA (SIN PRECOMPUTAR)
            # ==================================================
            for j in range(2 ** n_MU):

                valid = True
                t_new = t_old.copy()
                spikes = []

                # -------- validar transición ----------
                for k in range(n_MU):
                    spike_demanded = (j >> k) & 1

                    if spike_demanded and t_old[k] <= t_R_vec[k]:
                        valid = False
                        break

                if not valid:
                    continue

                # -------- construir estado ----------
                for k in range(n_MU):
                    if (j >> k) & 1:
                        t_new[k] = 0
                        spikes.append(1)
                    else:
                        t_new[k] += 1
                        spikes.append(0)

                # ==================================================
                # 🔹 UPDATE THETA
                # ==================================================
                theta_new, G_new = [], []

                for i in range(n_MU):
                    if config["theta_update"] == "Fast":
                        th, Gup = update_theta_fast(
                            theta_old[i], G_old[i],
                            t_old[i], t_new[i],
                            ell_infinity, t_R_vec[i],
                            config=config
                        )
                    else:
                        th, Gup = update_theta(
                            theta_old[i], G_old[i],
                            t_old[i], t_new[i],
                            ell_infinity, t_R_vec[i],
                            config
                        )

                    theta_new.append(th)
                    G_new.append(Gup)

                # ==================================================
                # 🔹 PSI
                # ==================================================
                psi = fill_psi(psi_buffer, t_new, ell_RI, n_MU)

                # ==================================================
                # 🔹 UPDATE H
                # ==================================================
                if h_estimator == "Kalman Fast":
                    H_new, P_new, y_pred, nu, _ = kalman_update_fast(y, psi, H_old, P_old, v)
                    likelihood = np.exp(-0.5 * (y - y_pred)**2 / nu) / np.sqrt(2*np.pi*nu)

                elif h_estimator == "Kalman":
                    H_new, P_new, y_pred, nu, _ = kalman_update(y, psi, H_old, P_old, v)
                    likelihood = np.exp(-0.5 * (y - y_pred)**2 / nu) / np.sqrt(2*np.pi*nu)

                else:  # LMS
                    y_pred = float(psi @ H_old)
                    H_new  = H_old + lms_step_size * (y - y_pred) * psi
                    P_new  = None
                    likelihood = np.exp(-0.5 * (y - y_pred)**2 / v) / np.sqrt(2*np.pi*v)

                # ==================================================
                # 🔹 PROBABILIDAD
                # ==================================================
                p = prob_old * np.prod([
                    r(t_old[i] + 1, theta_old[i], t_R_vec[i]) if spikes[i]
                    else 1 - r(t_old[i] + 1, theta_old[i], t_R_vec[i])
                    for i in range(n_MU)
                ]) * likelihood

                new_seq = {
                    't': t_new,
                    'theta': theta_new,
                    'G': G_new,
                    'H': H_new,
                    'prob': p,
                    'spike_history': spike_hist + [spikes]
                }

                if h_estimator in ("Kalman", "Kalman Fast"):
                    new_seq['P'] = P_new

                new_sequences[seq_id] = new_seq
                seq_id += 1

        # ==================================================
        # 🔹 NORMALIZACIÓN
        # ==================================================
        total_prob = sum(s['prob'] for s in new_sequences.values())

        if total_prob == 0:
            raise RuntimeError(f"⚠️ No hay secuencias válidas en n={n}")

        for s in new_sequences.values():
            s['prob'] /= total_prob

        sequences = dict(sorted(
            new_sequences.items(),
            key=lambda x: -x[1]['prob']
        )[:n_s])

        # ==================================================
        # 🔹 TRACKING THETA
        # ==================================================
        if n % 1000 == 0:
            best_seq = max(sequences.values(), key=lambda x: x['prob'])
            for mu_idx in range(n_MU):
                theta_history.append([
                    n,
                    mu_idx,
                    best_seq['theta'][mu_idx][0],
                    best_seq['theta'][mu_idx][1]
                ])

    print("\nDone.")

    # ==================================================
    # 🔹 RECONSTRUCCIÓN FINAL
    # ==================================================
    best_seq = max(sequences.values(), key=lambda x: x['prob'])

    t_vec = [ell_RI for _ in range(n_MU)]

    for n, spikes in enumerate(best_seq['spike_history']):
        for i in range(n_MU):
            U_est[i, n] = spikes[i]
            t_vec[i] = 0 if spikes[i] == 1 else t_vec[i] + 1

        psi = calculate_psi(t_vec, ell_RI, n_MU)
        Y_est[n] = np.dot(psi, best_seq['H'])

    H_final       = best_seq['H'].reshape(n_MU, ell_RI)
    Theta_final   = best_seq['theta']
    theta_history = np.array(theta_history)

    return U_est, Y_est, H_final, Theta_final, theta_history






def algorithm_2_Error(Y, n_MU, t_R_vec, ell_RI, n_s, ell_infinity, H0, P0, t_0, beta, v, config):
    n_samples = len(Y)

    # ==================================================
    # 🔹 CONFIGURACIÓN DEL ESTIMADOR
    # ==================================================
    h_estimator   = config["H_update"]
    lms_step_size = config["lms_step_size"]

    if h_estimator not in ("Kalman", "LMS", "Kalman Fast"):
        raise ValueError(f"H_estimator desconocido: '{h_estimator}'. Usa 'Kalman' o 'LMS'.")

    print(f"🔧 H_estimator: {h_estimator}" + (f" | lms_step_size={lms_step_size}" if h_estimator == "LMS" else ""))

    # ==================================================
    # 🔹 PRECOMPUTAR TRANSICIONES
    # ==================================================
    all_spike_combinations = []
    for j in range(2 ** n_MU):
        spikes = [(j >> k) & 1 for k in range(n_MU)]
        all_spike_combinations.append((j, spikes))

    # ==================================================
    # 🔹 INITIALISATION
    # ==================================================
    sequences = {}
    for j in range(2 ** n_MU):
        t_vec = [np.random.randint(t_R_vec[i] + 1, 3 * t_R_vec[i]) for i in range(n_MU)]

        seq = {
            't': t_vec,
            'theta': [np.array([t_0[i], beta[i]]) for i in range(n_MU)],
            'G': [np.eye(2) for _ in range(n_MU)],
            'H': np.concatenate(H0.copy()),
            'prob': 1.0,
            'spike_history': []
        }
        if h_estimator == "Kalman" or h_estimator == "Kalman Fast":
            seq['P'] = P0.copy()

        sequences[j] = seq

    U_est = np.zeros((n_MU, n_samples))
    Y_est = np.zeros(n_samples)
    theta_history = []
    psi_buffer = np.zeros(ell_RI * n_MU)

    # ==================================================
    # 🔹 TIMER
    # ==================================================
    update_every = max(500, n_samples // 100)
    bar_width = 40
    t0_loop = time.perf_counter()

    # ==================================================
    # 🔹 BOUCLE TEMPS
    # ==================================================
    for n in range(n_samples):

        # Progress bar
        if (n % update_every == 0) or (n == n_samples - 1):
            frac   = (n + 1) / n_samples
            filled = int(bar_width * frac)
            bar    = "█" * filled + "-" * (bar_width - filled)
            elapsed = time.perf_counter() - t0_loop
            rate    = (n + 1) / elapsed
            remain  = (n_samples - n - 1) / rate
            sys.stdout.write(
                f"\r[{bar}] {100*frac:6.2f}% | "
                f"{n+1}/{n_samples} | ETA {remain:6.1f}s"
            )
            sys.stdout.flush()

        if n % 1000 == 0 and n > 1:
            gc.collect()

        y = Y[n]
        new_sequences = {}
        seq_id = 0

        for seq in sequences.values():
            t_old      = seq['t']
            theta_old  = seq['theta']
            G_old      = seq['G']
            H_old      = seq['H']
            prob_old   = seq['prob']
            spike_hist = seq['spike_history']
            P_old      = seq['P'] if h_estimator == "Kalman" or h_estimator == "Kalman Fast" else None

            # ----- FILTRAR TRANSICIONES VÁLIDAS -----
            valid_transitions = []
            for j, spikes in all_spike_combinations:
                valid = True
                for k in range(n_MU):
                    if spikes[k] and t_old[k] <= t_R_vec[k]:
                        valid = False
                        break
                if not valid:
                    continue

                t_new = t_old.copy()
                for k in range(n_MU):
                    t_new[k] = 0 if spikes[k] else t_new[k] + 1

                valid_transitions.append((j, spikes, t_new))

            if len(valid_transitions) == 0: ## This should nver happen as  transition 0,0 is always valid
                spikes = [0] * n_MU
                t_new  = [t_old[k] + 1 for k in range(n_MU)]
                valid_transitions.append((0, spikes, t_new))

            # ----- PROCESAR TRANSICIONES -----
            for j, spikes, t_new in valid_transitions:

                # Actualizar theta
                theta_new, G_new = [], []
                for i in range(n_MU):

                    if config["theta_update"] == "Fast":
                        th, Gup = update_theta_fast(
                            theta_old[i], G_old[i],
                            t_old[i], t_new[i],
                            ell_infinity, t_R_vec[i], config= config
                        )
                        theta_new.append(th)
                        G_new.append(Gup)
                    elif config["theta_update"] == "Regular":
                        th, Gup = update_theta(
                            theta_old[i], G_old[i],
                            t_old[i], t_new[i],
                            ell_infinity, t_R_vec[i], config
                        )
                        theta_new.append(th)
                        G_new.append(Gup)


                # Calcular psi
                psi = fill_psi(psi_buffer, t_new, ell_RI, n_MU)

                # ==================================================
                # 🔹 ACTUALIZACIÓN DE H
                # ==================================================
                if h_estimator == "Kalman Fast":
                    H_new, P_new, y_pred, nu, _ = kalman_update_fast(y, psi, H_old, P_old, v)
                    likelihood = (
                        np.exp(-0.5 * (y - y_pred) ** 2 / nu)
                        / np.sqrt(2 * np.pi * nu)
                    )
                elif h_estimator == "Kalman":
                    H_new, P_new, y_pred, nu, _ = kalman_update(y, psi, H_old, P_old, v)
                    likelihood = (
                        np.exp(-0.5 * (y - y_pred) ** 2 / nu)
                        / np.sqrt(2 * np.pi * nu)
                    )
                else:  # LMS
                    y_pred = float(psi @ H_old)
                    H_new  = H_old + lms_step_size * (y - y_pred) * psi
                    P_new  = None
                    likelihood = (
                        np.exp(-0.5 * (y - y_pred) ** 2 / v)
                        / np.sqrt(2 * np.pi * v)
                    )

                # Probabilidad de la transición
                p = prob_old * np.prod([
                    r(t_old[i] + 1, theta_old[i], t_R_vec[i]) if spikes[i]
                    else 1 - r(t_old[i] + 1, theta_old[i], t_R_vec[i])
                    for i in range(n_MU)
                ]) * likelihood

                new_seq = {
                    't': t_new,
                    'theta': theta_new,
                    'G': G_new,
                    'H': H_new,
                    'prob': p,
                    'spike_history': spike_hist + [spikes]
                }
                if h_estimator == "Kalman" or h_estimator == "Kalman Fast":
                    new_seq['P'] = P_new

                new_sequences[seq_id] = new_seq
                seq_id += 1

        # ==================================================
        # 🔹 NORMALIZACIÓN
        # ==================================================
        total_prob = sum(s['prob'] for s in new_sequences.values())
        if total_prob == 0:
            print(f"\n⚠️ Aucune séquence valide à n = {n}")
            print(f"   Nombre de séquences avant: {len(sequences)}")
            print(f"   t_old d'une séquence exemple: {list(sequences.values())[0]['t']}")
            print(f"   t_R_vec: {t_R_vec}")

            for seq in sequences.values():
                t_new  = [seq['t'][k] + 1 for k in range(n_MU)]
                spikes = [0] * n_MU

                theta_new, G_new = [], []
                for i in range(n_MU):
                    if config["theta_update"] == "Fast":
                        th, Gup = update_theta_fast(
                            theta_old[i], G_old[i],
                            t_old[i], t_new[i],
                            ell_infinity, t_R_vec[i], config= config
                        )
                        theta_new.append(th)
                        G_new.append(Gup)
                    elif config["theta_update"] == "Regular":
                        th, Gup = update_theta(
                            theta_old[i], G_old[i],
                            t_old[i], t_new[i],
                            ell_infinity, t_R_vec[i], config
                        )
                        theta_new.append(th)
                        G_new.append(Gup)
                    # th, Gup = update_theta_fast(
                    #     seq['theta'][i], seq['G'][i],
                    #     seq['t'][i], t_new[i],
                    #     ell_infinity, t_R_vec[i], config= config
                    # )
                    # theta_new.append(th)
                    # G_new.append(Gup)

                psi = fill_psi(psi_buffer, t_new, ell_RI, n_MU)

                if h_estimator == "Kalman Fast":
                    H_new, P_new, y_pred, nu, _ = kalman_update_fast(y, psi, seq['H'], seq['P'], v)
                    likelihood = np.exp(-0.5*(y-y_pred)**2/nu) / np.sqrt(2*np.pi*nu)
                elif h_estimator == "Kalman":
                    H_new, P_new, y_pred, nu, _ = kalman_update(y, psi, seq['H'], seq['P'], v)
                    likelihood = np.exp(-0.5*(y-y_pred)**2/nu) / np.sqrt(2*np.pi*nu)
                else:
                    y_pred = float(psi @ seq['H'])
                    H_new  = seq['H'] + lms_step_size * (y - y_pred) * psi
                    P_new  = None
                    likelihood = np.exp(-0.5*(y-y_pred)**2/v) / np.sqrt(2*np.pi*v)

                new_seq = {
                    't': t_new,
                    'theta': theta_new,
                    'G': G_new,
                    'H': H_new,
                    'prob': seq['prob'] * likelihood,
                    'spike_history': seq['spike_history'] + [spikes]
                }
                if h_estimator == "Kalman" or h_estimator == "Kalman Fast" :
                    new_seq['P'] = P_new

                new_sequences[seq_id] = new_seq
                seq_id += 1

            total_prob = sum(s['prob'] for s in new_sequences.values())

        for s in new_sequences.values():
            s['prob'] /= total_prob

        sequences = dict(sorted(new_sequences.items(),
                                key=lambda x: -x[1]['prob'])[:n_s])

        # Guardar historial cada 1000 muestras
        if n % 1000 == 0:
            best_seq_at_time = max(sequences.values(), key=lambda x: x['prob'])
            for mu_idx in range(n_MU):
                theta_history.append([
                    n, mu_idx,
                    best_seq_at_time['theta'][mu_idx][0],
                    best_seq_at_time['theta'][mu_idx][1]
                ])

    print("\nDone.")

    # ==================================================
    # 🔹 RECONSTRUCCIÓN FINAL — igual para ambos métodos
    # ==================================================
    best_seq = max(sequences.values(), key=lambda x: x['prob'])
    t_vec = [ell_RI for _ in range(n_MU)]

    for n, spikes in enumerate(best_seq['spike_history']):
        for i in range(n_MU):
            U_est[i, n] = spikes[i]
            t_vec[i] = 0 if spikes[i] == 1 else t_vec[i] + 1

        psi = calculate_psi(t_vec, ell_RI, n_MU)
        Y_est[n] = np.dot(psi, best_seq['H'])

    H_final       = best_seq['H'].reshape(n_MU, ell_RI)
    Theta_final   = best_seq['theta']
    theta_history = np.array(theta_history)

    return U_est, Y_est, H_final, Theta_final, theta_history