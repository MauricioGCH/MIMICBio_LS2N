##Modelisation functions
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import inv
import gc
from tqdm import tqdm

# ------ ALGO 2 --------
# t_R = 150 refractory period
def r(t, theta, t_R = 150):
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

def calculate_psi_scalar(t_i, i, ell_RI, n_MU):
    psi = np.zeros(ell_RI * n_MU)
    if 0 <= t_i < ell_RI:
        psi[i * ell_RI + t_i] = 1
    return psi

def calculate_psi(T_column, ell_RI, n_MU):
    psi = np.zeros(ell_RI * n_MU)
    for i in range(n_MU):
        t_i = T_column[i]
        if 0 <= t_i < ell_RI:
            psi[i * ell_RI + t_i] = 1
    return psi

def calculate_Y_hat(psi, H):
    return float(psi @ H)

def calculate_nu(psi, P, v):
    return float(psi @ P @ psi.T) + v

def calculate_K(P, psi, nu):
    return P @ psi.T / nu

def update_theta(theta, G, t_prev, t_curr, ell, t_R, epsilon=1e-6):
    grad_Q = calculate_gradient_Q(t_prev, t_curr, theta, t_R)
    G = (1 - 1 / ell) * G + (1 / ell) * np.outer(grad_Q, grad_Q)
    G_reg = G + epsilon * np.eye(2)  # régularisation pour assurer l'inversibilité
    theta = theta - (1 / ell) * np.linalg.inv(G_reg) @ grad_Q
    assert  np.isfinite(theta[0]) or np.isfinite(theta[1]) , f" theta = {theta}, G_reg = {G_reg}"
    return theta, G

'''def update_theta(theta, G_unused, t_prev, t_curr, ell, t_R, epsilon=1e-6):
    grad_Q = calculate_gradient_Q(t_prev, t_curr, theta, t_R)
    
    # Descente de gradient à pas 1/ell
    theta = theta - (1 / ell) * grad_Q

    # Vérification
    assert np.isfinite(theta[0]) and np.isfinite(theta[1]), f"theta invalide : {theta}"
    return theta, None  # pas de G à retourner'''

def kalman_update(y, psi, H_prev, P_prev, v):
    breakpoint()
    y_pred = float(psi @ H_prev)
    nu = float(psi @ P_prev @ psi.T) + v
    K = P_prev @ psi.T / nu
    H_new = H_prev + K * (y - y_pred) # No hay Correccion de H ? 
    P_new = P_prev - np.outer(K, K) * nu
    assert np.all(np.isfinite(P_prev)) or np.all(np.isfinite(K)) or np.all(np.isfinite(P_new)), f"K invalide :\n{K}"
    return H_new, P_new, y_pred, nu, K

def algorithm_2(Y, n_MU, t_R, ell_RI, n_s, ell_infinity, H0, P0, t_0, beta, v):
    #global t_0, beta
    n_samples = len(Y)
    V = 3.0

    # ---------------------- initialisation ----------------------
    sequences = {}
    for j in range(2 ** n_MU):
        t_vec = [np.random.randint(t_R + 1, 3 * t_R) for _ in range(n_MU)]
        sequences[j] = {
            't': t_vec,
            'theta': [np.array([t_0[i], beta[i]]) for i in range(n_MU)],
            'G': [np.eye(2) for _ in range(n_MU)],
            'H': np.concatenate(H0.copy()),
            'P': P0.copy(),
            'prob': 1.0,
            'spike_history': []  # historique des spikes
        }

    U_est = np.zeros((n_MU, n_samples))
    Y_est = np.zeros(n_samples)

    # ---------------------- boucle temps -----------------------
    for n in tqdm(range(n_samples),desc="Algorithm 2"):
        if n % 1000 == 0 and n > 1:
            #print(f"n = {n}, nb séquences = {len(sequences)}")
            gc.collect()

        y = Y[n]
        new_sequences = {}
        seq_id = 0

        # -------- parcours des séquences conservées -------------
        for seq in sequences.values():
            t_old     = seq['t']
            theta_old = seq['theta']
            G_old     = seq['G']
            H_old     = seq['H']
            P_old     = seq['P']
            prob_old  = seq['prob']
            spike_hist = seq['spike_history']

            # ----- bifurcations possibles pour cette séquence ----
            for j in range(2 ** n_MU):
                valid = True
                t_new = t_old.copy()
                spikes = []

                for k in range(n_MU):
                    spike_demanded = (j >> k) & 1
                    if spike_demanded and t_old[k] <= t_R:
                        valid = False
                        break
                if not valid:
                    continue

                for k in range(n_MU):
                    if (j >> k) & 1:
                        t_new[k] = 0
                        spikes.append(1)
                    else:
                        t_new[k] += 1
                        spikes.append(0)

                theta_new, G_new = [], []
                for i in range(n_MU):
                    th, Gup = update_theta(theta_old[i], G_old[i],
                                           t_old[i], t_new[i],
                                           ell_infinity, t_R)
                    theta_new.append(th)
                    G_new.append(Gup)

                psi = calculate_psi(t_new, ell_RI, n_MU)
                H_new, P_new, y_pred, nu, K = kalman_update(y, psi, H_old, P_old, v)

                p = prob_old * np.prod([
                    r(t_old[i] + 1, theta_old[i]) if spikes[i]
                    else 1 - r(t_old[i] + 1, theta_old[i])
                    for i in range(n_MU)
                ]) * np.exp(-0.5 * (y - y_pred) ** 2 / nu) / np.sqrt(2 * np.pi * nu)

                new_sequences[seq_id] = {
                    't': t_new,
                    'theta': theta_new,
                    'G': G_new,
                    'H': H_new,
                    'P': P_new,
                    'prob': p,
                    'spike_history': spike_hist + [spikes]
                }
                seq_id += 1

        total_prob = sum(s['prob'] for s in new_sequences.values())
        if total_prob == 0:
            raise RuntimeError(f"Aucune séquence valide à l’instant n = {n}")
        for s in new_sequences.values():
            s['prob'] /= total_prob

        sequences = dict(sorted(new_sequences.items(),
                                key=lambda x: -x[1]['prob'])[:n_s])

    # ------------------- reconstruction finale -----------------------
    best_seq = max(sequences.values(), key=lambda x: x['prob'])

    # Initialisation des compteurs t_i
    t_vec = [ell_RI for _ in range(n_MU)]
    
    # Reconstruction de U_est et Y_est
    for n, spikes in enumerate(best_seq['spike_history']):
        for i in range(n_MU):
            U_est[i, n] = spikes[i]
            if spikes[i] == 1:
                t_vec[i] = 0
            else:
                t_vec[i] += 1
    
        psi = calculate_psi(t_vec, ell_RI, n_MU)
        Y_est[n] = np.dot(psi, best_seq['H'])


    H_final = best_seq['H'].reshape(n_MU, ell_RI)
    Theta_final = best_seq['theta']
    '''t_0 = np.array([th[0] for th in Theta_final])
    beta = np.array([th[1] for th in Theta_final]'''
    return U_est, Y_est, H_final, Theta_final
