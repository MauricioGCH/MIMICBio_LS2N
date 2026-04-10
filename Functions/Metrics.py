##Metrics
import os
import numpy as np
from matplotlib import pyplot as plt
import h5py

import yaml
import os
from datetime import datetime
import json

import plotly.graph_objects as go
from plotly.subplots import make_subplots

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


@staticmethod
def log_likelihood_weibull_refractory(isi_per_mu, theta_per_mu, t_R):
    """
    Calcula el log-likelihood TOTAL para múltiples unidades motoras.
    
    Parámetros:
    - isi_per_mu: lista de arrays, cada array contiene los ISI de una MU
    - theta_per_mu: lista de tuplas [(t0, beta), ...] para cada MU
    - t_R: período refractario en muestras (escalar, igual para todas las MUs)
    
    Retorna:
    - log_likelihood_total: float (suma de log-likelihoods de todas las MUs)
    - log_likelihood_promedio: float (promedio por muestra total)
    - log_likelihood_por_mu: lista de floats (individual por MU)
    """
    
    log_likelihood_por_mu = []
    n_total_muestras = 0
    
    for i, (isi, theta) in enumerate(zip(isi_per_mu, theta_per_mu)):
        
        # Si no hay ISIs para esta MU, saltar
        if len(isi) == 0:
            print(f"⚠️ MU {i+1}: No hay ISIs, LogL = -inf")
            log_likelihood_por_mu.append(-np.inf)
            continue
        
        t0, beta = theta
        
        # Verificar violaciones del período refractario
        if np.any(isi <= t_R):
            print(f"❌ MU {i+1}: {np.sum(isi <= t_R)} ISIs violan t_R={t_R}")
            log_likelihood_por_mu.append(-np.inf)
            continue
        
        # Calcular PMF para cada ISI de esta MU
        isi_f = isi.astype(float)
        denom = t0 - t_R
        
        # Prevenir división por cero
        if denom <= 0:
            print(f"❌ MU {i+1}: t0 ({t0}) <= t_R ({t_R})")
            log_likelihood_por_mu.append(-np.inf)
            continue
        
        arg = (isi_f - t_R) / denom
        arg_prev = (isi_f - 1 - t_R) / denom
        
        arg = np.maximum(arg, 0)
        arg_prev = np.maximum(arg_prev, 0)
        
        pmf = np.exp(-(arg_prev)**beta) - np.exp(-(arg)**beta)
        pmf = np.maximum(pmf, 1e-12)
        
        # Log-likelihood TOTAL de esta MU (suma, no promedio)
        logL_mu = np.sum(np.log(pmf))
        log_likelihood_por_mu.append(logL_mu)
        n_total_muestras += len(isi)
        
        print(f"MU {i+1}: n={len(isi)}, LogL={logL_mu:.4f}")
    
    # Log-likelihood TOTAL (suma de todas las MUs)
    log_likelihood_total = np.sum(log_likelihood_por_mu)
    
    # Log-likelihood PROMEDIO por muestra (para comparar con diferentes n)
    log_likelihood_promedio = log_likelihood_total / n_total_muestras if n_total_muestras > 0 else -np.inf
    
    return log_likelihood_total, log_likelihood_promedio, log_likelihood_por_mu