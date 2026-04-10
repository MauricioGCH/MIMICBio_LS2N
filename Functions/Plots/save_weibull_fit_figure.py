import os
import numpy as np
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import Functions.Metrics as Metrics
from scipy import signal



def save_weibull_fit_figure(online_results, config, exp_path, save = False):
    """
    Figura interactiva ISI vs Weibull por MU
    """
    
    U_est = online_results["U_est"]
    Theta_est = online_results["Theta_est"]

    fs = config["sampling_rate"]
    t_R = config["t_R"]

    isi_per_mu = Metrics.extract_isi_per_mu(U_est)

    fig = go.Figure()

   
    for i, isi in enumerate(isi_per_mu):

        if len(isi) == 0:
            print(f"⚠️ No ISI for MU {i+1}")
            continue

        # 🔹 parámetros Weibull
        t0, beta = Theta_est[i]


        # 🔹 HISTOGRAMA
        fig.add_trace(go.Histogram(
            x=isi, ##
            nbinsx=30,
            histnorm='probability density',
            name=f'ISI MU {i+1}',
            opacity=0.5
        ))

        # 🔹 WEIBULL
        t_max = int(min(1.5 * np.max(isi), 3 * t0))
        t_vals = np.arange(t_R + 1, t_max + 1)
        t_vals_sec = t_vals / fs

        pmf = Metrics.weibull_discrete_pmf(t_vals, t0, beta, t_R)
        
        fig.add_trace(go.Scatter(
            x=t_vals, ##
            y=pmf,
            mode='lines',
            name=f'Weibull MU {i+1} (t0={t0:.1f}, β={beta:.2f})'
        ))

        # 🔹 línea vertical t0
        if t0 <= t_max:
            fig.add_trace(go.Scatter(
            x=[t0, t0],
            y=[0, np.max(pmf) if len(pmf) > 0 else 1],
            mode='lines',
            name=f"t0 MU {i+1}",
            line=dict(dash='dash', width=2)
        ))
        else:
            print(f"⚠️ t0 fuera de rango MU {i+1}")

        # 🔹 estadísticas (print)
        print(f"\n=== MU {i+1} ===")
        print(f"ISI count: {len(isi)}")
        print(f"Mean: {np.mean(isi):.4f} samples")
        print(f"Std: {np.std(isi):.4f} samples")
        print(f"MU {i+1}: t0={t0}, max ISI={np.max(isi)}")

    # --- layout ---
    fig.update_layout(
        title="ISI vs Weibull fit",
        xaxis_title="Inter-spike interval (Samples)",
        yaxis_title="Probability density",
        template="simple_white",
        barmode='overlay'
    )

    # --- guardar ---
    save_path = os.path.join(exp_path, "weibull_fit.html")
    if save:
        fig.write_html(save_path, include_plotlyjs=True, full_html=True, div_id="fig_weibull")

    #print(f"✔️ Weibull figure saved: {save_path}")
    return fig