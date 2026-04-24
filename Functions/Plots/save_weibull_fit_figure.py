import os
import numpy as np
import plotly.graph_objects as go
import Functions.Metrics as Metrics



def save_weibull_fit_figure(online_results, config, exp_path, save=False):
    """
    Figura interactiva ISI vs Weibull por MU (versión online)
    """
    from plotly.subplots import make_subplots
    import numpy as np
    import os
    
    print("--- Generation of weibull distribution (online) ---")
    
    U_est = online_results["U_est"]
    Theta_est = online_results["Theta_est"]
    
    fs = config["sampling_rate"]
    t_R_list = config["t_R"]  # lista por MU (en muestras)
    
    # Extraer ISI por MU
    isi_per_mu = Metrics.extract_isi_per_mu(U_est)
    n_mus = len(isi_per_mu)
    
    # Colores para diferentes MUs
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Crear subplots verticales para cada MU
    fig = make_subplots(
        rows=n_mus, cols=1,
        subplot_titles=[f"Source {mu_idx+1} - Distribution ISI (n={len(isi_per_mu[mu_idx])} intervalles)" 
                       for mu_idx in range(n_mus)],
        shared_xaxes=False,
        vertical_spacing=0.08
    )
    
    for mu_idx in range(n_mus):
        isi = isi_per_mu[mu_idx]
        color = colors[mu_idx % len(colors)]
        
        if len(isi) == 0:
            continue
        
        t0, beta = Theta_est[mu_idx]
        t_R = t_R_list[mu_idx]
        
        # ===== 1. CONVERTIR A MILISEGUNDOS =====
        isi_ms = isi * 1000 / fs
        t0_ms = (t0 / fs) * 1000
        t_R_ms = (t_R / fs) * 1000
        
        # ===== 2. HISTOGRAMA EXPERIMENTAL =====
        fig.add_trace(
            go.Histogram(
                x=isi_ms,
                nbinsx=40,
                histnorm='probability density',
                name=f'Source {mu_idx+1} - Experimental',
                marker=dict(color=color, line=dict(color='black', width=0.5)),
                opacity=0.7,
                showlegend=(mu_idx == 0)
            ),
            row=mu_idx+1, col=1
        )
        
        # ===== 3. CURVA WEIBULL TEÓRICA =====
        max_isi_ms = np.percentile(isi_ms, 99) if len(isi_ms) > 0 else 500
        t_ms = np.linspace(t_R_ms, max_isi_ms, 500)
        t_samples = t_ms * fs / 1000
        
        pdf_samples = Metrics.weibull_discrete_pmf(t_samples, t0, beta, t_R)
        pdf_per_ms = pdf_samples * fs / 1000
        
        fig.add_trace(
            go.Scatter(
                x=t_ms,
                y=pdf_per_ms,
                mode='lines',
                name=f'Source {mu_idx+1} - Weibull théorique',
                line=dict(width=2.5, color=color, dash='dash'),
                showlegend=(mu_idx == 0)
            ),
            row=mu_idx+1, col=1
        )
        
        # ===== 4. LÍNEA VERTICAL t₀ =====
        fig.add_vline(
            x=t0_ms,
            line_dash="dot",
            line_color="green",
            line_width=2,
            annotation_text=f"t₀ = {t0_ms:.1f}ms",
            annotation_position="top",
            row=mu_idx+1, col=1
        )
        
        # ===== 5. LÍNEA VERTICAL t_R =====
        fig.add_vline(
            x=t_R_ms,
            line_dash="dash",
            line_color="purple",
            line_width=2,
            annotation_text=f"t_R = {t_R_ms:.1f}ms",
            annotation_position="bottom",
            row=mu_idx+1, col=1
        )
        
         # ===== 6. LEYENDA CON VALORES TEÓRICOS (t₀ y β) =====
        weibull_params = (
            f"<b>Weibull théorique (online):</b><br>"
            f"t₀ = {t0_ms:.1f} ms<br>"
            f"β = {beta:.3f}<br>"
            f"t_R = {t_R_ms:.1f} ms"
        )
        
        # Para n_mus=2:
        # - MU0 (mu_idx=0): y_paper = 0.65 (dentro del subplot superior)
        # - MU1 (mu_idx=1): y_paper = 0.15 (dentro del subplot inferior)
        if n_mus == 2:
            if mu_idx == 0:
                y_paper = 0.70  # Subplot superior
            else:
                y_paper = 0.20  # Subplot inferior
        else:
            y_paper = 0.75  # Fallback
        
        fig.add_annotation(
            text=weibull_params,
            xref="paper",
            yref="paper",
            x=0.75,
            y=y_paper,
            xanchor='left',
            yanchor='top',
            showarrow=False,
            font=dict(size=9, family='monospace', color=color),
            bgcolor='rgba(255, 255, 255, 0.95)',
            bordercolor=color,
            borderwidth=1.5,
            borderpad=6,
            align='left'
        )
        
        # Print debug
        print(f"\n=== Source  {mu_idx+1} ===")
        print(f"ISI count: {len(isi)}")
        print(f"t₀ estimé: {t0_ms:.2f} ms")
        print(f"β estimé: {beta:.3f}")
        print(f"t_R: {t_R_ms:.2f} ms")
    
    # ===== 7. CALCULAR LÍMITE Y REAL =====
    y_max_limit = 0
    for mu_idx in range(n_mus):
        if len(isi_per_mu[mu_idx]) > 0:
            isi_ms = isi_per_mu[mu_idx] * 1000 / fs
            hist, _ = np.histogram(isi_ms, bins=40, density=True)
            y_max_limit = max(y_max_limit, np.max(hist))
            
            t0, beta = Theta_est[mu_idx]
            t_R = t_R_list[mu_idx]
            t_R_ms = (t_R / fs) * 1000
            max_isi_ms = np.percentile(isi_ms, 99)
            t_ms = np.linspace(t_R_ms, max_isi_ms, 500)
            t_samples = t_ms * fs / 1000
            pdf_samples = Metrics.weibull_discrete_pmf(t_samples, t0, beta, t_R)
            pdf_per_ms = pdf_samples * fs / 1000
            y_max_limit = max(y_max_limit, np.max(pdf_per_ms))
    
    y_max_limit = y_max_limit * 1.2
    
    for mu_idx in range(n_mus):
        if len(isi_per_mu[mu_idx]) > 0:
            fig.update_yaxes(range=[0, y_max_limit], row=mu_idx+1, col=1)
    
    # ===== 8. LAYOUT =====
    fig.update_layout(
        autosize=True,
        height=450 * n_mus,
        template="plotly_white",
        margin=dict(l=80, r=60, t=100, b=80),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='lightgray',
            borderwidth=1
        ),
        title=dict(
            text=f"⏱️ Distribution ISI avec ajustement Weibull (Online) - {n_mus} unités motrices",
            x=0.5,
            xanchor='center',
            font=dict(size=16, family='Arial', weight='bold')
        )
    )
    
    fig.update_xaxes(title_text="Inter-spike interval (milliseconds)", row=n_mus, col=1)
    fig.update_yaxes(title_text="Probability density (ms⁻¹)")
    
    # ===== 9. GUARDAR =====
    if save:
        save_path = os.path.join(exp_path, "weibull_fit_online.html")
        fig.write_html(save_path, include_plotlyjs='cdn', full_html=True,
                      div_id="fig_weibull_online", config={'responsive': True, 'displayModeBar': True})
        print(f"\n✅ Weibull figure saved: {save_path}")
    
    return fig