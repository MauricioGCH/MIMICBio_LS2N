## leer funciones
import os
import numpy as np
import plotly.graph_objects as go
from Functions.Metrics import weibull_discrete_pmf
import scipy
def save_offline_figure(offline_results, signal_filtered, config, exp_path, save = True, spike_idx_true=None,OFFSET_peaks = None):
    """
    Crea y guarda figuras interactivas para resultados OFFLINE:
    1. Detección de spikes (señal + umbrales + spikes)
    2. MUAPs (todos los waveforms individuales + promedio) - MULTIPLE MUAPs
    3. Histograma ISI (Intervalos entre spikes) - MULTIPLE MUAPs
    
    Parameters:
    -----------
    offline_results : dict
    signal_filtered : array
    config : dict
    exp_path : str
    save : bool
    spike_idx_true : list of arrays, opcional
        Ground truth spike indices por fuente (en muestras de la señal completa)
    """
    
    fs = config["sampling_rate"]
    init_data = config["init_data"]
    
    # ==================================================
    # 🔹 DATOS PREPARACIÓN
    # ==================================================
    signal_window = signal_filtered[:init_data]
    #SNR = 20*np.log10(np.std(Y)/residual_std)
    L = len(signal_window)
    time = np.linspace(0, L / fs, L, endpoint=False)
    
    spike_idx = offline_results["spike_idx"]
    spike_times = spike_idx / fs
    
    sigma = offline_results["sigma"]
    threshold = config["threshold_sigma"] * sigma
    
    waveforms_list = offline_results["waveforms_per_mu"]
    mean_waveforms_list = offline_results["mean_waveforms_per_mu"]
    isi_list = offline_results["isi_per_mu"]
    
    n_mus = len(waveforms_list)
    
    assert len(mean_waveforms_list) == n_mus, "Inconsistent number of MUAPs"

    if not len(isi_list) == n_mus:
        print("For " + str(n_mus) + " only "+ str(isi_list) + " groups of isi where found")
    
    muap_time_ms = np.linspace(
        -config["window_pre_ms"], 
        config["window_post_ms"], 
        len(mean_waveforms_list[0])
    )
    
    colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # ==================================================
    # 🔹 FIGURA 1: DETECCIÓN DE SPIKES
    # ==================================================
    fig_spikes = go.Figure()
    
    fig_spikes.add_trace(go.Scatter(
        x=time,
        y=signal_window,
        mode='lines',
        name='Signal filtré',
        line=dict(width=1.5, color='#1f77b4'),
        opacity=0.8
    ))
    
    fig_spikes.add_trace(go.Scatter(
        x=time,
        y=[threshold] * len(time),
        mode='lines',
        name=f'+Threshold ({threshold:.2f} µV)',
        line=dict(width=1.5, dash='dash', color='#d62728'),
        opacity=0.7
    ))
    
    fig_spikes.add_trace(go.Scatter(
        x=time,
        y=[-threshold] * len(time),
        mode='lines',
        name=f'-Threshold ({-threshold:.2f} µV)',
        line=dict(width=1.5, dash='dash', color='#d62728'),
        opacity=0.7
    ))
    
    # Spikes detectados — X negra
    fig_spikes.add_trace(go.Scatter(
        x=spike_times,
        y=signal_window[spike_idx],
        mode='markers',
        name=f'Spikes détectés (n={len(spike_idx)})',
        marker=dict(
            symbol='x',
            size=10,
            color='black',
            line=dict(width=2)
        ),
        hovertemplate='<b>Spike détecté</b><br>Time: %{x:.3f} s<br>Amplitude: %{y:.2f} µV<extra></extra>'
    ))

    # ==================================================
    # 🔹 GROUND TRUTH SPIKES — X por fuente (solo si se proporcionan)
    # ==================================================
    if spike_idx_true is not None:
        colors_gt = ['#2ca02c', '#17becf', '#9467bd', '#8c564b']

        for src_idx, gt_idx in enumerate(spike_idx_true):
            # Filtrar al primer minuto (init_data)
            gt_idx_window = gt_idx[gt_idx < init_data]
             

            if len(gt_idx_window) == 0:
                continue
            
            gt_times = (gt_idx_window+OFFSET_peaks[src_idx]) / fs
            gt_amplitudes = signal_window[(gt_idx_window+OFFSET_peaks[src_idx])]

            fig_spikes.add_trace(go.Scatter(
                x=gt_times,
                y=gt_amplitudes,
                mode='markers',
                name=f'GT Source {src_idx + 1} (n={len(gt_idx_window)})',
                marker=dict(
                    symbol='x',
                    size=10,
                    color=colors_gt[src_idx % len(colors_gt)],
                    line=dict(width=2)
                ),
                hovertemplate=(
                    f'<b>GT Source {src_idx + 1}</b><br>'
                    'Time: %{x:.3f} s<br>'
                    'Amplitude: %{y:.2f} µV<extra></extra>'
                )
            ))
    
    fig_spikes.update_layout(
        autosize=True,
        height=600,
        template="plotly_white",
        margin=dict(l=80, r=60, t=100, b=80),
        hovermode='x unified',
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
            text=f"🔍 Détection de spikes - Offline (n={len(spike_idx)} spikes détectés)",
            x=0.5,
            xanchor='center',
            font=dict(size=16, family='Arial', weight='bold')
        ),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    fig_spikes.update_xaxes(
        title_text="Time (seconds)",
        showgrid=True,
        gridwidth=0.5,
        gridcolor='#e5e5e5',
        showline=True,
        linewidth=1,
        linecolor='black',
        mirror=True,
        zeroline=True,
        zerolinecolor='gray',
        zerolinewidth=0.5
    )
    
    fig_spikes.update_yaxes(
        title_text="Amplitude (µV)",
        showgrid=True,
        gridwidth=0.5,
        gridcolor='#e5e5e5',
        showline=True,
        linewidth=1,
        linecolor='black',
        mirror=True,
        zeroline=True,
        zerolinecolor='gray',
        zerolinewidth=0.5
    )
    
    stats_text = (
        f"<b>Statistiques:</b><br>"
        f"σ (bruit): {sigma:.2f} µV<br>"
        f"Seuil: {threshold:.2f} µV (±{config['threshold_sigma']}σ)<br>"
        f"Spikes détectés: {len(spike_idx)}<br>"
        f"Taux détection: {len(spike_idx)/(L/fs):.1f} spikes/s<br>"
        f"Unités motrices: {n_mus}"
    )
    
    fig_spikes.add_annotation(
        text=stats_text,
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        showarrow=False,
        font=dict(size=11, family='monospace'),
        bgcolor='rgba(255, 255, 255, 0.9)',
        bordercolor='gray',
        borderwidth=1,
        borderpad=8,
        align='left'
    )
    
    # ==================================================
    # 🔹 FIGURA 2: MUAPs MÚLTIPLES
    # ==================================================
    fig_muaps = go.Figure()
    
    for mu_idx in range(n_mus):
        waveforms = waveforms_list[mu_idx]
        mean_waveform = mean_waveforms_list[mu_idx]
        color = colors[mu_idx % len(colors)]
        
        max_waveforms = min(len(waveforms), 300)
        step = max(1, len(waveforms) // max_waveforms)
        
        for idx, wf in enumerate(waveforms[::step]):
            fig_muaps.add_trace(go.Scatter(
                x=muap_time_ms,
                y=wf,
                mode='lines',
                name=f'MU{mu_idx+1} - spike individuel',
                line=dict(width=0.8, color=color),
                opacity=0.1,
                showlegend=False,
                hoverinfo='skip',
                legendgroup=f'mu{mu_idx}'
            ))
        
        fig_muaps.add_trace(go.Scatter(
            x=muap_time_ms,
            y=mean_waveform,
            mode='lines',
            name=f'MU{mu_idx+1} - MUAP moyen (n={len(waveforms)} spikes)',
            line=dict(width=3, color=color),
            hovertemplate=f'<b>MU{mu_idx+1} - MUAP moyen</b><br>Time: %%{{x:.1f}} ms<br>Amplitude: %%{{y:.2f}} µV<extra></extra>',
            legendgroup=f'mu{mu_idx}'
        ))
        
        if len(waveforms) > 1:
            std_waveform = np.std(waveforms, axis=0)
            x_fill = np.concatenate([muap_time_ms, muap_time_ms[::-1]])
            y_fill = np.concatenate([mean_waveform + std_waveform, 
                                     (mean_waveform - std_waveform)[::-1]])
            
            fig_muaps.add_trace(go.Scatter(
                x=x_fill,
                y=y_fill,
                fill='toself',
                fillcolor=f'rgba({int(color[1:3],16)}, {int(color[3:5],16)}, {int(color[5:7],16)}, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name=f'MU{mu_idx+1} - ±1σ',
                showlegend=True,
                hoverinfo='skip',
                legendgroup=f'mu{mu_idx}'
            ))
    
    title_text = f"📊 MUAPs estimés - Offline ({n_mus} unités motrices"
    if n_mus == 2:
        title_text += " positive/negative"
    title_text += f", basé sur {sum(len(w) for w in waveforms_list)} spikes au total)"
    
    fig_muaps.update_layout(
        autosize=True,
        height=600,
        template="plotly_white",
        margin=dict(l=80, r=60, t=100, b=80),
        hovermode='closest',
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
            text=title_text,
            x=0.5,
            xanchor='center',
            font=dict(size=16, family='Arial', weight='bold')
        ),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    fig_muaps.update_xaxes(
        title_text="Time (ms)",
        showgrid=True,
        gridwidth=0.5,
        gridcolor='#e5e5e5',
        showline=True,
        linewidth=1,
        linecolor='black',
        mirror=True,
        zeroline=True,
        zerolinecolor='gray',
        zerolinewidth=0.5
    )
    
    fig_muaps.update_yaxes(
        title_text="Amplitude (µV)",
        showgrid=True,
        gridwidth=0.5,
        gridcolor='#e5e5e5',
        showline=True,
        linewidth=1,
        linecolor='black',
        mirror=True,
        zeroline=True,
        zerolinecolor='gray',
        zerolinewidth=0.5
    )
    
    quality_text = "<b>Qualité MUAPs:</b><br>"
    for mu_idx in range(min(n_mus, 4)):
        n_spikes = len(waveforms_list[mu_idx])
        snr = 20 * np.log10(np.max(np.abs(mean_waveforms_list[mu_idx])) / sigma) if sigma > 0 else 0
        quality_text += f"MU{mu_idx+1}: {n_spikes} spikes, SNR={snr:.1f} dB<br>"
    
    if n_mus > 4:
        quality_text += f"... y {n_mus-4} MU más"
    
    fig_muaps.add_annotation(
        text=quality_text,
        xref="paper", yref="paper",
        x=0.98, y=0.98,
        showarrow=False,
        font=dict(size=11, family='monospace'),
        bgcolor='rgba(255, 255, 255, 0.9)',
        bordercolor='gray',
        borderwidth=1,
        borderpad=8,
        align='left'
    )
    
    # ==================================================
    # 🔹 FIGURA 3: HISTOGRAMAS ISI MÚLTIPLES + WEIBULL TEÓRICO
    # ==================================================
    from plotly.subplots import make_subplots
    
    t0_init_list = offline_results["t0_init"]
    beta_init_list = offline_results["beta_init"]
    t_R_list = config["t_R"]
    fs = config["sampling_rate"]
    
    fig_isi = make_subplots(
        rows=n_mus, cols=1,
        subplot_titles=[f"MU{mu_idx+1} - Distribution ISI (n={len(isi_list[mu_idx])} intervalles)" 
                       for mu_idx in range(n_mus)],
        shared_xaxes=False,
        vertical_spacing=0.08
    )
    
    for mu_idx in range(n_mus):
        isi = isi_list[mu_idx]
        color = colors[mu_idx % len(colors)]
        
        if len(isi) > 0:
            isi_ms = isi * 1000
            
            fig_isi.add_trace(
                go.Histogram(
                    x=isi_ms,
                    nbinsx=50,
                    histnorm='probability density',
                    name=f'MU{mu_idx+1} - Experimental',
                    marker=dict(color=color, line=dict(color='black', width=0.5)),
                    opacity=0.7,
                    showlegend=(mu_idx == 0)
                ),
                row=mu_idx+1, col=1
            )
            
            t_R_ms = (t_R_list[mu_idx] / fs) * 1000
            t0_ms = (t0_init_list[mu_idx] / fs) * 1000
            
            max_isi_ms = np.max(isi_ms) if len(isi_ms) > 0 else 500
            t_ms = np.linspace(t_R_ms, max_isi_ms, 500)
            
            t_samples = t_ms * fs / 1000
            
            pdf_samples = weibull_discrete_pmf(t_samples, t0_init_list[mu_idx], 
                                               beta_init_list[mu_idx], t_R_list[mu_idx])
            
            pdf_per_ms = pdf_samples * fs / 1000
            
            fig_isi.add_trace(
                go.Scatter(
                    x=t_ms,
                    y=pdf_per_ms,
                    mode='lines',
                    name=f'MU{mu_idx+1} - Weibull théorique',
                    line=dict(width=2.5, color=color, dash='dash'),
                    showlegend=(mu_idx == 0)
                ),
                row=mu_idx+1, col=1
            )
            
            fig_isi.add_vline(
                x=t0_ms,
                line_dash="dot",
                line_color="green",
                line_width=2.5,
                annotation_text=f"t₀ = {t0_ms:.1f}ms",
                annotation_position="top",
                row=mu_idx+1, col=1
            )

            weibull_stats = (
                f"<b>Paramètres Weibull:</b><br>"
                f"t₀ = {t0_ms:.1f} ms<br>"
                f"β = {beta_init_list[mu_idx]:.3f}<br>"
                f"t_R = {t_R_ms:.1f} ms"
            )
            
            x_max_data = np.max(isi_ms) if len(isi_ms) > 0 else 500
            y_max_weibull = np.max(pdf_per_ms) if len(pdf_per_ms) > 0 else 0.1
            #print(f"MU{mu_idx+1}: x_max={x_max_data:.1f}, y_max={y_max_weibull:.5f}")
            
            fig_isi.add_annotation(
                text=weibull_stats,
                x=x_max_data * 0.70,
                y=y_max_weibull * 0.85,
                xanchor='left',
                yanchor='top',
                showarrow=False,
                font=dict(size=9, family='monospace', color=color),
                bgcolor='rgba(255, 255, 255, 0.95)',
                bordercolor=color,
                borderwidth=1.5,
                borderpad=6,
                align='left',
                row=mu_idx+1,
                col=1
            )
    
    y_max_limit = 0
    
    for mu_idx in range(n_mus):
        if len(isi_list[mu_idx]) > 0:
            isi_ms = isi_list[mu_idx] * 1000
            hist, _ = np.histogram(isi_ms, bins=50, density=True)
            y_max_limit = max(y_max_limit, np.max(hist))
            
            t_R_ms = (t_R_list[mu_idx] / fs) * 1000
            t0_ms = (t0_init_list[mu_idx] / fs) * 1000
            max_isi_ms = np.max(isi_ms)
            t_ms = np.linspace(t_R_ms, max_isi_ms, 500)
            t_samples = t_ms * fs / 1000
            pdf_samples = weibull_discrete_pmf(t_samples, t0_init_list[mu_idx], 
                                               beta_init_list[mu_idx], t_R_list[mu_idx])
            pdf_per_ms = pdf_samples * fs / 1000
            y_max_limit = max(y_max_limit, np.max(pdf_per_ms))
    
    y_max_limit = y_max_limit * 1.2
    
    for mu_idx in range(n_mus):
        fig_isi.update_yaxes(range=[0, y_max_limit], row=mu_idx+1, col=1)
    
    fig_isi.update_layout(
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
            text=f"⏱️ Distribution ISI avec ajustement Weibull - {n_mus} sources",
            x=0.5,
            xanchor='center',
            font=dict(size=16, family='Arial', weight='bold')
        )
    )
    
    fig_isi.update_xaxes(title_text="Inter-spike interval (milliseconds)", row=n_mus, col=1)
    fig_isi.update_yaxes(title_text="Probability density (ms⁻¹)")   
    
    # ==================================================
    # 🔹 GUARDAR FIGURAS
    # ==================================================
    save_path_spikes = None
    save_path_muaps = None
    save_path_isi = None
    
    if save:
        save_path_spikes = os.path.join(exp_path, "offline_spike_detection.html")
        fig_spikes.write_html(save_path_spikes, include_plotlyjs='cdn', full_html=True,
                             div_id="fig_offline_spikes", config={'responsive': True, 'displayModeBar': True})
        
        save_path_muaps = os.path.join(exp_path, "offline_muaps.html")
        fig_muaps.write_html(save_path_muaps, include_plotlyjs='cdn', full_html=True,
                            div_id="fig_offline_muaps", config={'responsive': True, 'displayModeBar': True})
        
        if len(isi_list[0]) > 0:
            save_path_isi = os.path.join(exp_path, "offline_isi_distribution.html")
            fig_isi.write_html(save_path_isi, include_plotlyjs='cdn', full_html=True,
                              div_id="fig_offline_isi", config={'responsive': True, 'displayModeBar': True})
    
    total_spikes = sum(len(w) for w in waveforms_list)
    print(f"\n{'='*60}")
    print(f"✅ Offline Analysis Complete - {n_mus} Motor Units")
    print(f"{'='*60}")
    for mu_idx in range(n_mus):
        print(f"   MU{mu_idx+1}: {len(waveforms_list[mu_idx])} spikes, "
              f"mean ISI: {np.mean(isi_list[mu_idx]) if len(isi_list[mu_idx])>0 else 0:.3f}s")
    print(f"   TOTAL: {total_spikes} spikes from {n_mus} MUs")
    if save:
        print(f"📁 Spike detection:  {save_path_spikes}")
        print(f"📁 MUAP analysis:    {save_path_muaps}")
        if len(isi_list[0]) > 0:
            print(f"📁 ISI distribution: {save_path_isi}")
    print(f"{'='*60}\n")
    
    return fig_spikes, fig_muaps, fig_isi if len(isi_list[0]) > 0 else None