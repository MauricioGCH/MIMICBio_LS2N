## leer funciones
import os
import numpy as np
from matplotlib import pyplot as plt
import os
import plotly.graph_objects as go

def save_offline_figure(offline_results, signal_filtered, config, exp_path, save = False):
    """
    Crea y guarda figuras interactivas para resultados OFFLINE:
    1. Detección de spikes (señal + umbrales + spikes)
    2. MUAPs (todos los waveforms individuales + promedio)
    3. Histograma ISI (Intervalos entre spikes)
    
    Returns:
    --------
    tuple: (fig_spikes, fig_muaps, fig_isi) - Figuras de Plotly
    """
    
    fs = config["sampling_rate"]
    init_data = config["init_data"]
    
    # ==================================================
    # 🔹 DATOS PREPARACIÓN
    # ==================================================
    signal_window = signal_filtered[:init_data]
    L = len(signal_window)
    time = np.linspace(0, L / fs, L, endpoint=False)
    
    spike_idx = offline_results["spike_idx"]
    spike_times = spike_idx / fs
    
    sigma = offline_results["sigma"]
    threshold = config["threshold_sigma"] * sigma
    
    waveforms = offline_results["waveforms_per_mu"]
    mean_waveform = offline_results["mean_waveforms_per_mu"]
    
    # ISI (intervalos entre spikes)
    isi = offline_results["isi_per_mu"]
    
    # Eje temporal para MUAPs (en ms)
    muap_time_ms = np.linspace(
        -config["window_pre_ms"], 
        config["window_post_ms"], 
        len(mean_waveform)
    )
    
    # ==================================================
    # 🔹 FIGURA 1: DETECCIÓN DE SPIKES
    # ==================================================
    fig_spikes = go.Figure()
    
    # Señal filtrada
    fig_spikes.add_trace(go.Scatter(
        x=time,
        y=signal_window,
        mode='lines',
        name='Signal filtré',
        line=dict(width=1.5, color='#1f77b4'),
        opacity=0.8
    ))
    
    # Umbral positivo
    fig_spikes.add_trace(go.Scatter(
        x=time,
        y=[threshold] * len(time),
        mode='lines',
        name=f'+Threshold ({threshold:.2f} µV)',
        line=dict(width=1.5, dash='dash', color='#d62728'),
        opacity=0.7
    ))
    
    # Umbral negativo
    fig_spikes.add_trace(go.Scatter(
        x=time,
        y=[-threshold] * len(time),
        mode='lines',
        name=f'-Threshold ({-threshold:.2f} µV)',
        line=dict(width=1.5, dash='dash', color='#d62728'),
        opacity=0.7
    ))
    
    # Spikes detectados (como puntos 'x')
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
        hovertemplate='<b>Spike</b><br>Time: %{x:.3f} s<br>Amplitude: %{y:.2f} µV<extra></extra>'
    ))
    
    # Layout para figura de spikes
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
    
    # Configurar ejes
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
    
    # Añadir estadísticas como anotación
    stats_text = (
        f"<b>Statistiques:</b><br>"
        f"σ (bruit): {sigma:.2f} µV<br>"
        f"Seuil: {threshold:.2f} µV (±{config['threshold_sigma']}σ)<br>"
        f"Spikes détectés: {len(spike_idx)}<br>"
        f"Taux détection: {len(spike_idx)/(L/fs):.1f} spikes/s"
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
    # 🔹 FIGURA 2: MUAPs (TODOS LOS WAVEFORMS + PROMEDIO)
    # ==================================================
    fig_muaps = go.Figure()
    
    # Limitar número de waveforms para rendimiento (mostrar máximo 300)
    max_waveforms = min(len(waveforms), 300)
    step = max(1, len(waveforms) // max_waveforms)
    
    # 1. Añadir todos los waveforms individuales en gris con opacidad
    for idx, wf in enumerate(waveforms[::step]):
        fig_muaps.add_trace(go.Scatter(
            x=muap_time_ms,
            y=wf,
            mode='lines',
            name=f'Spike individuel',
            line=dict(width=0.5, color='lightgray'),
            opacity=0.2,
            showlegend=False,
            hoverinfo='skip',
            legendgroup='individual'
        ))
    
    # 2. Añadir el waveform promedio (color vivo)
    fig_muaps.add_trace(go.Scatter(
        x=muap_time_ms,
        y=mean_waveform,
        mode='lines',
        name=f'MUAP moyen (n={len(waveforms)} spikes)',
        line=dict(width=3, color='#ff7f0e'),
        hovertemplate='<b>MUAP moyen</b><br>Time: %{x:.1f} ms<br>Amplitude: %{y:.2f} µV<extra></extra>'
    ))
    
    # 3. Opcional: Añadir banda de desviación estándar
    if len(waveforms) > 1:
        breakpoint()
        std_waveform = np.std(waveforms, axis=0)
        x_fill = np.concatenate([muap_time_ms, muap_time_ms[::-1]])
        y_fill = np.concatenate([mean_waveform + std_waveform, 
                                 (mean_waveform - std_waveform)[::-1]])
        
        fig_muaps.add_trace(go.Scatter(
            x=x_fill,
            y=y_fill,
            fill='toself',
            fillcolor='rgba(255, 127, 14, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='±1σ (variabilité)',
            showlegend=True,
            hoverinfo='skip'
        ))
    
    # Layout para MUAPs
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
            text=f"📊 MUAP estimé - Offline (basé sur {len(waveforms)} spikes)",
            x=0.5,
            xanchor='center',
            font=dict(size=16, family='Arial', weight='bold')
        ),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    # Configurar ejes para MUAPs
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
    
    # Añadir información sobre la calidad
    if len(waveforms) > 0:
        snr = 20 * np.log10(np.max(np.abs(mean_waveform)) / sigma) if sigma > 0 else 0
        quality_text = (
            f"<b>Qualité MUAP:</b><br>"
            f"Spikes utilisés: {len(waveforms)}<br>"
            f"SNR estimé: {snr:.1f} dB<br>"
            f"Variabilité (σ): {np.mean(std_waveform) if len(waveforms)>1 else 0:.2f} µV"
        )
        
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
    # 🔹 FIGURA 3: HISTOGRAMA ISI (Intervalos entre spikes)
    # ==================================================
    fig_isi = go.Figure()
    
    if len(isi) > 0:
        # Histograma de ISI
        fig_isi.add_trace(go.Histogram(
            x=isi,
            nbinsx=50,
            histnorm='probability density',
            name='Distribution ISI',
            marker=dict(color='#1f77b4', line=dict(color='black', width=0.5)),
            opacity=0.7,
            hovertemplate='ISI: %{x:.3f} s<br>Densité: %{y:.3f}<extra></extra>'
        ))
        
        # Línea vertical para media
        mean_isi = np.mean(isi)
        fig_isi.add_vline(
            x=mean_isi, 
            line_dash="solid", 
            line_color="red",
            line_width=2,
            annotation_text=f"Moyenne: {mean_isi:.3f} s",
            annotation_position="top"
        )
        
        # Línea vertical para mediana
        median_isi = np.median(isi)
        fig_isi.add_vline(
            x=median_isi, 
            line_dash="dash", 
            line_color="orange",
            line_width=2,
            annotation_text=f"Médiane: {median_isi:.3f} s",
            annotation_position="bottom"
        )
        
        # Layout para ISI
        fig_isi.update_layout(
            autosize=True,
            height=500,
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
                text=f"⏱️ Distribution des intervalles entre spikes (ISI) - n={len(isi)} intervalles",
                x=0.5,
                xanchor='center',
                font=dict(size=16, family='Arial', weight='bold')
            ),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # Configurar ejes para ISI
        fig_isi.update_xaxes(
            title_text="Inter-spike interval (seconds)",
            showgrid=True,
            gridwidth=0.5,
            gridcolor='#e5e5e5',
            showline=True,
            linewidth=1,
            linecolor='black',
            mirror=True,
            zeroline=True,
            zerolinecolor='gray'
        )
        
        fig_isi.update_yaxes(
            title_text="Probability density",
            showgrid=True,
            gridwidth=0.5,
            gridcolor='#e5e5e5',
            showline=True,
            linewidth=1,
            linecolor='black',
            mirror=True
        )
        
        # Estadísticas de ISI
        isi_stats = (
            f"<b>Statistiques ISI:</b><br>"
            f"Minimum: {np.min(isi):.3f} s<br>"
            f"Maximum: {np.max(isi):.3f} s<br>"
            f"Moyenne: {mean_isi:.3f} s<br>"
            f"Médiane: {median_isi:.3f} s<br>"
            f"Std: {np.std(isi):.3f} s<br>"
            f"CV: {np.std(isi)/mean_isi:.2f}"
        )
        
        fig_isi.add_annotation(
            text=isi_stats,
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
    # 🔹 GUARDAR FIGURAS
    # ==================================================
    save_path_spikes = None
    save_path_muaps = None
    save_path_isi = None
    
    
    
    if save:
        # Guardar detección de spikes
        save_path_spikes = os.path.join(exp_path, "offline_spike_detection.html")
        fig_spikes.write_html(
            save_path_spikes,
            include_plotlyjs='cdn',
            full_html=True,
            div_id="fig_offline_spikes",
            config={
                'responsive': True,
                'displayModeBar': True,
                'displaylogo': False,
                'scrollZoom': True
            }
        )
    
        # Guardar MUAPs
        save_path_muaps = os.path.join(exp_path, "offline_muaps.html")
        fig_muaps.write_html(
            save_path_muaps,
            include_plotlyjs='cdn',
            full_html=True,
            div_id="fig_offline_muaps",
            config={
                'responsive': True,
                'displayModeBar': True,
                'displaylogo': False,
                'scrollZoom': True
            }
        )
    
        # Guardar ISI (si hay datos)
        if len(isi) > 0:
            save_path_isi = os.path.join(exp_path, "offline_isi_distribution.html")
            fig_isi.write_html(
                save_path_isi,
                include_plotlyjs='cdn',
                full_html=True,
                div_id="fig_offline_isi",
                config={
                    'responsive': True,
                    'displayModeBar': True,
                    'displaylogo': False,
                    'scrollZoom': True
                }
            )
    
    # ==================================================
    # 🔹 IMPRIMIR INFO
    # ==================================================
    print(f"\n{'='*60}")
    print(f"✅ Offline Analysis Complete")
    print(f"{'='*60}")
    if save:
        print(f"📁 Spike detection:  {save_path_spikes}")
    
        print(f"📁 MUAP analysis:    {save_path_muaps}")
    if len(isi) > 0:
        print(f"📁 ISI distribution: {save_path_isi}")
    print(f"\n📊 Offline Statistics:")
    print(f"   • Spikes detected: {len(spike_idx)}")
    print(f"   • Waveforms extracted: {len(waveforms)}")
    print(f"   • Mean ISI: {np.mean(isi) if len(isi)>0 else 0:.3f} s")
    print(f"   • Firing rate: {len(spike_idx)/(L/fs):.1f} Hz")
    print(f"{'='*60}\n")
    
    return fig_spikes, fig_muaps, fig_isi if len(isi) > 0 else None