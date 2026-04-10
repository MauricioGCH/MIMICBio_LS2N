
import os
import numpy as np
import os
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

def extract_waveforms_online(signal, U_est, spike_idx_by_MU, fs, config):
    """
    Extrae waveforms individuales para cada MU usando los spikes detectados por el algoritmo online.
    
    Parameters:
    -----------
    signal : array
        Señal original (filtrada) completa
    U_est : list of arrays
        Lista de vectores de disparo para cada MU (cada uno con 0/1)
    spike_idx_by_MU : list of arrays
        Índices de los spikes para cada MU (opcional, se puede calcular de U_est)
    fs : int
        Frecuencia de muestreo
    config : dict
        Configuración con window_pre_ms y window_post_ms
    
    Returns:
    --------
    waveforms_by_MU : list of arrays
        Lista donde cada elemento es un array de waveforms (n_spikes × longitud_waveform)
    """
    
    pre = int(config["window_pre_ms"] * fs / 1000)
    post = int(config["window_post_ms"] * fs / 1000)
    
    waveforms_by_MU = []
    
    for mu_idx, u_vector in enumerate(U_est):
        # Obtener índices de spikes para esta MU
        breakpoint()
        spike_idx = np.where(u_vector == 1)[0]
        
        waveforms = []
        for t in spike_idx:
            if t - pre >= 0 and t + post < len(signal):
                snippet = signal[t - pre : t + post]
                waveforms.append(snippet)
        
        if len(waveforms) > 0:
            waveforms_by_MU.append(np.array(waveforms))
        else:
            waveforms_by_MU.append(None)
    
    return waveforms_by_MU


def save_spike_online_figure(online_results, signal_filtered, config, exp_path, save = False):
    """
    Crea y guarda una figura interactiva con:
    1. Señal vs reconstrucción
    2. Spikes (eventplot estilo raster)
    3. MUAPs estimados (con todos los waveforms individuales extraídos de la señal online)
    
    Parameters:
    -----------
    online_results : dict
        Resultados del algoritmo online con Y, Y_est, U_est, H_est, Theta_est
    signal_filtered : array
        Señal filtrada completa (para extraer waveforms individuales)
    config : dict
        Configuración del experimento
    exp_path : str
        Ruta donde guardar las figuras
    """
    
    Y = online_results["Y"] if "Y" in online_results else None
    Y_est = online_results["Y_est"]
    U_est = online_results["U_est"]
    H_est = online_results["H_est"]  # MUAPs promedio
    Theta_est = online_results["Theta_est"]

    n_MU = config["n_sources"]

    L = len(Y)
    time = np.linspace(
        start=0,
        stop=L / config["sampling_rate"],
        num=L,
        endpoint=False
    )
    
    # ==================================================
    # 🔹 EXTRAER WAVEFORMS INDIVIDUALES DEL ONLINE
    # ==================================================
    print("📊 Extrayendo waveforms individuales de los spikes online...")
    online_waveforms = extract_waveforms_online(
        signal=signal_filtered,
        U_est=U_est,
        spike_idx_by_MU=None,  # Se calcula automáticamente de U_est
        fs=config["sampling_rate"],
        config=config
    )
    breakpoint()
    # Estadísticas de cuántos spikes tiene cada MU
    for mu_idx, waveforms in enumerate(online_waveforms):
        if waveforms is not None:
            print(f"   MU {mu_idx+1}: {len(waveforms)} spikes individuales")
        else:
            print(f"   MU {mu_idx+1}: 0 spikes")
    
    # --- figura con subplots ---
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=(
            "Signal simulé vs reconstruit",
            "Trains d'impulsions estimés (U)"
        )
    )

    # ==================================================
    # 🔹 1. SIGNAL
    # ==================================================
    if Y is not None:
        if len(Y) > 10000:
            step = max(1, len(Y) // 10000)
            time_plot = time[::step]
            Y_plot = Y[::step]
            Y_est_plot = Y_est[::step]
        else:
            time_plot = time
            Y_plot = Y
            Y_est_plot = Y_est
            
        fig.add_trace(go.Scatter(
            x=time_plot,
            y=Y_plot,
            mode='lines',
            name='Signal simulé',
            line=dict(width=1.5, color='#1f77b4'),
            opacity=0.7,
            showlegend=True
        ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=time_plot if Y is not None else time,
        y=Y_est_plot if Y is not None else Y_est,
        mode='lines',
        name='Signal reconstruit',
        line=dict(width=2, dash='dash', color='#ff7f0e')
    ), row=1, col=1)

    # ==================================================
    # 🔹 2. SPIKES (RASTER PLOT)
    # ==================================================
    colors = px.colors.qualitative.Plotly
    
    for i in range(n_MU):
        spike_times = np.where(U_est[i] == 1)[0]
        spike_times_sec = spike_times / config["sampling_rate"]
        
        # Para raster plot, usar y = i (sin jitter para mejor claridad)
        y_pos = np.ones_like(spike_times_sec) * i

        if len(spike_times) > 0:
            fig.add_trace(go.Scatter(
                x=spike_times_sec,
                y=y_pos,
                mode='markers',
                name=f"MU {i+1} ({len(spike_times)} spikes)",
                marker=dict(
                    symbol='line-ns-open',
                    size=12,
                    color=colors[i % len(colors)],
                    line=dict(width=1)
                ),
                opacity=0.8,
                showlegend=n_MU <= 8,
                hovertemplate=f'<b>MU {i+1}</b><br>Time: %%{{x:.3f}} s<br>Spike index: %%{{customdata}}<extra></extra>',
                customdata=spike_times
            ), row=2, col=1)

    # ==================================================
    # 🔹 3. MUAPs FIGURE - CON TODOS LOS WAVEFORMS INDIVIDUALES DEL ONLINE
    # ==================================================
    fig_MUAPs = go.Figure()
    
    # Eje temporal para MUAPs (en ms)
    pre_ms = config["window_pre_ms"]
    post_ms = config["window_post_ms"]
    muap_time_ms = np.linspace(-pre_ms, post_ms, len(H_est[0]) if len(H_est) > 0 else 0)
    
    for i in range(n_MU):
        # 🔥 PRIMERO: Añadir TODOS los waveforms individuales en GRIS (extraídos del online)
        if i < len(online_waveforms) and online_waveforms[i] is not None:
            waveforms_i = online_waveforms[i]
            n_spikes = len(waveforms_i)
            
            # Limitar para rendimiento (mostrar máximo 200 spikes por MU)
            max_show = min(n_spikes, 200)
            step = max(1, n_spikes // max_show)
            
            # Normalizar y añadir cada waveform individual
            for idx, w in enumerate(waveforms_i[::step]):
                # Normalizar respecto al máximo del waveform individual
                w_norm = w / np.max(np.abs(w)) if np.max(np.abs(w)) > 0 else w
                
                fig_MUAPs.add_trace(go.Scatter(
                    x=muap_time_ms,
                    y=w_norm,
                    mode='lines',
                    line=dict(width=0.5, color='lightgray'),
                    opacity=0.15,
                    showlegend=False,
                    hoverinfo='skip',
                    legendgroup=f"MU_{i+1}_individual"
                ))
        
        # 🔥 SEGUNDO: Añadir el MUAP promedio (color vivo y grueso)
        muap_normalized = H_est[i] / np.max(np.abs(H_est[i])) if np.max(np.abs(H_est[i])) > 0 else H_est[i]
        
        n_spikes_total = len(online_waveforms[i]) if (i < len(online_waveforms) and online_waveforms[i] is not None) else 0
        
        fig_MUAPs.add_trace(go.Scatter(
            x=muap_time_ms,
            y=muap_normalized,
            mode='lines',
            name=f'MU {i+1} (promedio, n={n_spikes_total} spikes)',
            line=dict(width=3.5, color=colors[i % len(colors)]),
            showlegend=True,
            legendgroup=f"MU_{i+1}_avg",
            hovertemplate=f'<b>MU {i+1} Promedio</b><br>Time: %%{{x:.1f}} ms<br>Amplitude: %%{{y:.3f}}<extra></extra>'
        ))
        
        # 🔥 TERCERO: Añadir banda de desviación estándar (para mostrar variabilidad)
        if i < len(online_waveforms) and online_waveforms[i] is not None and len(online_waveforms[i]) > 1:
            waveforms_i = online_waveforms[i]
            std_waveform = np.std(waveforms_i, axis=0)
            
            # Normalizar la desviación estándar con el mismo factor que el promedio
            max_abs_mean = np.max(np.abs(H_est[i])) if np.max(np.abs(H_est[i])) > 0 else 1
            std_normalized = std_waveform / max_abs_mean
            
            # Crear banda de ±1σ
            x_fill = np.concatenate([muap_time_ms, muap_time_ms[::-1]])
            y_fill = np.concatenate([muap_normalized + std_normalized, 
                                     (muap_normalized - std_normalized)[::-1]])
            
            # Color de la banda (mismo color que el promedio pero con opacidad)
            color_rgb = colors[i % len(colors)].lstrip('#')
            r = int(color_rgb[0:2], 16)
            g = int(color_rgb[2:4], 16)
            b = int(color_rgb[4:6], 16)
            
            fig_MUAPs.add_trace(go.Scatter(
                x=x_fill,
                y=y_fill,
                fill='toself',
                fillcolor=f'rgba({r}, {g}, {b}, 0.15)',
                line=dict(color='rgba(0,0,0,0)'),
                name=f'MU {i+1} ±1σ (variabilidad)',
                showlegend=False,
                legendgroup=f"MU_{i+1}_std",
                hoverinfo='skip'
            ))

    # ==================================================
    # 🔹 LAYOUT PARA MUAPs
    # ==================================================
    fig_MUAPs.update_layout(
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
            text=f"📊 MUAPs Online - Spikes individuales (gris) + Promedio (color)",
            x=0.5,
            xanchor='center',
            font=dict(size=16, family='Arial', weight='bold')
        ),
        
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    # Configurar ejes de MUAPs
    fig_MUAPs.update_xaxes(
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
    
    fig_MUAPs.update_yaxes(
        title_text="Normalized Amplitude",
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
    
    # Añadir estadísticas globales como anotación
    total_spikes = sum(len(w) if w is not None else 0 for w in online_waveforms)
    stats_text = (
        f"<b>Estadísticas Online:</b><br>"
        f"Total MUs: {n_MU}<br>"
        f"Total spikes: {total_spikes}<br>"
        f"Tasa disparo media: {total_spikes/(L/config['sampling_rate']):.1f} Hz"
    )
    
    fig_MUAPs.add_annotation(
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
    # 🔹 LAYOUT PARA FIGURA PRINCIPAL
    # ==================================================
    fig.update_layout(
        autosize=True,
        height=700,
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
            text="📈 Résultats Algorithm 2 - Décomposition Online",
            x=0.5,
            xanchor='center',
            font=dict(size=16, family='Arial', weight='bold')
        ),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    # Configurar ejes
    fig.update_xaxes(
        title_text="Temps (s)", 
        row=1, col=1,
        showgrid=True,
        gridwidth=0.5,
        gridcolor='#e5e5e5',
        showline=True,
        linewidth=1,
        linecolor='black',
        mirror=True,
        zeroline=True,
        zerolinecolor='gray',
        zerolinewidth=1
    )
    fig.update_yaxes(
        title_text="Amplitude (mV)", 
        row=1, col=1,
        showgrid=True,
        gridwidth=0.5,
        gridcolor='#e5e5e5',
        showline=True,
        linewidth=1,
        linecolor='black',
        mirror=True
    )

    fig.update_xaxes(
        title_text="Temps (s)", 
        row=2, col=1,
        showgrid=True,
        gridwidth=0.5,
        gridcolor='#e5e5e5',
        showline=True,
        linewidth=1,
        linecolor='black',
        mirror=True
    )
    fig.update_yaxes(
        title_text="Motor Unit Index", 
        row=2, col=1,
        showgrid=True,
        gridwidth=0.5,
        gridcolor='#e5e5e5',
        showline=True,
        linewidth=1,
        linecolor='black',
        mirror=True,
        tickmode='linear',
        tick0=0,
        dtick=max(1, n_MU // 10)
    )

    # Actualizar títulos de subplots
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=14, family='Arial', weight='bold')

    # ==================================================
    # 🔹 GUARDAR FIGURAS
    # ==================================================
    save_path = None
    save_path_muaps = None
    if save:
        save_path = os.path.join(exp_path, "online_interactive.html")
        save_path_muaps = os.path.join(exp_path, "online_muaps_analysis.html")
        
        fig.write_html(
            save_path,
            include_plotlyjs='cdn',
            full_html=True,
            div_id="fig_online",
            config={
                'responsive': True,
                'displayModeBar': True,
                'displaylogo': False,
                'scrollZoom': True
            }
        )
        
        fig_MUAPs.write_html(
            save_path_muaps,
            include_plotlyjs='cdn',
            full_html=True,
            div_id="fig_online_muaps",
            config={
                'responsive': True,
                'displayModeBar': True,
                'displaylogo': False,
                'scrollZoom': True
            }
        )

    print(f"\n{'='*60}")
    print(f"✅ Online Analysis Complete")
    print(f"{'='*60}")
    if save:
        print(f"📁 Figure interactive: {save_path}")
        print(f"📁 MUAPs figure: {save_path_muaps}")
    print(f"\n📊 Online Statistics:")
    print(f"   • MUs detectées: {n_MU}")
    print(f"   • Total spikes: {total_spikes}")
    print(f"   • Taux: {total_spikes/(L/config['sampling_rate']):.1f} Hz")
    for i, w in enumerate(online_waveforms):
        n_spikes = len(w) if w is not None else 0
        print(f"   • MU {i+1}: {n_spikes} spikes")
    print(f"{'='*60}\n")
    
    return fig, fig_MUAPs