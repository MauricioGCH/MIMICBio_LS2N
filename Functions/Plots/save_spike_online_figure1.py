
import os
import numpy as np
from matplotlib import pyplot as plt
import os
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def save_spike_online_figure(online_results, offline_results, config, exp_path):
    """
    Crea y guarda una figura interactiva con:
    1. Señal vs reconstrucción
    2. Spikes (eventplot estilo raster)
    3. MUAPs estimados (con todos los waveforms individuales en gris)
    
    Parameters:
    -----------
    online_results : dict
        Resultados del algoritmo online con Y, Y_est, U_est, H_est, Theta_est
    offline_results : dict or None
        Resultados del offline con 'waveforms' para cada MU (si está disponible)
    """
    
    Y = online_results["Y"] if "Y" in online_results else None
    Y_est = online_results["Y_est"]
    U_est = online_results["U_est"]
    H_est = online_results["H_est"]  # MUAPs promedio
    Theta_est = online_results["Theta_est"]

    n_MU = len(U_est)

    L = len(Y)
    time = np.linspace(
        start=0,
        stop=L / config["sampling_rate"],
        num=L,
        endpoint=False
    )
    
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
            line=dict(width=1.5),
            opacity=0.7,
            showlegend=True
        ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=time_plot if Y is not None else time,
        y=Y_est_plot if Y is not None else Y_est,
        mode='lines',
        name='Signal reconstruit',
        line=dict(width=2, dash='dash', color='red')
    ), row=1, col=1)

    # ==================================================
    # 🔹 2. SPIKES
    # ==================================================
    colors = px.colors.qualitative.Plotly
    
    for i in range(n_MU):
        spike_times = np.where(U_est[i] == 1)[0]
        spike_times_sec = spike_times / config["sampling_rate"]
        
        y_pos = i + 0.05 * np.random.randn(len(spike_times)) if n_MU > 10 else np.ones_like(spike_times) * i

        if len(spike_times) > 0:
            fig.add_trace(go.Scatter(
                x=spike_times_sec,
                y=y_pos,
                mode='markers',
                name=f"MU {i+1}",
                marker=dict(
                    symbol='line-ns-open',
                    size=32,
                    color=colors[i % len(colors)],
                    line=dict(width=1)
                ),
                opacity=0.8,
                showlegend=n_MU <= 8
            ), row=2, col=1)

    # ==================================================
    # 🔹 3. MUAPs FIGURE - CON TODOS LOS WAVEFORMS INDIVIDUALES
    # ==================================================
    fig_MUAPs = go.Figure()
    
    # Verificar si tenemos waveforms individuales del offline
    has_individual_waveforms = (offline_results is not None and 
                                  'waveforms' in offline_results and 
                                  offline_results['waveforms'] is not None)
    
    for i in range(n_MU):
        # 🔥 PRIMERO: Añadir TODOS los waveforms individuales en GRIS (si están disponibles)
        if has_individual_waveforms and i < len(offline_results['waveforms']):
            waveforms_i = offline_results['waveforms'][i]  # Shape: (n_spikes, waveform_length)
            
            if waveforms_i is not None and len(waveforms_i) > 0:
                # Normalizar cada waveform individual para comparación justa
                for w in waveforms_i:
                    w_normalized = w / np.max(np.abs(w)) if np.max(np.abs(w)) > 0 else w
                    
                    fig_MUAPs.add_trace(go.Scatter(
                        y=w_normalized,
                        mode='lines',
                        name=f'MU {i+1} - spike individual',
                        line=dict(width=0.8, color='lightgray'),
                        opacity=0.3,
                        showlegend=False,  # Ocultar de la leyenda para no saturar
                        legendgroup=f"MU_{i+1}_individual",
                        hoverinfo='none'  # Mejorar rendimiento
                    ))
        
        # 🔥 SEGUNDO: Añadir el MUAP promedio (color vivo y grueso)
        muap_normalized = H_est[i] / np.max(np.abs(H_est[i])) if np.max(np.abs(H_est[i])) > 0 else H_est[i]
        
        fig_MUAPs.add_trace(go.Scatter(
            y=muap_normalized,
            mode='lines',
            name=f'MUAP {i+1} (promedio)',
            line=dict(width=4, color=colors[i % len(colors)]),
            showlegend=True,
            legendgroup=f"MU_{i+1}_avg"
        ))
        
        # 🔥 OPCIONAL: Añadir banda de desviación estándar (sombra)
        if has_individual_waveforms and i < len(offline_results['waveforms']):
            waveforms_i = offline_results['waveforms'][i]
            if waveforms_i is not None and len(waveforms_i) > 1:
                # Calcular desviación estándar
                std_waveform = np.std(waveforms_i, axis=0)
                std_normalized = std_waveform / np.max(np.abs(muap_normalized)) if np.max(np.abs(muap_normalized)) > 0 else std_waveform
                
                # Añadir banda de ±1σ (sombra)
                x_vals = np.arange(len(muap_normalized))
                fig_MUAPs.add_trace(go.Scatter(
                    x=np.concatenate([x_vals, x_vals[::-1]]),
                    y=np.concatenate([muap_normalized + std_normalized, 
                                      (muap_normalized - std_normalized)[::-1]]),
                    fill='toself',
                    fillcolor=f'rgba({i*50 % 255}, {i*100 % 255}, {i*150 % 255}, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name=f'MU {i+1} ±1σ',
                    showlegend=False,
                    legendgroup=f"MU_{i+1}_std"
                ))

    # ==================================================
    # 🔹 LAYOUT PARA MUAPs
    # ==================================================
    fig_MUAPs.update_layout(
        autosize=True,
        height=600,
        template="plotly_white",
        
        margin=dict(l=60, r=40, t=80, b=60),
        
        hovermode='closest',
        
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='black',
            borderwidth=1
        ),
        
        title=dict(
            text="MUAPs Shape - Individual spikes (gray) + Average (color)",
            x=0.5,
            xanchor='center',
            font=dict(size=16, family='Arial', weight='bold')
        ),
        
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    # Configurar ejes de MUAPs
    fig_MUAPs.update_xaxes(
        title_text="Time (samples)",
        showgrid=True,
        gridwidth=0.5,
        gridcolor='lightgray',
        zeroline=True,
        zerolinecolor='gray',
        zerolinewidth=0.5
    )
    
    fig_MUAPs.update_yaxes(
        title_text="Normalized Amplitude",
        showgrid=True,
        gridwidth=0.5,
        gridcolor='lightgray',
        zeroline=True,
        zerolinecolor='gray',
        zerolinewidth=0.5
    )
    
    # ==================================================
    # 🔹 LAYOUT PARA FIGURA PRINCIPAL
    # ==================================================
    fig.update_layout(
        autosize=True,
        height=700,
        template="plotly_white",
        margin=dict(l=60, r=40, t=80, b=60),
        hovermode='x unified',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='black',
            borderwidth=1
        ),
        title=dict(
            text="Résultats Algorithm 2",
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
        gridcolor='lightgray',
        zeroline=True,
        zerolinecolor='gray',
        zerolinewidth=1
    )
    fig.update_yaxes(
        title_text="Amplitude (mV)", 
        row=1, col=1,
        showgrid=True,
        gridwidth=0.5,
        gridcolor='lightgray'
    )

    fig.update_xaxes(
        title_text="Temps (s)", 
        row=2, col=1,
        showgrid=True,
        gridwidth=0.5,
        gridcolor='lightgray'
    )
    fig.update_yaxes(
        title_text="Motor Unit Index", 
        row=2, col=1,
        showgrid=True,
        gridwidth=0.5,
        gridcolor='lightgray',
        tickmode='linear',
        tick0=0,
        dtick=max(1, n_MU // 10)
    )

    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=14, family='Arial', weight='bold')

    # ==================================================
    # 🔹 GUARDAR
    # ==================================================
    save_path = os.path.join(exp_path, "online_interactive.html")
    save_path_muaps = os.path.join(exp_path, "muaps_analysis.html")
    
    # Opcional: guardar también una versión estática para informe
    # fig.write_html(save_path, config={'responsive': True})
    # fig_MUAPs.write_html(save_path_muaps, config={'responsive': True})

    print(f"✔️ Figure interactive sauvegardée: {save_path}")
    print(f"✔️ MUAPs figure sauvegardée: {save_path_muaps}")
    print(f"📊 Dimensions: Responsive (auto-ajustable)")
    
    return fig, fig_MUAPs