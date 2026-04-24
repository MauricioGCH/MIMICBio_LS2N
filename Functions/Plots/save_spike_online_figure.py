import os
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from Functions.Metrics import evaluate_spike_detection

def extract_waveforms_online(signal, U_est, spike_idx_by_MU, fs, config):
    pre = int(config["window_pre_ms"] * fs / 1000)
    post = int(config["window_post_ms"] * fs / 1000)
    
    waveforms_by_MU = []
    samples_lag = int(config["window_pre_ms"]*(1/1000)*fs)
    
    for u_vector in U_est:
        spike_idx = np.where(u_vector == 1)[0]
        waveforms = []
        for t in spike_idx:
            if t - pre >= 0 and t + post < len(signal):
                snippet = signal[t + samples_lag - pre : t + samples_lag + post]
                waveforms.append(snippet)
        
        if len(waveforms) > 0:
            waveforms_by_MU.append(np.array(waveforms))
        else:
            waveforms_by_MU.append(None)
    
    return waveforms_by_MU


def save_spike_online_figure(online_results, signal_filtered, config, exp_path, save=False, 
                             spike_idx_true=None, OFFSET_peaks=None):
    """
    Parameters:
    -----------
    spike_idx_true : list of arrays, opcional
        Ground truth spike indices por fuente (en muestras del dominio online)
    OFFSET_peaks : list of int, opcional
        Desplazamiento teórico para alinear ground truth al pico del spike
        (debe tener la misma longitud que spike_idx_true)
    """
    
    Y = online_results["Y"] if "Y" in online_results else None
    Y_est = online_results["Y_est"]
    U_est = online_results["U_est"]
    H_est = online_results["H_est"]
    Theta_est = online_results["Theta_est"]

    n_MU = config["n_sources"]

    L = len(Y)
    time = np.linspace(
        start=0,
        stop=L / config["sampling_rate"],
        num=L,
        endpoint=False
    )
    
    print("📊 Extrayendo waveforms individuales de los spikes online...") # esta detectando 0 waveforms
    
    online_waveforms = extract_waveforms_online(
        signal=signal_filtered,
        U_est=U_est,
        spike_idx_by_MU=None,
        fs=config["sampling_rate"],
        config=config
    )
    
    for mu_idx, waveforms in enumerate(online_waveforms):
        if waveforms is not None:
            print(f"   Source {mu_idx+1}: {len(waveforms)} spikes individuels")
        else:
            print(f"   Source {mu_idx+1}: 0 spikes")
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=(
            "Signal filtré vs reconstruit",
            "Trains d'impulsions estimés (U) vs Ground Truth"
        )
    )

    # ==================================================
    # 🔹 1. SIGNAL
    # ==================================================
    if Y is not None:
        if len(Y) > 50000:
            step = max(1, len(Y) // 50000)
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
            name='Signal filtré',
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
    # 🔹 2. RASTER PLOT — estimados + ground truth intercalados por fuente
    # ==================================================
    colors_raster = px.colors.qualitative.Plotly



    
    # Inicializar listas/arrays según n_MU
    sample_error = [[] for _ in range(n_MU)]  # Lista de listas, una por MU
    
    missed_detections = [0] * n_MU           # Lista de enteros, uno por MU
    false_positives = [0] * n_MU
    for i in range(n_MU):
        color_i = colors_raster[i % len(colors_raster)]

        # ----- Estimado: y = i (posición entera) -----
        spike_times_est = np.where(U_est[i] == 1)[0]
        spike_times_est_sec = spike_times_est / config["sampling_rate"]
        y_pos_est = np.ones_like(spike_times_est_sec) * i

        if len(spike_times_est) > 0:
            fig.add_trace(go.Scatter(
                x=spike_times_est_sec,
                y=y_pos_est,
                mode='markers',
                name=f"Source {i+1} estimé ({len(spike_times_est)} spikes)",
                marker=dict(
                    symbol='line-ns-open',
                    size=12,
                    color=color_i,
                    line=dict(width=1)
                ),
                opacity=0.9,
                showlegend=True,
                hovertemplate=(
                    f'<b>Source {i+1} estimé</b><br>'
                    'Time: %{x:.3f} s<br>'
                    f'Spike index: %{{customdata}}<extra></extra>'
                ),
                customdata=spike_times_est
            ), row=2, col=1)

        # ----- Ground truth: y = i - 0.35 (justo debajo) -----
        # 🔥 CORRECCIÓN CON OFFSET TEÓRICO (como en offline)
        if spike_idx_true is not None and i < len(spike_idx_true):
            gt_idx_i_raw = spike_idx_true[i]
            
            # Aplicar offset teórico si existe
            if OFFSET_peaks is not None and i < len(OFFSET_peaks):
                gt_idx_i = gt_idx_i_raw  #+ OFFSET_peaks[i] #No hay necesidad de offset, ya que el modelo tambien detecta el inicio, 
                                                                #es solamente para la parte offline donde se detecta es el pico y no el inicio del evento. Deberia eliminar esto.
            else:
                gt_idx_i = gt_idx_i_raw


            # Filtrar solo los que están dentro de la ventana online
            start = config["init_data"]
            end = start + L

            gt_idx_i = gt_idx_i[(gt_idx_i >= start) & (gt_idx_i < end)]
            


            # Calcular el error a nivel de muestras, es decir con exactitud de aceurdo al sampling rate
            idx_true = 0
            results = evaluate_spike_detection(spikes_abs=[x + config["init_data"] for x in spike_times_est], gt_peaks=gt_idx_i, tolerance= OFFSET_peaks[i] + 2, verbose= True)

            # for j in range(len(gt_idx_i)):

            #     if idx_true + missed_detections[i] > len(spike_times_est):
            #         xtra_missed = len(gt_idx_i) - len(spike_times_est)
            #         missed_detections[i] = missed_detections[i] + xtra_missed
                   
            #         break

            #     single_spike = spike_times_est[idx_true] + config["init_data"]
            #     single_spike_true = gt_idx_i[j]
            #     error = single_spike_true - single_spike
                

            #     ## If the error is bigger than the teorical offset, the spike wasn't detected. Im going to iterate accoring to the detected spikes
            #     #So when the condition is not meet, since im going from the beginning i will move i, but the index of the gt will remain until it finds a gt at offset samples lower
            #     # So i need to use another index for gt dependant on i but it changes with a condition
            #      # No esta entrando nunca, revisar. Podria tambien cambiar las listas por simplemente variables que se resetean por cada i, ya
            #                     #que al final por ahora solo los imprimo aunque talvez seria mejor dejar asi por si luego quiero devolver la variable
            #     if np.abs(error) < OFFSET_peaks[i]+ 2: 
            #         sample_error[i].append(error)
            #         idx_true +=1
            #     elif single_spike < single_spike_true - 2:
            #         # Falso positivo: spike detectado sin GT
            #         false_positives[i] += 1
            #         idx_true += 1  # Avanzar spike, mismo GT
                
            #     else:
            #         # Miss: GT no detectado
            #         missed_detections[i] += 1
            #         # No avanzar idx_true, el mismo spike podría corresponder al siguiente GT

            # print(f"--- Practical spike detection OFFset and Missed spikes --- \n")
            # print(f"There were {missed_detections[i]} for the corresponding Source {i +1}. \n ")
            # print(f"Moreover, the mean sample spike-peak detection was {np.mean(sample_error[i])} and the mode {multimode(sample_error[i])}. \n")

            gt_times_sec = (gt_idx_i - start) / config["sampling_rate"]
            y_pos_gt = np.ones_like(gt_times_sec) * (i - 0.35)

            fig.add_trace(go.Scatter(
                x=gt_times_sec,
                y=y_pos_gt,
                mode='markers',
                name=f"Source {i+1} GT ({len(gt_idx_i)} spikes)",
                marker=dict(
                    symbol='x',  # 🔥 Usamos 'x' igual que en offline
                    size=9,
                    color=color_i,
                    line=dict(width=1.5)
                ),
                opacity=0.7,
                showlegend=True,
                hovertemplate=(
                    f'<b>Source {i+1} GT (offset corregido)</b><br>'
                    'Time: %{x:.3f} s<extra></extra>'
                )
            ), row=2, col=1)
    
    # ==================================================
    # 🔹 3. MUAPs FIGURE
    # ==================================================
    fig_MUAPs = go.Figure()
    
    pre_ms = config["window_pre_ms"]
    post_ms = config["window_post_ms"]
    muap_time_ms = np.linspace(-pre_ms, post_ms, len(H_est[0]) if len(H_est) > 0 else 0)

    colors_muap = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    for i in range(n_MU):
        if online_waveforms[i] is not None:
            waveforms_i = online_waveforms[i]
            n_spikes = len(waveforms_i)
            color = colors_muap[i % len(colors_muap)]
            
            for w in waveforms_i:
                fig_MUAPs.add_trace(go.Scatter(
                    x=muap_time_ms,
                    y=w,
                    mode='lines',
                    line=dict(width=0.6, color=color),
                    opacity=0.7,
                    showlegend=False,
                    hoverinfo='skip',
                    legendgroup=f"Source_{i+1}_individual"
                ))
        
        n_spikes_total = len(online_waveforms[i]) if (i < len(online_waveforms) and online_waveforms[i] is not None) else 0
        color = colors_muap[i % len(colors_muap)]

        fig_MUAPs.add_trace(go.Scatter(
            x=muap_time_ms,
            y=H_est[i],
            mode='lines',
            name=f'Source {i+1} (moyen, n={n_spikes_total} spikes)',
            line=dict(width=3.5, color=color),
            showlegend=True,
            legendgroup=f"Source_{i+1}_avg",
            hovertemplate=f'<b>MU {i+1} Moyen</b><br>Time: %%{{x:.1f}} ms<br>Amplitude: %%{{y:.3f}}<extra></extra>'
        ))
        
        if i < len(online_waveforms) and online_waveforms[i] is not None and len(online_waveforms[i]) > 1:
            waveforms_i = online_waveforms[i]
            std_waveform = np.std(waveforms_i, axis=0)
            
            x_fill = np.concatenate([muap_time_ms, muap_time_ms[::-1]])
            y_fill = np.concatenate([H_est[i] + std_waveform, 
                                     (H_est[i] - std_waveform)[::-1]])
            
            color_rgb = colors_muap[i % len(colors_muap)].lstrip('#')
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
                legendgroup=f"Source_{i+1}_std",
                hoverinfo='skip'
            ))

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
            text=f"📊 Per source-APs Online - Spikes individuales (gris) + Promedio (color)",
            x=0.5,
            xanchor='center',
            font=dict(size=16, family='Arial', weight='bold')
        ),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    fig_MUAPs.update_xaxes(
        title_text="Time (ms)",
        showgrid=True, gridwidth=0.5, gridcolor='#e5e5e5',
        showline=True, linewidth=1, linecolor='black',
        mirror=True, zeroline=True, zerolinecolor='gray', zerolinewidth=0.5
    )
    
    fig_MUAPs.update_yaxes(
        title_text="Normalized Amplitude",
        showgrid=True, gridwidth=0.5, gridcolor='#e5e5e5',
        showline=True, linewidth=1, linecolor='black',
        mirror=True, zeroline=True, zerolinecolor='gray', zerolinewidth=0.5
    )
    
    total_spikes = sum(len(w) if w is not None else 0 for w in online_waveforms)
    stats_text = (
        f"<b>Statistiques en ligne:</b><br>"
        f"Nombre de sources: {n_MU}<br>"
        f"Spikes totaux: {total_spikes}<br>"
        f"Taux de décharge moyen: {total_spikes/(L/config['sampling_rate']):.1f} Hz"
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
    # 🔹 LAYOUT FIGURA PRINCIPAL
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
            text="📈 Résultats Algorithm 2 - Online",
            x=0.5,
            xanchor='center',
            font=dict(size=16, family='Arial', weight='bold')
        ),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    fig.update_xaxes(
        title_text="Temps (s)", row=1, col=1,
        showgrid=True, gridwidth=0.5, gridcolor='#e5e5e5',
        showline=True, linewidth=1, linecolor='black',
        mirror=True, zeroline=True, zerolinecolor='gray', zerolinewidth=1
    )
    fig.update_yaxes(
        title_text="Amplitude (mV)", row=1, col=1,
        showgrid=True, gridwidth=0.5, gridcolor='#e5e5e5',
        showline=True, linewidth=1, linecolor='black', mirror=True
    )
    fig.update_xaxes(
        title_text="Temps (s)", row=2, col=1,
        showgrid=True, gridwidth=0.5, gridcolor='#e5e5e5',
        showline=True, linewidth=1, linecolor='black', mirror=True
    )
    fig.update_yaxes(
        title_text="Source Index", row=2, col=1,
        showgrid=True, gridwidth=0.5, gridcolor='#e5e5e5',
        showline=True, linewidth=1, linecolor='black', mirror=True,
        tickmode='array',
        tickvals=list(range(n_MU)),
        ticktext=[f'S{i+1}' for i in range(n_MU)]
    )

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
            config={'responsive': True, 'displayModeBar': True, 'displaylogo': False, 'scrollZoom': True}
        )
        
        fig_MUAPs.write_html(
            save_path_muaps,
            include_plotlyjs='cdn',
            full_html=True,
            div_id="fig_online_muaps",
            config={'responsive': True, 'displayModeBar': True, 'displaylogo': False, 'scrollZoom': True}
        )

    print(f"\n{'='*60}")
    print(f"✅ Online Analysis Complete")
    print(f"{'='*60}")
    if save:
        print(f"📁 Figure interactive: {save_path}")
        print(f"📁 Per Source APs figure: {save_path_muaps}")
    print(f"\n📊 Online Statistics:")
    print(f"   • Sources detectées: {n_MU}")
    print(f"   • Total spikes: {total_spikes}")
    print(f"   • Taux: {total_spikes/(L/config['sampling_rate']):.1f} Hz")
    for i, w in enumerate(online_waveforms):
        n_spikes = len(w) if w is not None else 0
        print(f"   • MU {i+1}: {n_spikes} spikes")
    print(f"{'='*60}\n")

    print("\n🔍 DEBUG: Verificación de tiempos")
    print(f"Señal online: {len(Y)} muestras, {len(Y)/config['sampling_rate']:.2f} segundos")
    for mu_idx in range(n_MU):
        spikes = np.where(U_est[mu_idx] == 1)[0]
        if len(spikes) > 0:
            print(f"\nMU{mu_idx+1}:")
            print(f"  Primer spike: muestra {spikes[0]} = {spikes[0]/config['sampling_rate']:.3f}s")
            print(f"  Último spike: muestra {spikes[-1]} = {spikes[-1]/config['sampling_rate']:.3f}s")
            peak_H = np.argmax(np.abs(H_est[mu_idx]))
            peak_time_H = (peak_H - int(config["window_pre_ms"]*config["sampling_rate"]/1000)) / config["sampling_rate"] * 1000
            print(f"  Pico en H_est: muestra {peak_H} = {peak_time_H:.1f}ms (relativo al trigger)")
    
    return fig, fig_MUAPs