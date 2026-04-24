##plots

import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import signal



def save_spectral_analysis(Y, Y_pred, config, exp_path, save = False):
    """
    Análisis espectral tiempo-frecuencia usando STFT:
    - Espectrograma de señal original
    - Espectrograma de reconstrucción
    - Espectrograma del residual
    """
    
    fs = config["sampling_rate"]
    
    # 🔹 Parámetros de la STFT (ajustables)
    # Ventana: tamaño en segundos (típico: 0.05-0.1s para EMG)
    window_duration = 0.05  # 50 ms
    # Overlap típico: 50-75% para buena resolución temporal
    overlap_percentage = 0.75  # 75% overlap
    
    # Convertir a muestras
    nperseg = int(window_duration * fs)
    noverlap = int(nperseg * overlap_percentage)
    
    # 🔹 Calcular espectrogramas
    def compute_spectrogram(signal_data, signal_name):
        """
        Calcula el espectrograma usando scipy.signal.spectrogram
        Retorna: freqs, times, spectrogram (en dB)
        """
        # Calcular espectrograma
        f, t, Sxx = signal.spectrogram(
            signal_data,
            fs=fs,
            nperseg=nperseg,
            noverlap=noverlap,
            window='hann',  # Ventana de Hann (buena para EMG)
            scaling='density',  # Densidad espectral
            mode='psd'  # Densidad espectral de potencia
        )
        
        # Convertir a dB para mejor visualización
        Sxx_db = 10 * np.log10(Sxx + 1e-12)  # Pequeño epsilon para evitar log(0)
        
        return f, t, Sxx_db
    
    # Calcular para las tres señales
    f_orig, t_orig, Sxx_orig = compute_spectrogram(Y, 'Original')
    f_pred, t_pred, Sxx_pred = compute_spectrogram(Y_pred, 'Predicted')
    
    # Para el residual, podrías querer usar escala lineal para ver errores pequeños
    residual = Y - Y_pred
    f_res, t_res, Sxx_res = compute_spectrogram(residual, 'Residual')
    
    # 🔹 Crear figura con subplots
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            f'Espectrograma - Señal Original (ventana={window_duration}s, overlap={overlap_percentage*100}%)',
            'Espectrograma - Reconstrucción',
            'Espectrograma - Residual (error)'
        ),
        vertical_spacing=0.12,
        shared_xaxes=True  # Compartir eje x (tiempo) para sincronizar
    )
    
    # 🔹 Espectrograma 1: Señal original
    fig.add_trace(
        go.Heatmap(
            z=Sxx_orig,
            x=t_orig,
            y=f_orig,
            colorscale='Viridis',
            zmin=np.percentile(Sxx_orig, 5),  # Mejorar contraste
            zmax=np.percentile(Sxx_orig, 95),
            colorbar=dict(
                title="Power (dB)",
                x=1.02,
                len=0.3,
                y=0.8
            ),
            name='Original',
            showscale=True
        ),
        row=1, col=1
    )
    
    # 🔹 Espectrograma 2: Reconstrucción
    fig.add_trace(
        go.Heatmap(
            z=Sxx_pred,
            x=t_pred,
            y=f_pred,
            colorscale='Viridis',
            zmin=np.percentile(Sxx_pred, 5),
            zmax=np.percentile(Sxx_pred, 95),
            colorbar=dict(
                title="Power (dB)",
                x=1.02,
                len=0.3,
                y=0.4
            ),
            name='Predicted',
            showscale=True
        ),
        row=2, col=1
    )
    
    # 🔹 Espectrograma 3: Residual (escala diferente para ver errores)
    fig.add_trace(
        go.Heatmap(
            z=Sxx_res,
            x=t_res,
            y=f_res,
            colorscale='RdBu',  # Rojo-Azul para destacar diferencias
            zmin=-60,  # Límites fijos para comparación
            zmax=-20,
            colorbar=dict(
                title="Power (dB)",
                x=1.02,
                len=0.3,
                y=0.1
            ),
            name='Residual',
            showscale=True
        ),
        row=3, col=1
    )
    
    # 🔹 Layout mejorado
    fig.update_layout(
        title=dict(
            text="Time-Frequency Analysis (STFT)",
            x=0.5,
            font=dict(size=16)
        ),
        height=1200,  # Más alto para 3 espectrogramas
        autosize=True,
        template='plotly_white',
        margin=dict(l=60, r=80, t=100, b=60)
    )
    
    # Configurar ejes
    fig.update_xaxes(
        title_text="Time (seconds)",
        row=1, col=1,  # Solo el último subplot necesita título x
        showgrid=True,
        gridcolor='lightgray'
    )
    fig.update_xaxes(
        title_text="Time (seconds)",
        row=2, col=1,  # Solo el último subplot necesita título x
        showgrid=True,
        gridcolor='lightgray'
    )
    fig.update_xaxes(
        title_text="Time (seconds)",
        row=3, col=1,  # Solo el último subplot necesita título x
        showgrid=True,
        gridcolor='lightgray'
    )
    
    fig.update_yaxes(
        title_text="Frequency (Hz)",
        row=1, col=1,
        showgrid=True,
        gridcolor='lightgray',
        range=[0, fs/2]  # Mostrar hasta Nyquist
    )
    
    fig.update_yaxes(
        title_text="Frequency (Hz)",
        row=2, col=1,
        showgrid=True,
        gridcolor='lightgray',
        range=[0, fs/2]
    )
    
    fig.update_yaxes(
        title_text="Frequency (Hz)",
        row=3, col=1,
        showgrid=True,
        gridcolor='lightgray',
        range=[0, fs/2]
    )
    
    # 🔹 Opcional: Añadir líneas de frecuencia de interés (ej. ritmos beta, gamma)
    # Esto es útil para EMG
    #fig.add_hline(y=20, line_dash="dash", line_color="red", 
    #              annotation_text="Ritmo Beta", row=1, col=1)
    #fig.add_hline(y=60, line_dash="dash", line_color="orange",
    #               annotation_text="Ritmo Gamma", row=1, col=1)
    
    # 🔹 Guardar  
    if save:
        save_path = os.path.join(exp_path, "spectral_analysis_stft.html")
        fig.write_html(
            save_path,
            include_plotlyjs='cdn',
            full_html=True,
            div_id="fig_spectral",
            config={
                'responsive': True,
                'displayModeBar': True,
                'scrollZoom': True  # Permitir zoom en frecuencia/tiempo
            }
        )
    if save:
        print(f"✔️ Análisis tiempo-frecuencia guardado: {save_path}")
    #print(f"📊 Parámetros STFT: ventana={window_duration}s, overlap={overlap_percentage*100}%")
    #print(f"📈 Resolución temporal: {t_orig[1]-t_orig[0]:.3f}s")
    #print(f"📉 Resolución frecuencial: {f_orig[1]-f_orig[0]:.1f}Hz")
    
    # ==================================================
    # 🔹 FIGURA ACF DEL RESIDUAL - VERSIÓN MEJORADA
    # ==================================================
    max_lags = min(500, len(residual) // 2)

    # Calcular ACF normalizada
    acf_full = np.correlate(residual - np.mean(residual), 
                            residual - np.mean(residual), mode='full')
    acf_full = acf_full / acf_full[len(acf_full) // 2]
    acf = acf_full[len(acf_full) // 2 : len(acf_full) // 2 + max_lags + 1]

    lags = np.arange(0, max_lags + 1) / fs

    # Bandas de confianza 95%
    confidence = 1.96 / np.sqrt(len(residual))

    fig_acf = go.Figure()

    # 🔥 OPCIÓN 1: Usar líneas en lugar de barras (menos "zoom out")
    fig_acf.add_trace(go.Scatter(
        x=lags,
        y=acf,
        mode='lines+markers',
        name='ACF',
        line=dict(color='#1f77b4', width=1.5),
        marker=dict(size=3, color='#1f77b4'),
        hovertemplate='Lag: %{x:.4f}s<br>ACF: %{y:.4f}<extra></extra>'
    ))

    # 🔥 OPCIÓN 2: Barras más delgadas y con límite Y ajustado
    # fig_acf.add_trace(go.Bar(
    #     x=lags,
    #     y=acf,
    #     name='ACF',
    #     marker=dict(color='#1f77b4', opacity=0.7),
    #     width=0.01,  # Barras muy delgadas
    # ))

    # Banda de confianza
    fig_acf.add_trace(go.Scatter(
        x=lags, y=np.full_like(lags, confidence),
        mode='lines',
        name='95% Confidence (+)',
        line=dict(color='red', dash='dash', width=1.5)
    ))

    fig_acf.add_trace(go.Scatter(
        x=lags, y=np.full_like(lags, -confidence),
        mode='lines',
        name='95% Confidence (-)',
        line=dict(color='red', dash='dash', width=1.5),
        fill='tonexty',
        fillcolor='rgba(255, 0, 0, 0.05)'
    ))

    fig_acf.add_hline(y=0, line_color='black', line_width=0.5)

    # 🔥 MEJORA: Zoom en los lags pequeños (más relevantes)
    # En lugar de mostrar todos los 500 lags, mostrar primeros 100-200
    max_lags_display = min(200, max_lags)
    fig_acf.update_xaxes(range=[0, max_lags_display / fs])

    fig_acf.update_layout(
        autosize=True,
        height=450,
        margin=dict(l=80, r=60, t=100, b=80),
        template='plotly_white',
        title=dict(
            text='Residual Autocorrelation Function (ACF)',
            x=0.5,
            font=dict(size=16, family='Arial', weight='bold')
        ),
        xaxis=dict(
            title='Lag (seconds)',
            showgrid=True,
            gridcolor='#e5e5e5',
            showline=True,
            linecolor='black'
        ),
        yaxis=dict(
            title='Autocorrelation',
            showgrid=True,
            gridcolor='#e5e5e5',
            showline=True,
            linecolor='black',
            range=[-0.2, 1.05]  # 🔥 Ajustado para ver mejor los detalles
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        )
    )

    # Estadísticas
    lags_outside = np.sum(np.abs(acf[1:]) > confidence)
    pct_outside = 100 * lags_outside / max_lags

    acf_annotation = (
        f"<b>ACF Diagnostics:</b><br>"
        f"Max lags: {max_lags} ({max_lags/fs:.2f}s)<br>"
        f"Confidence band: ±{confidence:.4f}<br>"
        f"Lags outside band: {lags_outside} ({pct_outside:.1f}%)<br>"
        f"{'✅ White noise' if pct_outside < 5 else '⚠️ Some structure remains'}"
    )

    fig_acf.add_annotation(
        text=acf_annotation,
        xref='paper', yref='paper',
        x=0.98, y=0.98,
        showarrow=False,
        font=dict(size=11, family='monospace'),
        bgcolor='rgba(255,255,255,0.9)',
        bordercolor='gray',
        borderwidth=1,
        borderpad=8,
        align='left'
    )
    if save:
        save_path_acf = os.path.join(exp_path, "residual_acf.html")
        fig_acf.write_html(
            save_path_acf,
            include_plotlyjs='cdn',
            full_html=True,
            config={'responsive': True, 'displayModeBar': True, 'scrollZoom': True}
        )
        print(f"✔️ ACF guardado: {save_path_acf}")

    # Modificar el return existente
    return fig, fig_acf  # <- antes solo retornaba fig


    