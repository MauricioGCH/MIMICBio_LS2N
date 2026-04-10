import os
import numpy as np
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots



def save_residual_analysis(Y, Y_pred, config, exp_path, save = False):
    """
    Calcula y guarda análisis del residual:
    - señal real vs reconstrucción
    - residual
    - histograma del residual (figura separada)
    
    📊 Formato responsivo con márgenes equilibrados
    """
    # 🔹 residual
    residual = Y - Y_pred
    
    # 🔹 tiempo en segundos
    N = len(Y)
    fs = config["sampling_rate"]
    time = np.linspace(0, N / fs, N, endpoint=False)
    
    # 🔹 Estadísticas del residual para info
    residual_mean = np.mean(residual)
    residual_std = np.std(residual)
    residual_rms = np.sqrt(np.mean(residual**2))
    residual_max = np.max(np.abs(residual))
    
    # ==================================================
    # 🔹 FIGURA 1: Signal + Residual (shared x-axis)
    # ==================================================
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,  # Espaciado equilibrado
        subplot_titles=(
            "📈 Signal vs Reconstruction",
            "📉 Residual (Y - Y_pred)"
        )
    )
    
    # 🔹 Señal original (con opacidad para ver la reconstrucción detrás)
    fig.add_trace(go.Scatter(
        x=time, y=Y,
        mode='lines', 
        name='Signal Original',
        line=dict(width=1.5, color='#1f77b4'),
        opacity=0.7,
        legendgroup="signal",
        showlegend=True
    ), row=1, col=1)
    
    # 🔹 Reconstrucción (más gruesa y destacada)
    fig.add_trace(go.Scatter(
        x=time, y=Y_pred,
        mode='lines', 
        name='Reconstruction',
        line=dict(width=2, dash='dash', color='#ff7f0e'),
        legendgroup="reconstruction",
        showlegend=True
    ), row=1, col=1)
    
    # 🔹 Residual (con área sombreada para mejor visualización)
    fig.add_trace(go.Scatter(
        x=time, y=residual,
        mode='lines', 
        name='Residual',
        line=dict(width=1.5, color='#d62728'),
        fill='tozeroy',  # Área sombreada bajo la curva
        fillcolor='rgba(214, 39, 40, 0.2)',  # Rojo con transparencia
        legendgroup="residual",
        showlegend=True
    ), row=2, col=1)
    
    # 🔹 Añadir línea cero en el residual para referencia
    fig.add_hline(y=0, line_dash="solid", line_color="black", 
                  line_width=0.5, opacity=0.5, row=2, col=1)
    
    # 🔹 Añadir bandas de ±2σ en el residual
    fig.add_hrect(y0=-2*residual_std, y1=2*residual_std,
                  line_width=0, fillcolor="green", opacity=0.1,
                  row=2, col=1,
                  annotation_text=f"±2σ ({2*residual_std:.3f})",
                  annotation_position="top right")
    
    # ==================================================
    # 🔹 LAYOUT PRINCIPAL - Responsivo y bonito
    # ==================================================
    fig.update_layout(
        # 🔥 Responsivo
        autosize=True,
        height=800,  # Alto adecuado para 2 subplots
        
        # Márgenes equilibrados
        margin=dict(l=80, r=60, t=100, b=80),
        
        # Template moderno
        template="plotly_white",
        
        # Título principal
        title=dict(
            text=f"📊 Residual Analysis - Calidad de Reconstrucción",
            x=0.5,
            xanchor='center',
            font=dict(size=18, family='Arial', weight='bold'),
            y=0.98
        ),
        
        # Leyenda optimizada
        legend=dict(
            orientation="h",  # Horizontal para ahorrar espacio
            yanchor="bottom",
            y=1.06,
            xanchor="center",
            x=0.5,
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='lightgray',
            borderwidth=1
        ),
        
        # Grid y fondo
        plot_bgcolor='white',
        paper_bgcolor='white',
        
        # Tooltips unificados
        hovermode='x unified'
    )
    
    # ==================================================
    # 🔹 CONFIGURACIÓN DE EJES - Subplot 1
    # ==================================================
    fig.update_xaxes(
        title_text="",  # Sin título aquí porque shared_xaxes
        row=1, col=1,
        showgrid=True,
        gridwidth=0.5,
        gridcolor='#e5e5e5',
        showline=True,
        linewidth=1,
        linecolor='black',
        mirror=True
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
        mirror=True,
        zeroline=True,
        zerolinecolor='gray',
        zerolinewidth=0.5
    )
    
    # ==================================================
    # 🔹 CONFIGURACIÓN DE EJES - Subplot 2
    # ==================================================
    fig.update_xaxes(
        title_text="Time (seconds)",
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
        title_text="Residual Amplitude (mV)",
        row=2, col=1,
        showgrid=True,
        gridwidth=0.5,
        gridcolor='#e5e5e5',
        showline=True,
        linewidth=1,
        linecolor='black',
        mirror=True,
        zeroline=True,
        zerolinecolor='red',
        zerolinewidth=1
    )
    
    # ==================================================
    # 🔹 AÑADIR CAJA DE ESTADÍSTICAS (annotation)
    # ==================================================
    stats_text = (
        f"<b>Residual Statistics:</b><br>"
        f"Mean: {residual_mean:.4e}<br>"
        f"Std: {residual_std:.4e}<br>"
        f"RMS: {residual_rms:.4e}<br>"
        f"Max |error|: {residual_max:.4e}"
    )
    
    fig.add_annotation(
        text=stats_text,
        xref="paper", yref="paper",
        x=1.02, y=1.05,
        showarrow=False,
        font=dict(size=11, family='monospace'),
        bgcolor='rgba(255, 255, 255, 0.9)',
        bordercolor='gray',
        borderwidth=1,
        borderpad=8,
        align='left'
    )
    
    # ==================================================
    # 🔹 FIGURA 2: HISTOGRAMA (mejorado)
    # ==================================================
    fig_hist = go.Figure()
    
    # Histograma principal
    fig_hist.add_trace(go.Histogram(
        x=residual,
        nbinsx=50,
        histnorm='probability density',
        name='Residual Distribution',
        opacity=0.7,
        marker=dict(color='#d62728', line=dict(color='black', width=0.5)),
        hovertemplate='Value: %{x:.4f}<br>Density: %{y:.4f}<extra></extra>'
    ))
    
    # Añadir curva de distribución normal teórica (para comparación)
    x_norm = np.linspace(residual.min(), residual.max(), 100)
    y_norm = (1/(residual_std * np.sqrt(2*np.pi))) * np.exp(-0.5*((x_norm - residual_mean)/residual_std)**2)
    
    fig_hist.add_trace(go.Scatter(
        x=x_norm, y=y_norm,
        mode='lines',
        name='Normal Distribution',
        line=dict(color='blue', width=2, dash='dash'),
        hovertemplate='Value: %{x:.4f}<br>Theoretical: %{y:.4f}<extra></extra>'
    ))
    
    # Líneas verticales para estadísticas
    fig_hist.add_vline(x=residual_mean, line_dash="solid", line_color="green",
                       annotation_text=f"Mean: {residual_mean:.4f}",
                       annotation_position="top",
                       line_width=2)
    
    fig_hist.add_vline(x=residual_mean - residual_std, line_dash="dash", line_color="orange",
                       annotation_text=f"-1σ",
                       annotation_position="bottom")
    
    fig_hist.add_vline(x=residual_mean + residual_std, line_dash="dash", line_color="orange",
                       annotation_text=f"+1σ",
                       annotation_position="bottom")
    
    # Layout del histograma
    fig_hist.update_layout(
        autosize=True,
        height=500,
        margin=dict(l=80, r=60, t=100, b=80),
        template="plotly_white",
        title=dict(
            text=f"📊 Residual Distribution Analysis",
            x=0.5,
            xanchor='center',
            font=dict(size=16, family='Arial', weight='bold')
        ),
        xaxis=dict(
            title=dict(text="Residual Value (mV)", font=dict(size=12)),
            showgrid=True,
            gridwidth=0.5,
            gridcolor='#e5e5e5',
            showline=True,
            linewidth=1,
            linecolor='black',
            zeroline=True,
            zerolinecolor='gray'
        ),
        yaxis=dict(
            title=dict(text="Probability Density", font=dict(size=12)),
            showgrid=True,
            gridwidth=0.5,
            gridcolor='#e5e5e5',
            showline=True,
            linewidth=1,
            linecolor='black'
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor='rgba(255, 255, 255, 0.9)'
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='closest'
    )
    
    # Añadir estadísticas en el histograma
    shapiro_text = (
        f"<b>Quality Metrics:</b><br>"
        f"RMSE: {residual_rms:.4f}<br>"
        f"SNR: {20*np.log10(np.std(Y)/residual_std):.2f} dB<br>"
        f"Mean/Std ratio: {abs(residual_mean/residual_std):.3f}"
    )
    
    fig_hist.add_annotation(
        text=shapiro_text,
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
    
    # Guardar figura principal
    save_path= None
    save_path_hist = None
    if save:
        save_path = os.path.join(exp_path, "residual_analysis.html")
        fig.write_html(
            save_path,
            include_plotlyjs='cdn',
            full_html=True,
            div_id="fig_residual",
            config={
                'responsive': True,
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                'scrollZoom': True
            }
        )
        
        # Guardar histograma
        save_path_hist = os.path.join(exp_path, "residual_histogram.html")
        fig_hist.write_html(
            save_path_hist,
            include_plotlyjs='cdn',
            full_html=True,
            div_id="fig_residual_hist",
            config={
                'responsive': True,
                'displayModeBar': True,
                'displaylogo': False,
                'scrollZoom': True
            }
        )
    
    # ==================================================
    # 🔹 IMPRIMIR INFO EN CONSOLA
    # ==================================================
    print(f"\n{'='*60}")
    print(f"✅ Residual Analysis Complete")
    print(f"{'='*60}")
    if save:
        print(f"📁 Main figure: {save_path}")
        print(f"📁 Histogram:   {save_path_hist}")
    print(f"\n📊 Residual Statistics:")
    print(f"   • Mean:     {residual_mean:.4e}")
    print(f"   • Std:      {residual_std:.4e}")
    print(f"   • RMS:      {residual_rms:.4e}")
    print(f"   • Max|e|:   {residual_max:.4e}")
    print(f"   • SNR:      {20*np.log10(np.std(Y)/residual_std):.2f} dB")
    print(f"{'='*60}\n")
    
    return fig, fig_hist