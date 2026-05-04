import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import shapiro, kstest, norm
import matplotlib.pyplot as plt


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
    
    threshold_noise = 2*residual_std
    # Estadísticas como anotación
    outliers = np.sum(np.abs(residual) > threshold_noise)
    pct_outliers = 100 * outliers / len(residual)

    # ==================================================
    # 🔹 AÑADIR CAJA DE ESTADÍSTICAS (annotation)
    # ==================================================
    stats_text = (
        f"<b>Residual Statistics:</b><br>"
        f"Mean: {residual_mean:.4e}<br>"
        f"Std: {residual_std:.4e}<br>"
        f"RMS: {residual_rms:.4e}<br>"
        f"Max |error|: {residual_max:.4e}"
        f"Outliers > ±{threshold_noise:.0f}µV: {outliers} ({pct_outliers:.3f}%)"
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
        nbinsx=150,
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
    
    # Configurar escala log en Y
    #fig_hist.update_yaxes(type="log")

    threshold_noise = 2*residual_std
    # Estadísticas como anotación
    outliers = np.sum(np.abs(residual) > threshold_noise)
    pct_outliers = 100 * outliers / len(residual)

    # 🔹 Tests de normalidad
    sample = residual if len(residual) <= 3000 else np.random.choice(residual, 3000, replace=False)
    stat_shapiro, p_shapiro = shapiro(sample)
    # KS test contra distribución normal con media y std del residual
    #stat_ks, p_ks = kstest(residual, 'norm', args=(residual_mean, residual_std))

    # Interpretación
    shapiro_result = "✅ Normal" if p_shapiro > 0.05 else "❌ Not Normal"
    #ks_result = "✅ Normal" if p_ks > 0.05 else "❌ Not Normal"
    
    # Añadir estadísticas en el histograma

    shapiro_text = (
    f"<b>Quality Metrics:</b><br>"
    f"RMSE: {residual_rms:.4f}<br>"
    f"SNR: {20*np.log10(np.std(Y)/residual_std):.2f} dB<br>"
    f"Mean/Std ratio: {abs(residual_mean/residual_std):.3f}<br>"
    f"Outliers > ±2σ: {outliers} ({pct_outliers:.3f}%)<br>"
    f"<br>"
    f"<b>Normality Tests:</b><br>"
    f"Shapiro-Wilk: p={p_shapiro:.4f} {shapiro_result}<br>"
    #f"KS Test:      p={p_ks:.4f} {ks_result}"
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
    
    return fig, fig_hist, residual




""""Cambio      Tu versión          Nueva versión

Posiciones | Hazen aproximado (i-0.5)/n |  Blom (i-0.375)/(n+0.25) — más preciso en extremos

Línea de referencia | Diagonal y=x | Regresión por cuartiles — robusta a outliers

Bandas de confianza | Ausentes |  ±1.96·SE al 95% — permiten juzgar visualmente si las desviaciones son significativas"""

def save_qq_plot(residuals, fs, exp_path, save=True):
    """
    Genera y guarda un QQ plot de los residuos con bandas de confianza al 95%.
    
    Parámetros:
    -----------
    residuals : array
        Residuos del modelo
    fs : int
        Frecuencia de muestreo (para el título)
    exp_path : str
        Ruta de guardado
    save : bool
        Si True guarda el archivo, si False muestra en pantalla
    
    Retorna:
    --------
    fig : matplotlib Figure
    """
    import scipy.stats as stats

    # ==================================================
    # 🔹 ESTANDARIZAR RESIDUOS
    # ==================================================
    residuals_std = (residuals - np.mean(residuals)) / np.std(residuals)
    sorted_residuals = np.sort(residuals_std)
    n = len(sorted_residuals)

    # ==================================================
    # 🔹 CUANTILES TEÓRICOS
    # ==================================================
    # Posiciones de Hazen (más precisas que (i-0.5)/n para extremos)
    positions = (np.arange(1, n + 1) - 0.375) / (n + 0.25)
    theoretical_quantiles = stats.norm.ppf(positions)

    # ==================================================
    # 🔹 BANDAS DE CONFIANZA AL 95% (método de Kolmogorov-Smirnov)
    # La banda refleja la incertidumbre esperada si los datos son normales
    # ==================================================
    se = (1 / stats.norm.pdf(theoretical_quantiles)) * np.sqrt(
        positions * (1 - positions) / n
    )
    ci_upper = theoretical_quantiles + 1.96 * se
    ci_lower = theoretical_quantiles - 1.96 * se

    # ==================================================
    # 🔹 LÍNEA DE REFERENCIA (regresión robusta por cuartiles)
    # Más robusta que y=x cuando hay outliers
    # ==================================================
    q25_t, q75_t = np.percentile(theoretical_quantiles, [25, 75])
    q25_r, q75_r = np.percentile(sorted_residuals, [25, 75])
    slope = (q75_r - q25_r) / (q75_t - q25_t)
    intercept = q25_r - slope * q25_t

    line_x = np.array([theoretical_quantiles.min(), theoretical_quantiles.max()])
    line_y = slope * line_x + intercept

    # ==================================================
    # 🔹 FIGURA
    # ==================================================
    fig, ax = plt.subplots(figsize=(7, 7))

    # Banda de confianza
    ax.fill_between(
        theoretical_quantiles, ci_lower, ci_upper,
        alpha=0.15, color='royalblue', label='95% Confidence band'
    )

    # Puntos
    ax.scatter(
        theoretical_quantiles, sorted_residuals,
        s=8, alpha=0.4, color='steelblue', zorder=3
    )

    # Línea de referencia
    ax.plot(line_x, line_y, 'r--', lw=2, label='Reference line (IQR fit)')

    # ==================================================
    # 🔹 DECORACIÓN
    # ==================================================
    n_outside = np.sum(
        (sorted_residuals < ci_lower) | (sorted_residuals > ci_upper)
    )
    pct_outside = 100 * n_outside / n

    ax.set_xlabel('Theoretical quantiles (Normal)', fontsize=12)
    ax.set_ylabel('Observed quantiles (Residuals)', fontsize=12)
    ax.set_title(
        f'QQ Plot — Standardized Residuals\n'
        f'n={n} | {n_outside} points outside band ({pct_outside:.1f}%)',
        fontsize=13
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    # ==================================================
    # 🔹 GUARDAR
    # ==================================================
    if save:
        save_path = os.path.join(exp_path, "residual_qq_plot.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📁 QQ plot saved: {save_path}")
        print(f"   • Points outside 95% band: {n_outside} ({pct_outside:.1f}%)")
        print(f"   • Interpretation: {'✅ Approximately normal' if pct_outside < 10 else '⚠️ Deviations from normality detected'}")
    else:
        plt.show()

    return fig