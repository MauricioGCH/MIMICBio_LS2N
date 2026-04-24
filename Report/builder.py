import os
import json

def fig_to_html(fig):
    return fig.to_html(
        full_html=False,
        include_plotlyjs="cdn"
    )
def build_report(figures, metrics, config, exp_path):
    # 🔹 cargar template
    template_path = os.path.join(os.path.dirname(__file__), "template.html")
    with open(template_path, "r", encoding="utf-8") as f:
        html = f.read()

    # ==================================================
    # 🔹 FIGURAS
    # ==================================================

    spikes_html = fig_to_html(figures["spikes"])
    muaps_offline_html = fig_to_html(figures["muaps_offline"])
    isi_html = fig_to_html(figures["isi"])

    online_html = fig_to_html(figures["online"])
    MUAP_html = fig_to_html(figures["online_MUAP"])

    residual_html = fig_to_html(figures["residual"])
    residual_hist_html = fig_to_html(figures["residual_hist"])


    offline_block = f"""
    <div class="box">{spikes_html}</div>
    <div class="box">{muaps_offline_html}</div>
    <div class="box">{isi_html}</div>
    """


    online_block = f"""
    <div class="box">{online_html}</div>
    <div class="box">{MUAP_html}</div>
    """

    signal_block = f"""
    <div class="box">{residual_html}</div>
    <div class="box">{residual_hist_html}</div>
    """

    spectral_block = f"""
    <div class="box">
        {fig_to_html(figures["spectral"])}
    </div>
    <div class="box">
        {fig_to_html(figures["acf"])}
    </div>
    """

    weibull_block = f"""
    <div class="box">
        {fig_to_html(figures["weibull"])}
    </div>
    """

    # ==================================================
    # 🔹 METRICS
    # ==================================================
    metrics_html = "<ul>"
    for k, v in metrics.items():
        metrics_html += f"<li><b>{k}:</b> {v}</li>"
    metrics_html += "</ul>"

    metrics_block = f"""
    <div class="box">
        <pre>{json.dumps(config, indent=2)}</pre>
        {metrics_html}
    </div>
    """

    # ==================================================
    # 🔹 REEMPLAZOS
    # ==================================================
    offline_block
    html = html.replace("{{offline_block}}", offline_block)
    html = html.replace("{{online_block}}", online_block)
    html = html.replace("{{signal_block}}", signal_block)
    html = html.replace("{{spectral_block}}", spectral_block)
    html = html.replace("{{weibull_block}}", weibull_block)
    html = html.replace("{{metrics_block}}", metrics_block)

    # ==================================================
    # 🔹 GUARDAR
    # ==================================================
    save_path = os.path.join(exp_path, "report.html")
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"📊 Report saved: {save_path}")