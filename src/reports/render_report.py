#!/usr/bin/env python
import argparse
import logging
from pathlib import Path
import pandas as pd
import plotly.express as px
import folium
from folium.plugins import HeatMap
from jinja2 import Environment, FileSystemLoader, select_autoescape
from weasyprint import HTML
from pydantic import BaseModel, FilePath, ValidationError
import json
import pickle
import tempfile
import shutil
from PIL import Image
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

# --- Logging setup ---
ARTIFACTS = Path("artifacts")
LOGS_DIR = ARTIFACTS / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=LOGS_DIR / "reports.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

# --- Pydantic validation ---
class InputPaths(BaseModel):
    features: FilePath
    metrics: FilePath
    model: FilePath

# --- Data loading ---
def load_data(features_path, metrics_path, model_path):
    logging.info("Loading features from %s", features_path)
    df = pd.read_parquet(features_path)
    logging.info("Loading metrics from %s", metrics_path)
    with open(metrics_path) as f:
        metrics = json.load(f)
    logging.info("Loading model from %s", model_path)
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return df, metrics, model

# --- Chart generation ---
def generate_charts(df):
    charts = {}

    # Top-10 bairros por volume
    top10 = df.groupby('neighborhood')['request_count'].sum().sort_values(ascending=False).head(10).reset_index()
    fig_bar = px.bar(
        top10,
        x='neighborhood',
        y='request_count',
        title='Top 10 bairros por chamados 311',
        color='request_count',
        color_continuous_scale=['#007BFF', '#28A745'],
        template='simple_white'
    )
    fig_bar.update_layout(font_family="Arial")
    charts['bar_top10'] = fig_bar

    # Heatmap (lat/lon fictícios)
    # Gerar lat/lon fictício para cada bairro
    neighborhoods = top10['neighborhood'].tolist()
    center_lat, center_lon = 49.246, -123.116
    lats = [center_lat + 0.01 * (i - 5) for i in range(len(neighborhoods))]
    lons = [center_lon + 0.01 * (i - 5) for i in range(len(neighborhoods))]
    heat_data = [[lat, lon, count] for lat, lon, count in zip(lats, lons, top10['request_count'])]
    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles='cartodbpositron')
    HeatMap(heat_data, radius=25, blur=15, max_zoom=1).add_to(fmap)
    charts['heatmap'] = fmap

    return charts

# --- Insights generation ---
def create_insights(df):
    insights = []
    # Δ semanal e anual para top categorias/bairros
    df['week'] = pd.to_datetime(df['week'])
    recent = df[df['week'] == df['week'].max()]
    prev4 = df[df['week'] >= (df['week'].max() - pd.Timedelta(weeks=4))]
    for cat in recent['service_category'].unique():
        for bairro in recent['neighborhood'].unique():
            cur = recent[(recent['service_category'] == cat) & (recent['neighborhood'] == bairro)]['request_count'].sum()
            prev = prev4[(prev4['service_category'] == cat) & (prev4['neighborhood'] == bairro)]['request_count'].mean()
            if prev > 0:
                delta = (cur - prev) / prev * 100
                if abs(delta) > 20 and cur > 10:
                    trend = "🔺" if delta > 0 else "🔻"
                    insights.append(
                        f"{trend} {cat} {delta:+.0f}% em {bairro} vs 4 semanas: {'clima seco?' if cat.lower()=='graffiti' else 'atenção!'}"
                    )
    if not insights:
        insights.append("ℹ️ Nenhuma variação significativa detectada nas últimas semanas.")
    return insights

# --- Template rendering ---
def render_templates(insights, charts, output_html):
    # Prepare chart images
    tmp_dir = tempfile.mkdtemp()
    bar_path = Path(tmp_dir) / "bar_top10.png"
    charts['bar_top10'].write_image(str(bar_path), width=900, height=600, scale=2)

    # Folium map to PNG via HTML screenshot
    fmap_path = Path(tmp_dir) / "heatmap.html"
    charts['heatmap'].save(str(fmap_path))
    fmap_png = Path(tmp_dir) / "heatmap.png"
    # Use headless Chrome to screenshot the map
    options = Options()
    options.headless = True
    options.add_argument("--window-size=900,600")
    driver = webdriver.Chrome(options=options)
    driver.get(f"file://{fmap_path.resolve()}")
    driver.save_screenshot(str(fmap_png))
    driver.quit()

    # Jinja2 render
    env = Environment(
        loader=FileSystemLoader("templates"),
        autoescape=select_autoescape(['html'])
    )
    template = env.get_template("resumo.html")
    html = template.render(
        insights=insights,
        bar_chart=str(bar_path),
        heatmap=str(fmap_png),
        palette={"blue": "#007BFF", "green": "#28A745"},
        font="Arial"
    )
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html)
    shutil.rmtree(tmp_dir)
    return html

# --- Social export ---
def export_social(html_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Use headless browser to render HTML and screenshot
    options = Options()
    options.headless = True
    options.add_argument("--window-size=1080,1350")
    driver = webdriver.Chrome(options=options)
    driver.get(f"file://{Path(html_path).resolve()}")
    for i in range(3):  # Assume 3 slides: insights, bar, heatmap
        slide_path = output_dir / f"carrossel_{i+1}.png"
        driver.save_screenshot(str(slide_path))
        # Add alt-text (not in PNG, but for metadata)
        img = Image.open(slide_path)
        img.info['Description'] = "Gráfico de chamados 311 em Vancouver"
        img.save(slide_path)
    driver.quit()
    # Hashtags (for user to copy)
    hashtags = "#Vancouver311 #UrbanInsights"
    logging.info("Social export done. Hashtags: %s", hashtags)

# --- Save outputs ---
def save_outputs(pdf_path, social_path, html_content):
    pdf_path = Path(pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    HTML(string=html_content).write_pdf(str(pdf_path))
    logging.info("PDF salvo em %s", pdf_path)
    # Social PNGs já salvos em export_social

# --- Main ---
def main():
    parser = argparse.ArgumentParser(description="Gera relatório Vancouver 311")
    parser.add_argument("--features", required=True, help="Arquivo features.parquet")
    parser.add_argument("--metrics", required=True, help="Arquivo eval_metrics.json")
    parser.add_argument("--model", required=True, help="Arquivo modelo.pkl")
    parser.add_argument("--output-pdf", required=True, help="Caminho PDF de saída")
    parser.add_argument("--output-social", required=True, help="Dir para PNGs carrossel")
    args = parser.parse_args()

    # Validação
    try:
        InputPaths(features=args.features, metrics=args.metrics, model=args.model)
    except ValidationError as e:
        logging.error("Erro de validação: %s", e)
        print("Erro de validação nos caminhos dos arquivos.")
        return

    df, metrics, model = load_data(args.features, args.metrics, args.model)
    charts = generate_charts(df)
    insights = create_insights(df)
    tmp_html = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    html_content = render_templates(insights, charts, tmp_html.name)
    save_outputs(args.output_pdf, args.output_social, html_content)
    export_social(tmp_html.name, args.output_social)
    logging.info("Relatório gerado com sucesso.")

if __name__ == "__main__":
    main()
