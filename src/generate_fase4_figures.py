"""
src/generate_fase4_figures.py
==============================
Genera las figuras analíticas y de arquitectura específicas para la Fase 4
y consolida las figuras clave de las fases anteriores para el Informe Final en LaTeX.
"""

import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

def generate_figures():
    output_dir = "reports/fase4/figures"
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # 1. Copiar figuras clave de Fases 1, 2 y 3
    source_figures = {
        "reports/fase1/figures/fig1_univariate_numerical.png": "fig_eda_univariate.png",
        "reports/fase1/figures/fig3_correlation_matrices.png": "fig_eda_correlations.png",
        "reports/fase2/figures/fig1_missing_mar_tenure.png": "fig_missing_mar.png",
        "reports/fase2/figures/fig3_engineered_features.png": "fig_engineered_features.png",
        "reports/fase3/figures/fig1_regression_comparison.png": "fig_regression_comparison.png",
        "reports/fase3/figures/fig2_classification_roc_pr.png": "fig_classification_roc_pr.png",
        "reports/fase3/figures/fig3_pytorch_loss_curve.png": "fig_pytorch_loss_curve.png",
        "reports/fase3/figures/fig4_clustering_elbow_silhouette.png": "fig_clustering_elbow_silhouette.png",
        "reports/fase3/figures/fig5_cluster_profiles.png": "fig_cluster_profiles.png",
        "reports/fase3/figures/fig6_shap_summary.png": "fig_shap_summary.png"
    }

    for src, dst in source_figures.items():
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(output_dir, dst))
            print(f"[OK] Copiado {src} -> {dst}")

    # 2. Generar Figura: Diagrama de Arquitectura MLOps y Ciclo CRISP-DM
    fig, ax = plt.subplots(figsize=(13, 6.2), dpi=300)
    ax.axis('off')

    stages = [
        {"title": "1. Ingesta & Auditoría", "desc": "Dataset Telco (7,043)\nAuditoría MAR en TotalCharges\nImputación Tenure=0 -> $0.0", "color": "#002B5B", "x": 0.05},
        {"title": "2. Preprocesamiento", "desc": "Ingeniería de 4 variables\nColumnTransformer estricto\nSplit Estratificado (70/15/15)\nSin Fuga de Datos (Leakage)", "color": "#014F86", "x": 0.28},
        {"title": "3. Modelado Avanzado", "desc": "XGBoost & Random Forest\nRed Neuronal PyTorch (BCE)\nClustering K-Means (K=3)\nExplicabilidad SHAP (XAI)", "color": "#2A6F97", "x": 0.51},
        {"title": "4. Serialización & MLOps", "desc": "Pipelines Joblib en disco\nDataset Enriquecido CSV\nMétricas consolidadas JSON\nVersionamiento Git/GitHub", "color": "#C5A059", "x": 0.74},
        {"title": "5. Dashboard Streamlit", "desc": "Filtros dinámicos reactivos\nTarjetas KPI ejecutivas\nSimulador en vivo Churn/Valor\nProyección PCA 2D/3D", "color": "#28A745", "x": 0.97}
    ]

    for i, s in enumerate(stages):
        # Cuadro principal
        rect = patches.FancyBboxPatch(
            (s['x'] - 0.09, 0.25), 0.17, 0.52,
            boxstyle="round,pad=0.03,rounding_size=0.04",
            facecolor=s['color'], edgecolor='none', alpha=0.92
        )
        ax.add_patch(rect)

        # Texto del título
        ax.text(s['x'] - 0.005, 0.69, s['title'], ha='center', va='center', color='white', fontweight='bold', fontsize=10.5)
        # Línea divisoria
        ax.plot([s['x'] - 0.075, s['x'] + 0.065], [0.63, 0.63], color='white', lw=1, alpha=0.8)
        # Descripción
        ax.text(s['x'] - 0.005, 0.44, s['desc'], ha='center', va='center', color='#F8F9FA', fontsize=8.2, linespacing=1.4)

        # Flecha conectora
        if i < len(stages) - 1:
            ax.annotate(
                '', xy=(stages[i+1]['x'] - 0.095, 0.51), xytext=(s['x'] + 0.085, 0.51),
                arrowprops=dict(facecolor='#002B5B', edgecolor='none', width=2.5, headwidth=8.0)
            )

    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(0.15, 0.85)
    plt.title("Arquitectura del Sistema Analítico Integral de Ciencia de Datos y MLOps (Fases 1 a 4)", fontsize=13, fontweight='bold', pad=15, color='#002B5B')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig_mlops_architecture.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("[OK] Generada 'fig_mlops_architecture.png'")

    # 3. Generar Figura: Matriz Estratégica de Negocio (Valor de Cliente vs Riesgo de Churn)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    # Segmentos K-Means
    clusters_info = [
        {"name": "Clúster 0: Premium Fidelizado", "x": 58.5, "y": 15.0, "size": 2185, "color": "#002B5B", "desc": "Alta permanencia (58.5 m)\nFactura ~$90 USD\nAcción: Lealtad & VIP"},
        {"name": "Clúster 1: Nuevo en Alto Riesgo", "x": 15.2, "y": 45.4, "size": 2714, "color": "#DC3545", "desc": "Baja permanencia (15.2 m)\nFactura ~$75 USD\nAcción: Campaña de Rescate"},
        {"name": "Clúster 2: Económico Básico", "x": 27.6, "y": 14.5, "size": 2144, "color": "#C5A059", "desc": "Permanencia media (27.6 m)\nFactura ~$26 USD\nAcción: Venta Cruzada (Cross-Sell)"}
    ]

    for c in clusters_info:
        ax.scatter(c['x'], c['y'], s=c['size']/1.8, color=c['color'], alpha=0.6, edgecolors='black', linewidth=1.5, label=f"{c['name']} (N={c['size']:,})")
        ax.text(c['x'], c['y'] + 2.5, c['desc'], ha='center', va='bottom', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=c['color'], alpha=0.9))

    ax.axhline(26.54, color='gray', linestyle='--', alpha=0.7, label='Tasa Global Churn (26.5%)')
    ax.set_xlabel("Antigüedad Media del Segmento (Meses)", fontweight='bold', fontsize=11)
    ax.set_ylabel("Tasa de Deserción Observada (% Churn)", fontweight='bold', fontsize=11)
    ax.set_title("Matriz de Segmentación Profiláctica y Protocolos de Intervención de Negocio", fontweight='bold', fontsize=12, pad=15)
    ax.set_xlim(5, 75)
    ax.set_ylim(5, 60)
    ax.legend(frameon=True, loc='upper right')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig_prescriptive_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("[OK] Generada 'fig_prescriptive_matrix.png'")

if __name__ == "__main__":
    generate_figures()
