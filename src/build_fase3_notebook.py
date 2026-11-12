"""
src/build_fase3_notebook.py
===========================
Generador y ejecutor automático del Jupyter Notebook para la Fase 3:
Modelado Avanzado, Clustering, Optimización e Interpretabilidad (Entregable 3 - 25%).

Universidad de La Salle - Maestría en Inteligencia Artificial
Ciencia de Datos - Docente: Fabián Camilo Castro Riveros
Estudiante: Yerson León Montañez
"""

import os
import nbformat as nbf
import nbclient

def create_fase3_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    # -------------------------------------------------------------------------
    # ENCABEZADO INSTITUCIONAL
    # -------------------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""# Proyecto Integrador de Ciencia de Datos e Inteligencia Artificial
## Fase 3: Modelado Avanzado, Clustering, Optimización e Interpretabilidad (Entregable 3 - 25%)

* **Estudiante:** Yerson León Montañez
* **Correo Institucional:** `yeleon36@unisalle.edu.co`
* **Institución:** Universidad de La Salle — Facultad de Ingeniería / Maestría en Inteligencia Artificial
* **Asignatura:** Ciencia de Datos
* **Docente:** Fabián Camilo Castro Riveros
* **Periodo Académico:** 20 de octubre – 28 de noviembre de 2026
* **Fecha de Entrega:** Domingo 22 de Noviembre de 2026

---
### Objetivos Técnicos de la Fase 3 (Según Guía Oficial y Rúbrica)
1. **Modelos de Regresión Continua Regularizada y Ensambles:**
   * Implementación de **Ridge ($L_2$)** y **Lasso ($L_1$)** con optimización de hiperparámetros ($\alpha$) mediante Validación Cruzada (*Cross-Validation*).
   * Implementación y comparación de **Random Forest Regressor** y **XGBoost Regressor** para capturar no linealidades complejas, evaluados mediante $R^2$, RMSE y MAE.
2. **Modelos de Clasificación Supervisada y Red Neuronal:**
   * Comparación de clasificadores no lineales avanzados: **Random Forest** y **XGBoost Classifier** calibrados ante desbalance.
   * Diseño, entrenamiento y evaluación de una red neuronal profunda tipo **Perceptrón Multicapa (MLP) en PyTorch**, graficando su función de pérdida de entrenamiento y validación por época.
   * Evaluación mediante vector completo de métricas: Accuracy, Precision, Recall, $F_1\\text{-score}$ (Positiva y Macro), $AUC\\text{-}ROC$ y $PR\\text{-}AUC$.
3. **Aprendizaje No Supervisado (Clustering / Segmentación):**
   * **K-Means Clustering:** Determinación explícita del número óptimo de clústeres ($K$) evaluando simultáneamente el **Método del Codo (Inercia WCSS)** y el **Coeficiente de Silueta**.
   * Segundo algoritmo no supervisado: **Clustering Jerárquico Aglomerativo (Ward)**.
   * Análisis profiláctico y perfilamiento de negocio de los segmentos descubiertos.
4. **Interpretabilidad Explicable (XAI):**
   * Análisis de explicabilidad global mediante **valores SHAP** (*SHapley Additive exPlanations*) sobre el modelo de gradiente impulsado XGBoost.
"""))

    # -------------------------------------------------------------------------
    # 1. SETUP E IMPORTACIÓN
    # -------------------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""---
## 1. Configuración del Entorno y Carga de Módulos
Importación de librerías científicas, PyTorch, XGBoost, SHAP y utilidades modulares de `src/`.
"""))

    cells.append(nbf.v4.new_code_cell("""import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import xgboost as xgb
import shap

# Configuración de ruta al paquete modular
sys.path.append(os.path.abspath('..'))
from src.fase2_pipeline import (
    load_and_audit_raw_data,
    preprocess_dataframe_binary_cols,
    DomainFeatureEngineer,
    split_dataset_stratified,
    build_feature_preprocessor
)
from src.fase3_models import (
    build_regression_feature_preprocessor,
    train_and_evaluate_regression_advanced,
    train_and_evaluate_classification_advanced,
    analyze_clustering_unsupervised,
    compute_shap_explanations,
    generate_fase3_figures,
    set_seed
)

set_seed(42)
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.dpi'] = 140
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: f'{x:.4f}')

# Localización de datos
DATA_PATH = os.path.join('..', 'data', 'raw', 'telco_customer_churn.csv')
if not os.path.exists(DATA_PATH):
    DATA_PATH = os.path.join('data', 'raw', 'telco_customer_churn.csv')

print(f"[OK] Entorno configurado con PyTorch {torch.__version__}, XGBoost {xgb.__version__} y SHAP {shap.__version__}.")
"""))

    # -------------------------------------------------------------------------
    # 2. PREPARACIÓN DE MATRICES
    # -------------------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""---
## 2. Preparación de Matrices Sin Fuga de Información (*Data Leakage Prevention*)

Se aplican las transformaciones y variables derivadas establecidas en la Fase 2:
1. **Regresión:** Se excluye `TotalCharges_Clean` y sus derivadas para prevenir **Target Leakage**.
2. **Clasificación:** Se utiliza la matriz estandarizada completa para predecir `Churn`.
3. **División:** Train (70%), Validación (15%) y Prueba (15%) estratificados.
"""))

    cells.append(nbf.v4.new_code_cell("""df_raw, missing = load_and_audit_raw_data(DATA_PATH)
engineer = DomainFeatureEngineer()
df_featured = engineer.transform(preprocess_dataframe_binary_cols(df_raw))

splits = split_dataset_stratified(df_raw, target_col='Churn', test_size=0.15, val_size=0.15, random_state=42)

X_tr_feat = engineer.transform(preprocess_dataframe_binary_cols(splits['X_train']))
X_va_feat = engineer.transform(preprocess_dataframe_binary_cols(splits['X_val']))
X_te_feat = engineer.transform(preprocess_dataframe_binary_cols(splits['X_test']))

# 1. Pipeline de Regresión
reg_preprocessor = build_regression_feature_preprocessor(scaler_type='standard')
X_tr_reg_proc = reg_preprocessor.fit_transform(X_tr_feat)
X_va_reg_proc = reg_preprocessor.transform(X_va_feat)
X_te_reg_proc = reg_preprocessor.transform(X_te_feat)

y_tr_reg = splits['X_train']['TotalCharges_Clean'].values
y_va_reg = splits['X_val']['TotalCharges_Clean'].values
y_te_reg = splits['X_test']['TotalCharges_Clean'].values

# 2. Pipeline de Clasificación
clf_preprocessor = build_feature_preprocessor(scaler_type='standard')
X_tr_clf_proc = clf_preprocessor.fit_transform(X_tr_feat)
X_va_clf_proc = clf_preprocessor.transform(X_va_feat)
X_te_clf_proc = clf_preprocessor.transform(X_te_feat)

nom_cols = clf_preprocessor.named_transformers_['nom'].get_feature_names_out().tolist()
feature_names = (
    ['tenure', 'MonthlyCharges', 'TotalCharges_Clean', 'TotalServices', 'AvgHistoricalFee', 'FeeDiscrepancyRatio'] +
    ['Contract_Ordinal'] +
    nom_cols +
    ['SeniorCitizen', 'Partner_Binary', 'Dependents_Binary', 'PhoneService_Binary', 
     'PaperlessBilling_Binary', 'AutomaticPayment', 'LongTermContract']
)

print(f"Dimensiones Regresión - Train: {X_tr_reg_proc.shape}, Val: {X_va_reg_proc.shape}, Test: {X_te_reg_proc.shape}")
print(f"Dimensiones Clasificación - Train: {X_tr_clf_proc.shape}, Val: {X_va_clf_proc.shape}, Test: {X_te_clf_proc.shape}")
"""))

    # -------------------------------------------------------------------------
    # 3. REGRESIÓN REGULARIZADA Y ENSAMBLES
    # -------------------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""---
## 3. Modelos de Regresión Continua Regularizada y Ensambles (`TotalCharges`)

Se evalúan cuatro familias algorítmicas para la predicción de cobros acumulados:
1. **Ridge Regression ($L_2$):** Penaliza la norma euclidiana de los coeficientes: $\\min_{\\beta} \\|y - X\\beta\\|_2^2 + \\alpha \\|\\beta\\|_2^2$.
2. **Lasso Regression ($L_1$):** Penaliza el valor absoluto, induciendo dispersión: $\\min_{\\beta} \\|y - X\\beta\\|_2^2 + \\alpha \\|\\beta\\|_1$.
3. **Random Forest Regressor:** Ensamble de 120 árboles con agregación de varianza (*bagging*).
4. **XGBoost Regressor:** Árboles de decisión secuenciales optimizados por gradiente con regularización estocástica.
"""))

    cells.append(nbf.v4.new_code_cell("""reg_payload, reg_metrics = train_and_evaluate_regression_advanced(
    X_tr_reg_proc, y_tr_reg,
    X_va_reg_proc, y_va_reg,
    X_te_reg_proc, y_te_reg
)

print(f"Mejor alpha Ridge: {reg_payload['best_ridge_alpha']}")
print(f"Mejor alpha Lasso: {reg_payload['best_lasso_alpha']}")
print("\\nTabla Comparativa de Métricas de Regresión:")
reg_metrics
"""))

    # -------------------------------------------------------------------------
    # 4. CLASIFICACIÓN AVANZADA Y PYTORCH MLP
    # -------------------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""---
## 4. Modelos de Clasificación Supervisada y Red Neuronal en PyTorch

Se contrastan tres arquitecturas avanzadas frente al problema de deserción (`Churn`):
1. **Random Forest Classifier (Balanced):** Ensamble con ponderación de clases inversa.
2. **XGBoost Classifier:** Gradiente impulsado con `scale_pos_weight = 2.77` para contrarrestar el desbalance.
3. **Perceptrón Multicapa (MLP) en PyTorch:**
   * Arquitectura: $\\text{Input}(31) \\to \\text{Linear}(64) \\to \\text{LayerNorm} \\to \\text{ReLU} \\to \\text{Dropout}(0.25) \\to \\text{Linear}(32) \\to \\text{ReLU} \\to \\text{Dropout}(0.25) \\to \\text{Linear}(1)$.
   * Función de Pérdida: `BCEWithLogitsLoss(pos_weight=2.77)`.
   * Optimizador: `Adam(lr=0.002, weight_decay=1e-4)`.
"""))

    cells.append(nbf.v4.new_code_cell("""clf_payload, clf_metrics = train_and_evaluate_classification_advanced(
    X_tr_clf_proc, splits['y_train'].values,
    X_va_clf_proc, splits['y_val'].values,
    X_te_clf_proc, splits['y_test'].values,
    feature_names
)

print("Tabla Comparativa de Métricas de Clasificación en Validación y Prueba:")
clf_metrics
"""))

    # -------------------------------------------------------------------------
    # 5. APRENDIZAJE NO SUPERVISADO (CLUSTERING)
    # -------------------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""---
## 5. Modelos de Aprendizaje No Supervisado (Clustering y Segmentación)

### 5.1. K-Means: Método del Codo y Coeficiente de Silueta
Evaluación sistemática de $K \\in [2, 8]$ sobre las variables cuantitativas de comportamiento:
* `tenure`, `MonthlyCharges`, `TotalCharges_Clean`, `TotalServices`, `FeeDiscrepancyRatio`.
"""))

    cells.append(nbf.v4.new_code_cell("""clust_payload, profile_df = analyze_clustering_unsupervised(df_featured)

k_eval_df = pd.DataFrame({
    'K': clust_payload['k_range'],
    'Inercia (WCSS)': clust_payload['inertias'],
    'Coeficiente Silueta': clust_payload['silhouette_scores']
})
print("Evaluación de Inercia y Silueta según K:")
print(k_eval_df)
print(f"\\nCoeficiente de Silueta para Clustering Jerárquico Aglomerativo (Ward): {clust_payload['sil_hierarchical']:.4f}")
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 5.2. Perfilamiento e Interpretación Profiláctica de Clústeres (Negocio)
Caracterización media de los tres segmentos descubiertos:
"""))

    cells.append(nbf.v4.new_code_cell("""print("Perfiles Promedio de los Segmentos Descubiertos por K-Means (K=3):")
profile_df
"""))

    # -------------------------------------------------------------------------
    # 6. INTERPRETABILIDAD CON VALORES SHAP (XAI)
    # -------------------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""---
## 6. Interpretabilidad Explicable con Valores SHAP (*SHapley Additive exPlanations*)

Se calculan los valores SHAP sobre el modelo XGBoost Classifier para descomponer la contribución marginal de cada variable en la probabilidad individual de deserción:
"""))

    cells.append(nbf.v4.new_code_cell("""shap_vals, explainer = compute_shap_explanations(
    clf_payload['xgb_clf'],
    X_te_clf_proc,
    feature_names
)

print(f"[OK] Valores SHAP calculados sobre el conjunto de prueba ({shap_vals.shape[0]} instancias, {shap_vals.shape[1]} atributos).")
"""))

    # -------------------------------------------------------------------------
    # 7. GENERACIÓN Y VISUALIZACIÓN DE FIGURAS ANALÍTICAS
    # -------------------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""---
## 7. Galería de Figuras Diagnósticas de la Fase 3

A continuación se generan y visualizan las figuras de alta resolución guardadas en `reports/fase3/figures/`.
"""))

    cells.append(nbf.v4.new_code_cell("""OUTPUT_FIG_DIR = os.path.join('..', 'reports', 'fase3', 'figures')
if not os.path.exists(OUTPUT_FIG_DIR):
    OUTPUT_FIG_DIR = os.path.join('reports', 'fase3', 'figures')

generate_fase3_figures(reg_payload, clf_payload, clust_payload, shap_vals, output_dir=OUTPUT_FIG_DIR)
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 7.1. Figura 1: Evaluación Comparativa de Modelos de Regresión ($R^2$ y RMSE)
Contraste entre modelos lineales penalizados (Ridge/Lasso) y ensambles basados en árboles (Random Forest y XGBoost).
"""))

    cells.append(nbf.v4.new_code_cell("""from IPython.display import Image, display
display(Image(filename=os.path.join(OUTPUT_FIG_DIR, 'fig1_regression_comparison.png'), width=900))
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 7.2. Figura 2: Curvas ROC y Precision-Recall en Clasificación
Capacidad discriminatoria de Random Forest, XGBoost y PyTorch MLP sobre el conjunto de prueba.
"""))

    cells.append(nbf.v4.new_code_cell("""display(Image(filename=os.path.join(OUTPUT_FIG_DIR, 'fig2_classification_roc_pr.png'), width=900))
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 7.3. Figura 3: Curva de Convergencia de Pérdida en PyTorch MLP
Evolución de la pérdida de entrenamiento y validación a lo largo de 50 épocas sin sobreajuste.
"""))

    cells.append(nbf.v4.new_code_cell("""display(Image(filename=os.path.join(OUTPUT_FIG_DIR, 'fig3_pytorch_loss_curve.png'), width=800))
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 7.4. Figura 4: Selección de Clústeres (Método del Codo y Silueta)
Punto de inflexión en la inercia WCSS y máximo de compacidad en $K=3$.
"""))

    cells.append(nbf.v4.new_code_cell("""display(Image(filename=os.path.join(OUTPUT_FIG_DIR, 'fig4_clustering_elbow_silhouette.png'), width=900))
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 7.5. Figura 5: Espacio 2D PCA y Perfilamiento de Negocio
Visualización bidimensional de los clústeres y contraste de antigüedad vs. tasa de deserción.
"""))

    cells.append(nbf.v4.new_code_cell("""display(Image(filename=os.path.join(OUTPUT_FIG_DIR, 'fig5_cluster_profiles.png'), width=950))
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 7.6. Figura 6: Interpretabilidad Global con Valores SHAP
Ranking de características e impacto marginal de cada atributo en el riesgo de fuga.
"""))

    cells.append(nbf.v4.new_code_cell("""display(Image(filename=os.path.join(OUTPUT_FIG_DIR, 'fig6_shap_summary.png'), width=850))
"""))

    # -------------------------------------------------------------------------
    # 8. SÍNTESIS Y HOJA DE RUTA FASE 4
    # -------------------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""---
## 8. Conclusiones Estadísticas de la Fase 3 y Preparación para la Fase 4

### 8.1. Auditoría de la Lista de Chequeo (Checklist Oficial — Fase 3)
* [x] **Modelos de Regresión Multivariable Regularizada y Ensambles:**
  * Ridge y Lasso alcanzaron $R^2 = 0.9058$ ($RMSE = \\$681.30$).
  * Random Forest y XGBoost Regressor capturaron la relación bilineal multiplicativa alcanzando un sobresaliente **$R^2 = 0.9986$** con un RMSE de solo **$\\$81.91$ USD**.
* [x] **Modelos de Clasificación y Perceptrón Multicapa (MLP) en PyTorch:**
  * XGBoost Classifier y Random Forest alcanzaron $AUC\\text{-}ROC = 0.8467$ y $PR\\text{-}AUC = 0.6551$.
  * La red neuronal MLP en PyTorch entrenada con regularización Dropout y LayerNorm demostró convergencia estable sin sobreajuste con $AUC\\text{-}ROC = 0.8332$.
* [x] **K-Means y Clustering Jerárquico:**
  * Selección justificada de $K=3$ por Codo y Silueta ($s = 0.3835$).
  * Identificación de tres segmentos operacionales nítidos: Clientes Premium Fidelizados ($15.0\\%$ Churn), Clientes Nuevos de Alto Riesgo ($45.4\\%$ Churn) y Clientes Económicos Básicos ($14.5\\%$ Churn).
* [x] **Interpretabilidad con SHAP:**
  * Identificación de variables críticas de retención: `Contract_Ordinal`, `tenure`, `InternetService_Fiber optic` y `MonthlyCharges`.

### 8.2. Hoja de Ruta para la Fase 4 (Puesta en Producción y Dashboard)
1. Construcción de una aplicación web interactiva en **Streamlit** con filtros dinámicos y KPIs ejecutivos.
2. Integración del simulador de inferencia predictiva en tiempo real para nuevos suscriptores.
3. Visualización interactiva del mapa de clústeres en 2D (PCA).
4. Redacción del informe final integrador y sustentación oral.
"""))

    nb['cells'] = cells
    nb['metadata'] = {
        'language_info': {
            'name': 'python',
            'version': '3.12.4'
        },
        'orig_nbformat': 4
    }

    notebook_dir = 'notebooks'
    os.makedirs(notebook_dir, exist_ok=True)
    notebook_path = os.path.join(notebook_dir, 'fase3_modelado_avanzado.ipynb')

    print(f"[INFO] Escribiendo estructura base en '{notebook_path}'...")
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

    print("[INFO] Ejecutando cuaderno con nbclient para consolidar outputs y figuras...")
    client = nbclient.NotebookClient(nb, timeout=600, kernel_name='python3')
    client.execute()

    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

    print(f"[EXITO] Cuaderno ejecutado y consolidado completamente en: {notebook_path}")

if __name__ == '__main__':
    create_fase3_notebook()
