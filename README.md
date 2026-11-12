# 🏛️ Proyecto Integrador de Ciencia de Datos e Inteligencia Artificial
### Sistema Inteligente de Predicción de Deserción (Churn), Estimación de Valor Económico y Segmentación Profiláctica

[![Python Version](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Framework Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B.svg)](https://streamlit.io/)
[![Machine Learning](https://img.shields.io/badge/Scikit--Learn-1.5+-F7931E.svg)](https://scikit-learn.org/)
[![Gradient Boosting](https://img.shields.io/badge/XGBoost-2.0+-green.svg)](https://xgboost.readthedocs.io/)
[![Deep Learning](https://img.shields.io/badge/PyTorch-2.2+-EE4C2C.svg)](https://pytorch.org/)
[![Explainability](https://img.shields.io/badge/XAI-SHAP-brightgreen.svg)](https://shap.readthedocs.io/)
[![LaTeX Report](https://img.shields.io/badge/Report-LaTeX_PDF-002B5B.svg)](reports/fase4/main.pdf)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

---

## 📌 Información Institucional
* **Institución:** Universidad de La Salle
* **Facultad:** Facultad de Ingeniería — Departamento de Ciencias Básicas
* **Programa:** Maestría en Inteligencia Artificial
* **Asignatura:** Ciencia de Datos
* **Docente Titular:** Fabián Camilo Castro Riveros
* **Estudiante / Autor:** Yerson León Montañez (`yeleon36@unisalle.edu.co`)
* **Periodo Académico:** 2026-II
* **Repositorio Oficial:** [https://github.com/yhleonm/proyecto_ciencia_datos_la_salle](https://github.com/yhleonm/proyecto_ciencia_datos_la_salle)

---

## 📖 Resumen Ejecutivo del Proyecto
En la industria de las telecomunicaciones, el Costo de Adquisición de Clientes (CAC) supera entre 5 y 7 veces el costo de retener un suscriptor activo. Este proyecto implementa un ciclo de vida analítico riguroso bajo el estándar **CRISP-DM / MLOps** sobre el dataset **Telco Customer Churn** ($7{,}043$ observaciones y $21$ variables), neutralizando sistemáticamente la fuga de datos (*Data Leakage*) y resolviendo el mecanismo de ausencia de datos condicionado **MAR (Missing at Random)** en los cobros acumulados.

### Componentes de la Solución Analítica:
1. **Regresión Continua Multivariada:** Predicción de facturación acumulada (`TotalCharges`) para cuantificar el valor del cliente (CLV).
2. **Clasificación Supervisada Sensible al Costo:** Detección probabilística de deserción (`Churn`) equilibrando el desbalance de clases ($26.54\%$ positivos) mediante factor de penalización $w_{\text{pos}} = 2.77$.
3. **Redes Neuronales Profundas (PyTorch MLP):** Perceptrón Multicapa con normalización por capas (`LayerNorm`), `Dropout` y convergencia monótona con `Weighted BCE Loss`.
4. **Aprendizaje No Supervisado (Clustering K-Means & Jerárquico):** Segmentación profiláctica en $K=3$ perfiles de negocio validada matemáticamente con el Método del Codo y Coeficiente de Silueta ($s = 0.3835$).
5. **Explicabilidad Axiomática (XAI con SHAP):** Cuantificación del impacto marginal de cada característica basada en la teoría cooperativa de juegos de Shapley.
6. **Dashboard Interactivo en Streamlit:** Aplicación web con filtros dinámicos reactivos, tarjetas KPI directivas, simulador predictivo en vivo y proyecciones PCA 2D/3D con Plotly.

---

## 📊 Matriz Consolidada de Resultados Técnicos

### 1. Modelos de Regresión (TotalCharges en USD)
| Modelo | Partición | $R^2$ Score | RMSE (USD) | MAE (USD) | Diagnóstico Metodológico |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Baseline Regresión Lineal (Fase 2)** | Test | $0.9037$ | $\$688.85$ | $\$558.12$ | Modelo de referencia lineal sin regularización. |
| **Ridge Regression ($L_2$, $\alpha=1.0$)** | Test | $0.9058$ | $\$681.38$ | $\$539.52$ | Regularización cuadrática contra colinealidad. |
| **Lasso Regression ($L_1$, $\alpha=0.5$)** | Test | $0.9058$ | $\$681.30$ | $\$539.45$ | Selección parsimoniosa de atributos. |
| **XGBoost Regressor** | Test | $0.9980$ | $\$99.24$ | $\$71.86$ | Árboles potenciados con interacción no lineal. |
| 🏆 **Random Forest Regressor (Final)** | **Test** | $\mathbf{0.9986}$ | $\mathbf{\$81.91}$ | $\mathbf{\$55.88}$ | **Reducción del error cuadrático en un $88.1\%$.** |

### 2. Modelos de Clasificación (Predicción de Churn)
| Modelo | Partición | Accuracy | Precision | Recall | $F_1$ (Pos) | AUC-ROC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Regresión Logística (Fase 2)** | Test | $0.7417$ | $0.5078$ | $\mathbf{0.8107}$ | $0.6245$ | $0.8492$ | $0.6415$ |
| **Random Forest Classifier (Balanced)** | Test | $0.7758$ | $\mathbf{0.5518}$ | $0.7679$ | $0.6418$ | $0.8466$ | $0.6539$ |
| 🏆 **XGBoost Classifier (Final)** | **Test** | $\mathbf{0.7720}$ | $0.5457$ | $\mathbf{0.7821}$ | $\mathbf{0.6432}$ | $\mathbf{0.8467}$ | $\mathbf{0.6551}$ |
| **PyTorch MLP (50 Épocas, Weighted BCE)** | Test | $0.7483$ | $0.5165$ | $0.8036$ | $0.6285$ | $0.8332$ | $0.6260$ |

### 3. Segmentación No Supervisada de Clientes ($K=3$)
| Clúster | Nombre de Negocio | N Clientes | Antigüedad | Tarifa Mes | Cobro Total | % Pago Auto | % Churn | Protocolo Prescriptivo |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **0** | **Premium Fidelizado** | $2{,}185$ | $58.5$ m | $\$90.07$ | $\$5{,}252.41$ | $62.5\%$ | **$15.0\%$** | Atención posventa VIP y renovación de routers. |
| **1** | **Nuevo en Alto Riesgo** | $2{,}714$ | $15.2$ m | $\$74.94$ | $\$1{,}131.34$ | $30.6\%$ | **$45.4\%$** | **Foco Crítico:** Descuento por migración a 12 meses. |
| **2** | **Económico Básico** | $2{,}144$ | $27.6$ m | $\$26.09$ | $\$703.92$ | $40.6\%$ | **$14.5\%$** | Venta cruzada (*cross-sell*) de seguridad digital. |

---

## 🏗️ Estructura del Repositorio

El repositorio sigue estrictamente el estándar de proyectos reproducibles exigido por la guía de la Universidad de La Salle:

```bash
proyecto_ciencia_datos_la_salle/
├── data/
│   ├── raw/
│   │   └── telco_customer_churn.csv              # Dataset original público (7,043 registros)
│   └── processed/
│       └── telco_churn_dashboard_data.csv        # Dataset enriquecido con predicciones y PCA
├── notebooks/
│   ├── fase1_eda.ipynb                           # Cuaderno EDA exhaustivo univariado/multivariado
│   ├── fase2_preprocesamiento_baseline.ipynb     # Cuaderno de limpieza, ingeniería y baselines
│   └── fase3_modelado_avanzado.ipynb             # Cuaderno de regresión, XGBoost, PyTorch y K-Means
├── src/
│   ├── __init__.py
│   ├── eda_utils.py                              # Utilidades gráficas y estadísticas de la Fase 1
│   ├── fase2_pipeline.py                         # Pipelines Scikit-Learn y prevención de leakage
│   ├── fase3_models.py                           # Modelos avanzados, arquitectura PyTorch y XAI
│   ├── export_fase4_artifacts.py                 # Script MLOps de serialización de artefactos
│   └── generate_fase4_figures.py                 # Generador de figuras de alta resolución
├── dashboard/
│   ├── app.py                                    # Aplicación interactiva principal en Streamlit
│   └── models/                                   # Artefactos serializados (Joblib y JSON)
│       ├── clf_preprocessor.joblib               # Preprocesador de clasificación ajustado en Train
│       ├── xgb_classifier.joblib                 # Modelo final XGBoost Classifier
│       ├── rf_classifier.joblib                  # Modelo Random Forest Classifier
│       ├── reg_preprocessor.joblib               # Preprocesador de regresión sin Target Leakage
│       ├── rf_regressor.joblib                   # Modelo final Random Forest Regressor
│       ├── xgb_regressor.joblib                  # Modelo XGBoost Regressor
│       ├── cluster_artifacts.joblib              # K-Means (K=3), Scaler y PCA (2D/3D)
│       └── metrics_summary.json                  # Resumen cuantitativo de métricas
├── reports/
│   ├── fase1/                                    # Entregable Técnico 1 (LaTeX y PDF compilado)
│   ├── fase2/                                    # Entregable Técnico 2 (LaTeX y PDF compilado)
│   ├── fase3/                                    # Entregable Técnico 3 (LaTeX y PDF compilado)
│   └── fase4/                                    # Entregable Técnico Final (LaTeX y PDF compilado)
│       ├── main.tex                              # Código fuente en LaTeX (artículo formal)
│       ├── main.pdf                              # Informe Técnico Final compilado (15 páginas)
│       └── figures/                              # Figuras vectoriales e infografías de Fase 4
├── requirements.txt                              # Dependencias con versiones fijas reproducibles
├── .gitignore                                    # Exclusión de binarios temporales y caches
└── README.md                                     # Documentación ejecutiva principal
```

---

## 🚀 Guía de Instalación y Ejecución Local

### 1. Clonar el Repositorio
```bash
git clone https://github.com/yhleonm/proyecto_ciencia_datos_la_salle.git
cd proyecto_ciencia_datos_la_salle
```

### 2. Configurar Entorno Virtual (Recomendado)
```bash
# En Windows (PowerShell):
python -m venv venv
.\venv\Scripts\Activate.ps1

# En Linux / macOS:
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar Dependencias
```bash
pip install -r requirements.txt
```

### 4. Re-exportar Artefactos MLOps (Opcional)
Si deseas regenerar los modelos serializados y el dataset enriquecido desde cero:
```bash
python -m src.export_fase4_artifacts
```

### 5. Lanzar el Dashboard Interactivo de Streamlit
```bash
streamlit run dashboard/app.py
```
La aplicación se abrirá automáticamente en tu navegador web en `http://localhost:8501`.

---

## 🖥️ Módulos del Dashboard Interactivo en Streamlit

1. **Filtros Dinámicos Reactivos (Sidebar):**
   * Deslizadores interactivos por rango de antigüedad (`tenure`: 0 a 72 meses) y facturación mensual.
   * Filtros multiselección por modalidad de contrato, método de pago, tecnología de internet y segmento K-Means.
2. **Panel Ejecutivo & Tarjetas KPI:**
   * Recálculo instantáneo de la base filtrada: Total Clientes, Tasa de Deserción (% Churn), Ingreso Mensual Recurrente en Riesgo ($ MRR) y Métricas de los Modelos (AUC 0.847, $R^2$ 0.999).
   * Gráficos interactivos Plotly con distribuciones de cargos y tasas de fuga por canal de pago.
3. **Simulador de Predicción en Vivo (Inferencia en Tiempo Real):**
   * Formulario interactivo para ingresar el perfil de cualquier cliente nuevo o existente.
   * Diagnóstico inmediato: Probabilidad porcentual de Churn, semáforo de riesgo (Bajo / Moderado / Crítico), estimación de cargos totales acumulados y clúster K-Means asignado.
   * **Recomendaciones Prescriptivas:** Planes de acción comercial personalizados (migración de contrato, incentivos de pago automático, soporte preventivo).
4. **Segmentación Tridimensional (Clustering K-Means & PCA):**
   * Visualización interactiva en espacio bidimensional (PCA 2D) o tridimensional (PCA 3D con rotación orbital en tiempo real).
   * Perfiles estadísticos promedio de cada clúster y fichas explicativas de intervención.
5. **Auditoría Metodológica & Interpretabilidad XAI:**
   * Tablas comparativas de todas las fases, curvas ROC / Precision-Recall, curvas de pérdida PyTorch y gráficos Beeswarm de SHAP.

---

## 📑 Informe Técnico Final en LaTeX

El informe final consolidado de la Fase 4 fue elaborado con calidad editorial bajo la estructura formal de un artículo científico/técnico:
* **Ubicación del PDF compilado:** [`reports/fase4/main.pdf`](reports/fase4/main.pdf)
* **Código fuente LaTeX:** [`reports/fase4/main.tex`](reports/fase4/main.tex)
* **Extensión:** 15 páginas rigurosamente diagramadas con referencias bibliográficas IEEE/APA y figuras vectoriales de alta resolución.

Para recompilar el informe localmente:
```bash
pdflatex -interaction=nonstopmode -output-directory=reports/fase4 reports/fase4/main.tex
pdflatex -interaction=nonstopmode -output-directory=reports/fase4 reports/fase4/main.tex
```

---

## ⚖️ Licencia y Reconocimientos
Este proyecto se distribuye bajo la licencia **MIT License**. Desarrollado con propósitos académicos y de investigación en el marco de la Maestría en Inteligencia Artificial de la **Universidad de La Salle** (Bogotá D.C., Colombia).
