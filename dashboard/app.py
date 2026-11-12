"""
dashboard/app.py
================
Aplicación Web Interactiva de Ciencia de Datos y Machine Learning
Proyecto Integrador - Fase 4: Despliegue en Streamlit

Universidad de La Salle - Maestría en Inteligencia Artificial
Asignatura: Ciencia de Datos
Docente: Fabián Camilo Castro Riveros
Estudiante: Yerson León Montañez
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib

# -----------------------------------------------------------------------------
# CONFIGURACIÓN GENERAL DE LA PÁGINA
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Telco Churn Analytics | Univ. de La Salle",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS institucionales personalizados (Azul Salle #002B5B y Dorado #C5A059)
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #002B5B 0%, #014F86 100%);
        padding: 24px;
        border-radius: 12px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 4px 12px rgba(0, 43, 91, 0.15);
    }
    .main-header h1 {
        color: #FFFFFF;
        font-size: 2.1rem;
        font-weight: 700;
        margin-bottom: 6px;
    }
    .main-header p {
        color: #E0E7FF;
        font-size: 1.05rem;
        margin: 0;
    }
    .badge-salle {
        background-color: #C5A059;
        color: #002B5B;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
        display: inline-block;
        margin-top: 8px;
    }
    .kpi-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #002B5B;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        margin-bottom: 12px;
    }
    .kpi-title {
        font-size: 0.88rem;
        color: #6C757D;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .kpi-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #002B5B;
        margin-top: 4px;
    }
    .kpi-sub {
        font-size: 0.82rem;
        color: #28A745;
        font-weight: 500;
    }
    .kpi-sub-alert {
        font-size: 0.82rem;
        color: #DC3545;
        font-weight: 500;
    }
    .risk-high {
        background-color: #F8D7DA;
        border-left: 5px solid #DC3545;
        color: #721C24;
        padding: 15px;
        border-radius: 8px;
    }
    .risk-med {
        background-color: #FFF3CD;
        border-left: 5px solid #FFC107;
        color: #856404;
        padding: 15px;
        border-radius: 8px;
    }
    .risk-low {
        background-color: #D4EDDA;
        border-left: 5px solid #28A745;
        color: #155724;
        padding: 15px;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# CARGA DE DATOS Y MODELOS EN CACHÉ
# -----------------------------------------------------------------------------
@st.cache_data
def load_dataset():
    """Carga el dataset procesado con clusterización y predicciones."""
    data_path = "data/processed/telco_churn_dashboard_data.csv"
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
    else:
        # Fallback si no ha corrido el export
        raw_path = "data/raw/telco_customer_churn.csv"
        df = pd.read_csv(raw_path)
        df['TotalCharges_Clean'] = pd.to_numeric(df['TotalCharges'].replace(' ', np.nan), errors='coerce').fillna(0.0)
        df['Cluster_ID'] = 0
        df['Cluster_Name'] = 'Sin segmentar'
        df['Churn_Probability'] = 0.26
        df['Churn_Risk_Level'] = 'Bajo Riesgo'
        df['PCA1'] = 0.0
        df['PCA2'] = 0.0
        df['PCA3'] = 0.0
        df['Predicted_TotalCharges'] = df['TotalCharges_Clean']
    return df


@st.cache_resource
def load_models_and_artifacts():
    """Carga los modelos entrenados y artefactos serializados."""
    artifacts = {}
    models_dir = "dashboard/models"
    try:
        artifacts['clf_preprocessor'] = joblib.load(os.path.join(models_dir, "clf_preprocessor.joblib"))
        artifacts['xgb_clf'] = joblib.load(os.path.join(models_dir, "xgb_classifier.joblib"))
        artifacts['rf_clf'] = joblib.load(os.path.join(models_dir, "rf_classifier.joblib"))
        artifacts['reg_preprocessor'] = joblib.load(os.path.join(models_dir, "reg_preprocessor.joblib"))
        artifacts['rf_reg'] = joblib.load(os.path.join(models_dir, "rf_regressor.joblib"))
        artifacts['cluster'] = joblib.load(os.path.join(models_dir, "cluster_artifacts.joblib"))
    except Exception as e:
        artifacts['error'] = str(e)

    metrics_path = os.path.join(models_dir, "metrics_summary.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            artifacts['metrics'] = json.load(f)
    else:
        artifacts['metrics'] = {}

    return artifacts


# Cargar recursos
df_data = load_dataset()
artifacts = load_models_and_artifacts()

# -----------------------------------------------------------------------------
# ENCABEZADO INSTITUCIONAL
# -----------------------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>🏛️ Sistema Inteligente de Retención y Valoración de Clientes</h1>
    <p>Proyecto Integrador de Ciencia de Datos e Inteligencia Artificial | Maestría en IA — Universidad de La Salle</p>
    <div class="badge-salle">Fase 4: Puesta en Producción & Dashboard Interactivo MLOps</div>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# BARRA LATERAL: FILTROS DINÁMICOS OBLIGATORIOS (MÓDULO 1)
# -----------------------------------------------------------------------------
st.sidebar.image("https://www.lasalle.edu.co/images/logo_lasalle.png", width=220) if False else None
st.sidebar.markdown("## 🔍 Filtros de Segmentación")
st.sidebar.markdown("---")

# 1. Filtro de Antigüedad (Tenure)
min_tenure, max_tenure = int(df_data['tenure'].min()), int(df_data['tenure'].max())
tenure_range = st.sidebar.slider(
    "Antigüedad del Cliente (Meses):",
    min_value=min_tenure,
    max_value=max_tenure,
    value=(min_tenure, max_tenure),
    step=1
)

# 2. Filtro de Tarifa Mensual
min_mc, max_mc = float(df_data['MonthlyCharges'].min()), float(df_data['MonthlyCharges'].max())
monthly_range = st.sidebar.slider(
    "Tarifa Mensual (USD):",
    min_value=min_mc,
    max_value=max_mc,
    value=(min_mc, max_mc),
    step=5.0
)

# 3. Filtro por Tipo de Contrato
contract_options = list(df_data['Contract'].unique())
selected_contracts = st.sidebar.multiselect(
    "Tipo de Contrato:",
    options=contract_options,
    default=contract_options
)

# 4. Filtro por Servicio de Internet
internet_options = list(df_data['InternetService'].unique())
selected_internet = st.sidebar.multiselect(
    "Tecnología de Internet:",
    options=internet_options,
    default=internet_options
)

# 5. Filtro por Método de Pago
payment_options = list(df_data['PaymentMethod'].unique())
selected_payments = st.sidebar.multiselect(
    "Método de Pago:",
    options=payment_options,
    default=payment_options
)

# 6. Filtro por Clúster de Segmentación
cluster_options = list(df_data['Cluster_Name'].unique()) if 'Cluster_Name' in df_data.columns else []
selected_clusters = st.sidebar.multiselect(
    "Segmento Analítico (K-Means):",
    options=cluster_options,
    default=cluster_options
)

# 7. Filtro por Nivel de Riesgo Predicho
if 'Churn_Risk_Level' in df_data.columns:
    risk_options = list(df_data['Churn_Risk_Level'].dropna().unique())
    selected_risks = st.sidebar.multiselect(
        "Nivel de Riesgo de Fuga:",
        options=risk_options,
        default=risk_options
    )
else:
    selected_risks = []

# APLICACIÓN DE FILTROS AL DATAFRAME
df_filtered = df_data[
    (df_data['tenure'] >= tenure_range[0]) & (df_data['tenure'] <= tenure_range[1]) &
    (df_data['MonthlyCharges'] >= monthly_range[0]) & (df_data['MonthlyCharges'] <= monthly_range[1]) &
    (df_data['Contract'].isin(selected_contracts)) &
    (df_data['InternetService'].isin(selected_internet)) &
    (df_data['PaymentMethod'].isin(selected_payments))
]

if selected_clusters and 'Cluster_Name' in df_filtered.columns:
    df_filtered = df_filtered[df_filtered['Cluster_Name'].isin(selected_clusters)]

if selected_risks and 'Churn_Risk_Level' in df_filtered.columns:
    df_filtered = df_filtered[df_filtered['Churn_Risk_Level'].isin(selected_risks)]

st.sidebar.markdown("---")
st.sidebar.info(f"**Registros seleccionados:** {len(df_filtered):,} de {len(df_data):,} ({len(df_filtered)/len(df_data)*100:.1f}%)")


# -----------------------------------------------------------------------------
# PESTAÑAS PRINCIPALES DEL DASHBOARD
# -----------------------------------------------------------------------------
tabs = st.tabs([
    "📈 Panel Ejecutivo & KPIs",
    "🤖 Simulador de Predicción en Vivo",
    "🎯 Segmentación Profiláctica (Clustering)",
    "🧪 Arquitectura de Modelos & XAI"
])


# =============================================================================
# PESTAÑA 1: TARJETAS KPI EJECUTIVAS Y NARRATIVA DE NEGOCIO (MÓDULO 2)
# =============================================================================
with tabs[0]:
    st.subheader("Indicadores Clave de Rendimiento (KPIs Ejecutivos)")
    
    if len(df_filtered) == 0:
        st.warning("No hay registros que coincidan con los filtros seleccionados.")
    else:
        # Cálculos de negocio
        total_subscribers = len(df_filtered)
        churn_count = (df_filtered['Churn'] == 'Yes').sum()
        churn_rate = (churn_count / total_subscribers) * 100
        
        mrr_total = df_filtered['MonthlyCharges'].sum()
        mrr_at_risk = df_filtered[df_filtered['Churn'] == 'Yes']['MonthlyCharges'].sum()
        mrr_risk_pct = (mrr_at_risk / mrr_total) * 100 if mrr_total > 0 else 0.0

        avg_tenure = df_filtered['tenure'].mean()
        avg_monthly = df_filtered['MonthlyCharges'].mean()

        # Fila 1: 4 KPIs Principales
        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

        with kpi_col1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Clientes en Cohorte</div>
                <div class="kpi-value">{total_subscribers:,}</div>
                <div class="kpi-sub">Permanencia media: {avg_tenure:.1f} meses</div>
            </div>
            """, unsafe_allow_html=True)

        with kpi_col2:
            st.markdown(f"""
            <div class="kpi-card" style="border-left-color: #DC3545;">
                <div class="kpi-title">Tasa de Deserción (Churn)</div>
                <div class="kpi-value" style="color: #DC3545;">{churn_rate:.2f}%</div>
                <div class="kpi-sub-alert">{churn_count:,} clientes desertores</div>
            </div>
            """, unsafe_allow_html=True)

        with kpi_col3:
            st.markdown(f"""
            <div class="kpi-card" style="border-left-color: #C5A059;">
                <div class="kpi-title">MRR Total en Riesgo</div>
                <div class="kpi-value" style="color: #002B5B;">${mrr_at_risk:,.0f} <span style="font-size:1rem;color:#666;">USD</span></div>
                <div class="kpi-sub-alert">{mrr_risk_pct:.1f}% de la facturación mensual</div>
            </div>
            """, unsafe_allow_html=True)

        with kpi_col4:
            st.markdown(f"""
            <div class="kpi-card" style="border-left-color: #28A745;">
                <div class="kpi-title">Capacidad Predictiva (AUC)</div>
                <div class="kpi-value" style="color: #28A745;">0.847</div>
                <div class="kpi-sub">XGBoost ($R^2$ Regresión: 0.999)</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Fila 2: Visualizaciones Gráficas Interactivas con Plotly
        col_g1, col_g2 = st.columns(2)

        with col_g1:
            # Gráfico de barras por tipo de contrato y tasa de churn
            churn_by_contract = df_filtered.groupby('Contract')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100).reset_index()
            fig_contract = px.bar(
                churn_by_contract,
                x='Contract',
                y='Churn',
                title="<b>Tasa de Deserción por Modalidad Contractual (%)</b>",
                color='Contract',
                color_discrete_map={
                    'Month-to-month': '#DC3545',
                    'One year': '#C5A059',
                    'Two year': '#002B5B'
                },
                labels={'Churn': 'Tasa de Churn (%)', 'Contract': 'Contrato'}
            )
            fig_contract.update_layout(showlegend=False, template='plotly_white', height=360)
            st.plotly_chart(fig_contract, use_container_width=True)

        with col_g2:
            # Distribución de Cargos Mensuales por Churn
            fig_hist = px.histogram(
                df_filtered,
                x='MonthlyCharges',
                color='Churn',
                barmode='overlay',
                title="<b>Distribución de Tarifas Mensuales por Estado de Churn</b>",
                color_discrete_map={'No': '#002B5B', 'Yes': '#DC3545'},
                labels={'MonthlyCharges': 'Tarifa Mensual (USD)', 'count': 'Frecuencia'}
            )
            fig_hist.update_layout(template='plotly_white', height=360)
            st.plotly_chart(fig_hist, use_container_width=True)

        # Fila 3: Riesgo por Método de Pago y Servicios de Conectividad
        col_g3, col_g4 = st.columns(2)

        with col_g3:
            churn_pay = df_filtered.groupby('PaymentMethod')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100).reset_index()
            fig_pay = px.bar(
                churn_pay.sort_values(by='Churn', ascending=True),
                x='Churn',
                y='PaymentMethod',
                orientation='h',
                title="<b>Vulnerabilidad de Fuga según Método de Pago</b>",
                color='Churn',
                color_continuous_scale='Reds',
                labels={'Churn': '% Deserción', 'PaymentMethod': 'Método'}
            )
            fig_pay.update_layout(template='plotly_white', height=340)
            st.plotly_chart(fig_pay, use_container_width=True)

        with col_g4:
            # Desglose por Segmento K-Means
            if 'Cluster_Name' in df_filtered.columns:
                cluster_dist = df_filtered['Cluster_Name'].value_counts().reset_index()
                cluster_dist.columns = ['Segmento', 'Clientes']
                fig_pie = px.pie(
                    cluster_dist,
                    names='Segmento',
                    values='Clientes',
                    title="<b>Distribución de Clientes por Clúster K-Means</b>",
                    color_discrete_sequence=['#002B5B', '#DC3545', '#C5A059'],
                    hole=0.4
                )
                fig_pie.update_layout(template='plotly_white', height=340)
                st.plotly_chart(fig_pie, use_container_width=True)


# =============================================================================
# PESTAÑA 2: MÓDULO DE PREDICCIÓN EN VIVO (MÓDULO 3)
# =============================================================================
with tabs[1]:
    st.subheader("🎯 Simulador de Predicción en Tiempo Real (Machine Learning en Vivo)")
    st.markdown("""
    Este simulador utiliza el ensamble optimizado **XGBoost Classifier** y el **Random Forest Regressor** 
    para evaluar simultáneamente la **probabilidad de deserción** y el **valor acumulado de ciclo de vida (TotalCharges)** de un cliente.
    """)

    with st.form("prediction_form"):
        st.markdown("#### 1. Perfil Demográfico y Vinculación")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            input_gender = st.selectbox("Género:", ["Female", "Male"])
        with c2:
            input_senior = st.selectbox("Adulto Mayor (SeniorCitizen):", [0, 1], format_func=lambda x: "Sí" if x == 1 else "No")
        with c3:
            input_partner = st.selectbox("Tiene Pareja (Partner):", ["No", "Yes"])
        with c4:
            input_dependents = st.selectbox("Tiene Dependientes:", ["No", "Yes"])

        st.markdown("#### 2. Servicios Contratados")
        c5, c6, c7, c8 = st.columns(4)
        with c5:
            input_phone = st.selectbox("Línea Telefónica:", ["Yes", "No"])
            input_multiple = st.selectbox("Múltiples Líneas:", ["No", "Yes", "No phone service"])
        with c6:
            input_internet = st.selectbox("Servicio de Internet:", ["Fiber optic", "DSL", "No"])
            input_security = st.selectbox("Seguridad Online:", ["No", "Yes", "No internet service"])
        with c7:
            input_backup = st.selectbox("Copia de Respaldo Online:", ["No", "Yes", "No internet service"])
            input_device = st.selectbox("Protección de Dispositivo:", ["No", "Yes", "No internet service"])
        with c8:
            input_tech = st.selectbox("Soporte Técnico Especializado:", ["No", "Yes", "No internet service"])
            input_tv = st.selectbox("Transmisión de TV:", ["No", "Yes", "No internet service"])
            input_movies = st.selectbox("Transmisión de Películas:", ["No", "Yes", "No internet service"])

        st.markdown("#### 3. Condiciones Comerciales y Facturación")
        c9, c10, c11 = st.columns(3)
        with c9:
            input_tenure = st.slider("Antigüedad (Meses):", min_value=1, max_value=72, value=12)
            input_contract = st.selectbox("Tipo de Contrato:", ["Month-to-month", "One year", "Two year"])
        with c10:
            input_paperless = st.selectbox("Facturación Electrónica:", ["Yes", "No"])
            input_payment = st.selectbox(
                "Método de Pago:",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ]
            )
        with c11:
            input_monthly = st.slider("Tarifa Mensual (USD):", min_value=18.0, max_value=120.0, value=75.0, step=0.5)

        submit_btn = st.form_submit_button("⚡ Generar Predicción y Diagnóstico Prescriptivo", use_container_width=True)

    if submit_btn:
        # Construir registro de prueba idéntico al pipeline de entrenamiento
        input_dict = {
            'customerID': ['SIMULATED_USER'],
            'gender': [input_gender],
            'SeniorCitizen': [input_senior],
            'Partner': [input_partner],
            'Dependents': [input_dependents],
            'tenure': [input_tenure],
            'PhoneService': [input_phone],
            'MultipleLines': [input_multiple],
            'InternetService': [input_internet],
            'OnlineSecurity': [input_security],
            'OnlineBackup': [input_backup],
            'DeviceProtection': [input_device],
            'TechSupport': [input_tech],
            'StreamingTV': [input_tv],
            'StreamingMovies': [input_movies],
            'Contract': [input_contract],
            'PaperlessBilling': [input_paperless],
            'PaymentMethod': [input_payment],
            'MonthlyCharges': [input_monthly],
            'TotalCharges': [str(input_monthly * max(1, input_tenure))]
        }
        user_df = pd.DataFrame(input_dict)
        user_df['TotalCharges_Clean'] = input_monthly * max(1, input_tenure)
        user_df['Partner_Binary'] = (user_df['Partner'] == 'Yes').astype(int)
        user_df['Dependents_Binary'] = (user_df['Dependents'] == 'Yes').astype(int)
        user_df['PhoneService_Binary'] = (user_df['PhoneService'] == 'Yes').astype(int)
        user_df['PaperlessBilling_Binary'] = (user_df['PaperlessBilling'] == 'Yes').astype(int)

        # Variables derivadas de dominio
        def calc_services(row):
            cnt = 0
            if row['PhoneService'] == 'Yes': cnt += 1
            if row['MultipleLines'] == 'Yes': cnt += 1
            if row['InternetService'] in ['DSL', 'Fiber optic']: cnt += 1
            for k in ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']:
                if row[k] == 'Yes': cnt += 1
            return cnt

        user_df['TotalServices'] = user_df.apply(calc_services, axis=1)
        user_df['AvgHistoricalFee'] = user_df['TotalCharges_Clean'] / max(1, input_tenure)
        user_df['FeeDiscrepancyRatio'] = np.clip(user_df['MonthlyCharges'] / (user_df['AvgHistoricalFee'] + 1e-5), 0.2, 5.0)
        auto_pay = ['Bank transfer (automatic)', 'Credit card (automatic)']
        user_df['AutomaticPayment'] = user_df['PaymentMethod'].isin(auto_pay).astype(int)
        user_df['LongTermContract'] = user_df['Contract'].isin(['One year', 'Two year']).astype(int)

        # Realizar predicciones
        if 'clf_preprocessor' in artifacts and 'xgb_clf' in artifacts:
            clf_prep = artifacts['clf_preprocessor']
            model_xgb = artifacts['xgb_clf']
            reg_prep = artifacts['reg_preprocessor']
            model_reg = artifacts['rf_reg']

            X_in_clf = clf_prep.transform(user_df)
            prob_churn = model_xgb.predict_proba(X_in_clf)[0, 1]

            X_in_reg = reg_prep.transform(user_df)
            pred_charges = model_reg.predict(X_in_reg)[0]

            # Clasificación de riesgo
            st.markdown("### 📊 Resultado de la Evaluación Algorítmica")
            res_c1, res_c2, res_c3 = st.columns(3)

            with res_c1:
                st.metric("Probabilidad de Deserción", f"{prob_churn*100:.1f}%")
                # Barra de progreso
                st.progress(float(prob_churn))

            with res_c2:
                if prob_churn >= 0.60:
                    st.markdown("""
                    <div class="risk-high">
                        <strong>⚠️ RIESGO CRÍTICO DE FUGA</strong><br>
                        Alta propensión a cancelar el servicio en el próximo ciclo de facturación.
                    </div>
                    """, unsafe_allow_html=True)
                elif prob_churn >= 0.35:
                    st.markdown("""
                    <div class="risk-med">
                        <strong>⚡ RIESGO MODERADO</strong><br>
                        Cliente vulnerable a ofertas de la competencia o variaciones de precio.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="risk-low">
                        <strong>✅ CLIENTE ESTABLE / BAJO RIESGO</strong><br>
                        Comportamiento leal y alta probabilidad de permanencia.
                    </div>
                    """, unsafe_allow_html=True)

            with res_c3:
                st.metric("Facturación Total Acumulada Proyectada", f"${pred_charges:,.2f} USD")
                st.caption(f"Valor estimado por el ensamble Random Forest ($R^2=0.999$).")

            # Mapeo a Clúster K-Means
            if 'cluster' in artifacts:
                c_art = artifacts['cluster']
                sc = c_art['scaler']
                km = c_art['kmeans']
                u_clust = user_df[c_art['clustering_cols']].values
                u_scaled = sc.transform(u_clust)
                c_pred = km.predict(u_scaled)[0]
                c_name = c_art['cluster_name_map'].get(c_pred, "Desconocido")
                st.info(f"📍 **Segmento Asignado:** Clúster {c_pred} — **{c_name}** ({c_art['cluster_desc_map'].get(c_pred, '')})")

            # Prescripciones de Negocio
            st.markdown("#### 💡 Recomendaciones Prescriptivas de Retención:")
            recs = []
            if input_contract == 'Month-to-month':
                recs.append("🎯 **Migración de Contrato:** Ofrecer descuento del 15% sobre la tarifa mensual condicionada a la firma de contrato a 12 meses.")
            if input_payment == 'Electronic check':
                recs.append("💳 **Domiciliación Bancaria:** El cheque electrónico concentra la mayor tasa de deserción; incentivar la domiciliación en tarjeta con un bono de bienvenida.")
            if input_tech == 'No' and input_internet != 'No':
                recs.append("🛠️ **Soporte Técnico:** Ofrecer 3 meses de soporte técnico y seguridad digital sin costo para incrementar las barreras de cambio.")
            if input_tenure < 12:
                recs.append("📞 **Protocolo Onboarding:** Programar llamada de seguimiento de calidad del servicio a los 30 y 90 días para mitigar la deserción temprana.")
            if not recs:
                recs.append("⭐ **Fidelización Activa:** El suscriptor muestra alta lealtad. Incorporar al programa VIP de beneficios sin saturar con promociones agresivas.")

            for r in recs:
                st.markdown(r)

        else:
            st.error("Los artefactos de modelos aún no han sido serializados. Ejecute primero el script de exportación.")


# =============================================================================
# PESTAÑA 3: MÓDULO DE SEGMENTACIÓN (CLUSTERING K-MEANS) (MÓDULO 4)
# =============================================================================
with tabs[2]:
    st.subheader("🎯 Módulo de Segmentación Profiláctica de Clientes (K-Means & PCA)")
    st.markdown("""
    En concordancia con los hallazgos de la Fase 3, se implementó una segmentación no supervisada sobre 5 variables
    clave estandarizadas ($\text{tenure}, \text{MonthlyCharges}, \text{TotalCharges}, \text{TotalServices}, \text{FeeDiscrepancyRatio}$).
    El número óptimo de clústeres es **$K=3$**, validado mediante el Método del Codo y el Coeficiente de Silueta ($s = 0.3835$).
    """)

    # Visualización PCA 2D / 3D con Plotly
    dim_choice = st.radio("Modalidad de Proyección:", ["Espacio PCA 2D", "Espacio PCA 3D"], horizontal=True)

    color_cluster_map = {
        'Premium Fidelizado': '#002B5B',
        'Nuevo en Alto Riesgo': '#DC3545',
        'Económico Básico': '#C5A059'
    }

    if dim_choice == "Espacio PCA 2D":
        fig_pca = px.scatter(
            df_filtered,
            x='PCA1',
            y='PCA2',
            color='Cluster_Name' if 'Cluster_Name' in df_filtered.columns else 'Cluster_ID',
            hover_data=['customerID', 'tenure', 'MonthlyCharges', 'Contract', 'Churn', 'Churn_Probability'],
            color_discrete_map=color_cluster_map,
            title="<b>Proyección de Clientes en el Espacio Reducido PCA (2D)</b>",
            labels={'PCA1': 'Componente Principal 1 (Consumo & Antigüedad)', 'PCA2': 'Componente Principal 2 (Servicios)'}
        )
        fig_pca.update_traces(marker=dict(size=6, opacity=0.7))
        fig_pca.update_layout(template='plotly_white', height=520)
        st.plotly_chart(fig_pca, use_container_width=True)
    else:
        fig_pca3 = px.scatter_3d(
            df_filtered.sample(min(1500, len(df_filtered)), random_state=42),
            x='PCA1',
            y='PCA2',
            z='PCA3',
            color='Cluster_Name' if 'Cluster_Name' in df_filtered.columns else 'Cluster_ID',
            hover_data=['customerID', 'tenure', 'MonthlyCharges', 'Contract', 'Churn'],
            color_discrete_map=color_cluster_map,
            title="<b>Proyección Tridimensional PCA (3D) de Clústeres (Muestra Representativa)</b>"
        )
        fig_pca3.update_traces(marker=dict(size=4, opacity=0.8))
        fig_pca3.update_layout(template='plotly_white', height=560)
        st.plotly_chart(fig_pca3, use_container_width=True)

    # Perfiles Estadísticos de los Clústeres
    st.markdown("### 📋 Perfiles Medios de Negocio por Segmento")
    if 'Cluster_Name' in df_filtered.columns:
        prof_summary = df_data.groupby('Cluster_Name').agg(
            Clientes=('customerID', 'count'),
            Antigüedad_Meses=('tenure', 'mean'),
            Tarifa_Mensual_USD=('MonthlyCharges', 'mean'),
            Cobro_Total_USD=('TotalCharges_Clean', 'mean'),
            Tasa_Churn_Pct=('Churn', lambda x: (x == 'Yes').mean() * 100),
            Pago_Automatico_Pct=('AutomaticPayment', lambda x: x.mean() * 100),
            Contrato_Largo_Pct=('LongTermContract', lambda x: x.mean() * 100)
        ).reset_index()

        prof_summary['Antigüedad_Meses'] = prof_summary['Antigüedad_Meses'].round(1)
        prof_summary['Tarifa_Mensual_USD'] = prof_summary['Tarifa_Mensual_USD'].map(lambda x: f"${x:.2f}")
        prof_summary['Cobro_Total_USD'] = prof_summary['Cobro_Total_USD'].map(lambda x: f"${x:.2f}")
        prof_summary['Tasa_Churn_Pct'] = prof_summary['Tasa_Churn_Pct'].map(lambda x: f"{x:.1f}%")
        prof_summary['Pago_Automatico_Pct'] = prof_summary['Pago_Automatico_Pct'].map(lambda x: f"{x:.1f}%")
        prof_summary['Contrato_Largo_Pct'] = prof_summary['Contrato_Largo_Pct'].map(lambda x: f"{x:.1f}%")

        st.dataframe(prof_summary, use_container_width=True, hide_index=True)

    # Diagnóstico Profiláctico en 3 Columnas
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        st.markdown("""
        <div style="background-color:#EBF3FA; padding:15px; border-radius:8px; border-top: 4px solid #002B5B;">
            <h4 style="color:#002B5B; margin-top:0;">Clúster 0: Premium Fidelizado</h4>
            <p><strong>Perfil:</strong> Antigüedad ~58 meses, factura mensual ~$90 USD, más de 6 servicios contratados y alta adopción de débito automático.</p>
            <p><strong>Tasa de Churn:</strong> 15.0%</p>
            <p><strong>Estrategia:</strong> Retención por lealtad, atención posventa prioritaria y programas VIP.</p>
        </div>
        """, unsafe_allow_html=True)

    with sc2:
        st.markdown("""
        <div style="background-color:#FDE8E8; padding:15px; border-radius:8px; border-top: 4px solid #DC3545;">
            <h4 style="color:#DC3545; margin-top:0;">Clúster 1: Nuevo en Alto Riesgo</h4>
            <p><strong>Perfil:</strong> Antigüedad ~15 meses, contratos mes a mes (85%), pagos manuales con cheque electrónico y facturación media-alta.</p>
            <p><strong>Tasa de Churn:</strong> 45.4%</p>
            <p><strong>Estrategia:</strong> Foco crítico de rescate: descuentos por vinculación a 12 meses y cambio de método de pago.</p>
        </div>
        """, unsafe_allow_html=True)

    with sc3:
        st.markdown("""
        <div style="background-color:#FEF6E4; padding:15px; border-radius:8px; border-top: 4px solid #C5A059;">
            <h4 style="color:#856404; margin-top:0;">Clúster 2: Económico Básico</h4>
            <p><strong>Perfil:</strong> Antigüedad moderada (~28 meses), consumo austero (~$26 USD), usuarios principalmente de telefonía básica.</p>
            <p><strong>Tasa de Churn:</strong> 14.5%</p>
            <p><strong>Estrategia:</strong> Campañas de venta cruzada (Cross-Selling) de streaming y fibra óptica económica para elevar el ARPU.</p>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# PESTAÑA 4: ARQUITECTURA DE MODELOS, EVALUACIÓN COMPARATIVA Y XAI
# =============================================================================
with tabs[3]:
    st.subheader("🧪 Auditoría Metodológica de Modelado & Explicabilidad SHAP")
    
    st.markdown("""
    Durante el ciclo de vida del proyecto se entrenaron, validaron y contrastaron modelos supervisados y no supervisados,
    garantizando ausencia de fuga de datos (*Data Leakage*), estratificación rigurosa y explicabilidad basada en teoría cooperativa de juegos.
    """)

    # Tablas comparativas
    m_col1, m_col2 = st.columns(2)

    with m_col1:
        st.markdown("#### 1. Modelos de Clasificación (Predicción de Churn)")
        clf_metrics_data = {
            'Modelo': ['Baseline Regresión Logística', 'Random Forest (Balanced)', 'XGBoost Classifier (Final)', 'PyTorch Perceptrón Multicapa (MLP)'],
            'Partición': ['Test', 'Test', 'Test', 'Test'],
            'Accuracy': [0.7417, 0.7758, 0.7720, 0.7483],
            'Precision': [0.5078, 0.5518, 0.5457, 0.5165],
            'Recall': [0.8107, 0.7679, 0.7821, 0.8036],
            'F1-Score': [0.6245, 0.6418, 0.6432, 0.6285],
            'AUC-ROC': [0.8492, 0.8466, 0.8467, 0.8332]
        }
        st.dataframe(pd.DataFrame(clf_metrics_data), use_container_width=True, hide_index=True)

    with m_col2:
        st.markdown("#### 2. Modelos de Regresión (TotalCharges en USD)")
        reg_metrics_data = {
            'Modelo': ['Baseline Regresión Lineal', 'Ridge Regression (L2)', 'Lasso Regression (L1)', 'Random Forest Regressor (Final)', 'XGBoost Regressor'],
            'Partición': ['Test', 'Test', 'Test', 'Test', 'Test'],
            'R2 Score': [0.9037, 0.9058, 0.9058, 0.9986, 0.9980],
            'RMSE (USD)': [688.85, 681.38, 681.30, 81.91, 99.24],
            'MAE (USD)': [558.12, 539.52, 539.45, 55.88, 71.86]
        }
        st.dataframe(pd.DataFrame(reg_metrics_data), use_container_width=True, hide_index=True)

    # Galería de Diagnósticos Visuales
    st.markdown("#### 3. Evidencias Diagnósticas del Informe Técnico")
    fig_c1, fig_c2 = st.columns(2)
    with fig_c1:
        if os.path.exists("reports/fase3/figures/fig2_classification_roc_pr.png"):
            st.image("reports/fase3/figures/fig2_classification_roc_pr.png", caption="Curvas ROC y Precision-Recall en Conjunto de Prueba", use_container_width=True)
        if os.path.exists("reports/fase3/figures/fig3_pytorch_loss_curve.png"):
            st.image("reports/fase3/figures/fig3_pytorch_loss_curve.png", caption="Curva de Convergencia de Pérdida en Red Neuronal PyTorch", use_container_width=True)

    with fig_c2:
        if os.path.exists("reports/fase3/figures/fig6_shap_summary.png"):
            st.image("reports/fase3/figures/fig6_shap_summary.png", caption="Importancia Global y Direccionalidad de Factores con Valores SHAP", use_container_width=True)
        if os.path.exists("reports/fase3/figures/fig4_clustering_elbow_silhouette.png"):
            st.image("reports/fase3/figures/fig4_clustering_elbow_silhouette.png", caption="Método del Codo y Coeficiente de Silueta para K-Means", use_container_width=True)

# -----------------------------------------------------------------------------
# PIE DE PÁGINA
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6C757D; font-size: 0.85rem; padding: 10px;">
    <strong>Universidad de La Salle</strong> | Facultad de Ingeniería — Maestría en Inteligencia Artificial<br>
    Asignatura: Ciencia de Datos | Docente: Fabián Camilo Castro Riveros | Autor: Yerson León Montañez (2026)
</div>
""", unsafe_allow_html=True)
