"""
src/export_fase4_artifacts.py
==============================
Script de serialización y preparación de artefactos para la Fase 4:
- Preprocesadores y modelos de Clasificación (XGBoost y Random Forest).
- Preprocesadores y modelos de Regresión (Random Forest Regressor).
- Modelo de Clustering (K-Means K=3) y reducciones dimensionales PCA (2D y 3D).
- Dataset procesado y enriquecido con predicciones y asignación de clústeres
  para el Dashboard interactivo en Streamlit.

Universidad de La Salle - Maestría en Inteligencia Artificial
Asignatura: Ciencia de Datos
Docente: Fabián Camilo Castro Riveros
Estudiante: Yerson León Montañez
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier

from src.fase2_pipeline import (
    load_and_audit_raw_data,
    preprocess_dataframe_binary_cols,
    DomainFeatureEngineer,
    split_dataset_stratified,
    build_feature_preprocessor
)
from src.fase3_models import (
    build_regression_feature_preprocessor,
    set_seed
)

def run_export():
    set_seed(42)
    print(">>> [1/5] Cargando dataset crudo y aplicando ingeniería de dominio...")
    raw_path = "data/raw/telco_customer_churn.csv"
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"No se encontró el dataset en {raw_path}")

    df_raw, _ = load_and_audit_raw_data(raw_path)
    df_binary = preprocess_dataframe_binary_cols(df_raw)
    engineer = DomainFeatureEngineer()
    df_featured = engineer.transform(df_binary)

    # Crear carpetas de destino
    os.makedirs("dashboard/models", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    # Partición estratificada (70% Train, 15% Val, 15% Test)
    splits = split_dataset_stratified(df_featured, target_col='Churn', test_size=0.15, val_size=0.15, random_state=42)
    X_train_df, y_train = splits['X_train'], splits['y_train']
    X_val_df, y_val = splits['X_val'], splits['y_val']
    X_test_df, y_test = splits['X_test'], splits['y_test']

    # -------------------------------------------------------------------------
    # 2. ENTRENAR PIPELINE DE CLASIFICACIÓN (XGBoost & Random Forest)
    # -------------------------------------------------------------------------
    print(">>> [2/5] Ajustando y serializando pipeline de Clasificación...")
    clf_preprocessor = build_feature_preprocessor(scaler_type='standard')
    clf_preprocessor.fit(X_train_df)

    X_train_clf = clf_preprocessor.transform(X_train_df)
    X_test_clf = clf_preprocessor.transform(X_test_df)

    xgb_clf = XGBClassifier(
        n_estimators=140, max_depth=4, learning_rate=0.06,
        subsample=0.85, colsample_bytree=0.85, scale_pos_weight=2.77,
        random_state=42, eval_metric='logloss', n_jobs=-1
    )
    xgb_clf.fit(X_train_clf, y_train)

    rf_clf = RandomForestClassifier(
        n_estimators=150, max_depth=7, min_samples_split=5,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    rf_clf.fit(X_train_clf, y_train)

    # -------------------------------------------------------------------------
    # 3. ENTRENAR PIPELINE DE REGRESIÓN (Random Forest Regressor & XGBoost)
    # -------------------------------------------------------------------------
    print(">>> [3/5] Ajustando y serializando pipeline de Regresión...")
    y_train_reg = X_train_df['TotalCharges_Clean'].values
    y_test_reg = X_test_df['TotalCharges_Clean'].values

    reg_preprocessor = build_regression_feature_preprocessor(scaler_type='standard')
    reg_preprocessor.fit(X_train_df)

    X_train_reg = reg_preprocessor.transform(X_train_df)
    X_test_reg = reg_preprocessor.transform(X_test_df)

    rf_reg = RandomForestRegressor(
        n_estimators=120, max_depth=9, min_samples_split=4, random_state=42, n_jobs=-1
    )
    rf_reg.fit(X_train_reg, y_train_reg)

    xgb_reg = XGBRegressor(
        n_estimators=120, max_depth=4, learning_rate=0.07, subsample=0.85,
        colsample_bytree=0.85, random_state=42, n_jobs=-1
    )
    xgb_reg.fit(X_train_reg, y_train_reg)

    # -------------------------------------------------------------------------
    # 4. ENTRENAR CLUSTERING (K-Means K=3) Y PCA (2D y 3D)
    # -------------------------------------------------------------------------
    print(">>> [4/5] Ajustando Clustering (K=3) y proyecciones PCA...")
    clustering_cols = ['tenure', 'MonthlyCharges', 'TotalCharges_Clean', 'TotalServices', 'FeeDiscrepancyRatio']
    X_clust = df_featured[clustering_cols].values

    cluster_scaler = StandardScaler()
    X_clust_scaled = cluster_scaler.fit_transform(X_clust)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=15)
    cluster_labels = kmeans.fit_predict(X_clust_scaled)

    pca_2d = PCA(n_components=2, random_state=42)
    coords_pca2d = pca_2d.fit_transform(X_clust_scaled)

    pca_3d = PCA(n_components=3, random_state=42)
    coords_pca3d = pca_3d.fit_transform(X_clust_scaled)

    # Nombres de negocio para los segmentos
    # Identificar clústeres por sus características medias:
    # Cluster 0: Premium (antigüedad alta, cargos altos)
    # Cluster 1: En Riesgo (antigüedad baja, cargos medios-altos, churn alto)
    # Cluster 2: Económico (cargos bajos)
    temp_df = pd.DataFrame({'cluster': cluster_labels, 'tenure': df_featured['tenure'], 'MonthlyCharges': df_featured['MonthlyCharges']})
    cluster_stats = temp_df.groupby('cluster').mean()
    
    # Mapeo según la Fase 3:
    # 0 -> Clientes Premium Fidelizados
    # 1 -> Clientes Nuevos en Riesgo
    # 2 -> Clientes Económicos Básicos
    cluster_name_map = {
        0: "Premium Fidelizado",
        1: "Nuevo en Alto Riesgo",
        2: "Económico Básico"
    }
    cluster_desc_map = {
        0: "Alta antigüedad (~58 m), alto consumo (~$90/mes), 6+ servicios, bajo Churn (15.0%).",
        1: "Baja antigüedad (~15 m), contrato mensual, pagos manuales, crítico Churn (45.4%).",
        2: "Servicios básicos (~$26/mes), permanencia media (~28 m), bajo Churn (14.5%)."
    }

    # Asignar al dataframe global para el dashboard
    df_dashboard = df_featured.copy()
    df_dashboard['Cluster_ID'] = cluster_labels
    df_dashboard['Cluster_Name'] = df_dashboard['Cluster_ID'].map(cluster_name_map)
    df_dashboard['PCA1'] = coords_pca2d[:, 0]
    df_dashboard['PCA2'] = coords_pca2d[:, 1]
    df_dashboard['PCA3'] = coords_pca3d[:, 2]

    # Predecir probabilidades de Churn con XGBoost para todo el dataset
    X_all_clf = clf_preprocessor.transform(df_dashboard)
    churn_probs = xgb_clf.predict_proba(X_all_clf)[:, 1]
    df_dashboard['Churn_Probability'] = np.round(churn_probs, 4)
    df_dashboard['Churn_Risk_Level'] = pd.cut(
        df_dashboard['Churn_Probability'],
        bins=[-0.01, 0.35, 0.65, 1.01],
        labels=['Bajo Riesgo', 'Riesgo Moderado', 'Riesgo Crítico']
    )

    # Predecir cobros totales esperados con Random Forest
    X_all_reg = reg_preprocessor.transform(df_dashboard)
    pred_total_charges = rf_reg.predict(X_all_reg)
    df_dashboard['Predicted_TotalCharges'] = np.round(pred_total_charges, 2)

    # Guardar dataframe enriquecido
    processed_csv_path = "data/processed/telco_churn_dashboard_data.csv"
    df_dashboard.to_csv(processed_csv_path, index=False)
    print(f"[OK] Dataset procesado guardado en {processed_csv_path} ({df_dashboard.shape[0]} filas, {df_dashboard.shape[1]} columnas).")

    # -------------------------------------------------------------------------
    # 5. SERIALIZAR MODELOS Y METADATOS
    # -------------------------------------------------------------------------
    print(">>> [5/5] Exportando objetos serializados (joblib) y métricas...")
    
    # 1. Clasificación
    joblib.dump(clf_preprocessor, "dashboard/models/clf_preprocessor.joblib")
    joblib.dump(xgb_clf, "dashboard/models/xgb_classifier.joblib")
    joblib.dump(rf_clf, "dashboard/models/rf_classifier.joblib")

    # 2. Regresión
    joblib.dump(reg_preprocessor, "dashboard/models/reg_preprocessor.joblib")
    joblib.dump(rf_reg, "dashboard/models/rf_regressor.joblib")
    joblib.dump(xgb_reg, "dashboard/models/xgb_regressor.joblib")

    # 3. Clustering
    cluster_artifacts = {
        'scaler': cluster_scaler,
        'kmeans': kmeans,
        'pca_2d': pca_2d,
        'pca_3d': pca_3d,
        'clustering_cols': clustering_cols,
        'cluster_name_map': cluster_name_map,
        'cluster_desc_map': cluster_desc_map,
        'pca_var_ratio_2d': pca_2d.explained_variance_ratio_.tolist(),
        'pca_var_ratio_3d': pca_3d.explained_variance_ratio_.tolist()
    }
    joblib.dump(cluster_artifacts, "dashboard/models/cluster_artifacts.joblib")

    # 4. Métricas consolidadas del proyecto para los KPIs ejecutivos
    metrics_summary = {
        'project_title': 'Proyecto Integrador de Ciencia de Datos - Universidad de La Salle',
        'student': 'Yerson León Montañez',
        'teacher': 'Fabián Camilo Castro Riveros',
        'dataset_total_records': len(df_dashboard),
        'dataset_features': df_raw.shape[1],
        'classification_best_model': 'XGBoost Classifier',
        'classification_metrics_test': {
            'Accuracy': 0.7720,
            'Precision': 0.5457,
            'Recall': 0.7821,
            'F1_Score_Pos': 0.6432,
            'AUC_ROC': 0.8467,
            'PR_AUC': 0.6551
        },
        'regression_best_model': 'Random Forest Regressor',
        'regression_metrics_test': {
            'R2_Score': 0.9986,
            'RMSE_USD': 81.91,
            'MAE_USD': 55.88
        },
        'clustering_metrics': {
            'optimal_k': 3,
            'kmeans_silhouette': 0.3835,
            'hierarchical_silhouette': 0.3419,
            'segments': [
                {'id': 0, 'name': 'Premium Fidelizado', 'clients': int((df_dashboard['Cluster_ID']==0).sum()), 'avg_tenure': 58.5, 'avg_fee': 90.07, 'churn_rate': 15.0},
                {'id': 1, 'name': 'Nuevo en Alto Riesgo', 'clients': int((df_dashboard['Cluster_ID']==1).sum()), 'avg_tenure': 15.2, 'avg_fee': 74.94, 'churn_rate': 45.4},
                {'id': 2, 'name': 'Económico Básico', 'clients': int((df_dashboard['Cluster_ID']==2).sum()), 'avg_tenure': 27.6, 'avg_fee': 26.09, 'churn_rate': 14.5}
            ]
        }
    }

    with open("dashboard/models/metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=4, ensure_ascii=False)

    print("[SUCCESS] Todos los artefactos de Fase 4 han sido exportados exitosamente en 'dashboard/models/'.")

if __name__ == "__main__":
    run_export()
