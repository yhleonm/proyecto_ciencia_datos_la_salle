"""
src/fase3_models.py
===================
Módulo de Modelado Avanzado, Redes Neuronales en PyTorch, Clustering y
Explicabilidad (XAI) con SHAP para la Fase 3 del Proyecto Integrador.

Universidad de La Salle - Maestría en Inteligencia Artificial
Asignatura: Ciencia de Datos
Docente: Fabián Camilo Castro Riveros
Estudiante: Yerson León Montañez
"""

import os
import random
from typing import Dict, Tuple, Any, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    silhouette_score, roc_curve, precision_recall_curve
)
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from xgboost import XGBRegressor, XGBClassifier
import shap

from src.fase2_pipeline import (
    load_and_audit_raw_data,
    preprocess_dataframe_binary_cols,
    DomainFeatureEngineer,
    split_dataset_stratified,
    build_feature_preprocessor
)

from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer

def build_regression_feature_preprocessor(scaler_type: str = 'standard') -> ColumnTransformer:
    """
    Construye el preprocesador para el problema de regresión de TotalCharges.
    EXCLUYE estrictamente TotalCharges_Clean y sus derivadas para prevenir Target Leakage.
    """
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalServices']
    ordinal_features = ['Contract']
    binary_features = ['SeniorCitizen', 'Partner_Binary', 'Dependents_Binary', 'PhoneService_Binary', 
                       'PaperlessBilling_Binary', 'AutomaticPayment', 'LongTermContract']
    nominal_features = ['gender', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                        'StreamingMovies', 'PaymentMethod']

    contract_order = [['Month-to-month', 'One year', 'Two year']]

    numeric_transformer = StandardScaler() if scaler_type == 'standard' else RobustScaler()
    ordinal_transformer = OrdinalEncoder(categories=contract_order, handle_unknown='use_encoded_value', unknown_value=-1)
    nominal_transformer = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('ord', ordinal_transformer, ordinal_features),
            ('nom', nominal_transformer, nominal_features),
            ('bin', 'passthrough', binary_features)
        ],
        remainder='drop'
    )
    return preprocessor

def set_seed(seed: int = 42):
    """Fija la semilla pseudoaleatoria para reproducibilidad estricta."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# -----------------------------------------------------------------------------
# 1. BLOQUE 1: REGRESIÓN REGULARIZADA Y ENSAMBLES (TotalCharges)
# -----------------------------------------------------------------------------

def train_and_evaluate_regression_advanced(
    X_train_proc: np.ndarray, y_train_reg: np.ndarray,
    X_val_proc: np.ndarray, y_val_reg: np.ndarray,
    X_test_proc: np.ndarray, y_test_reg: np.ndarray
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Entrena y compara:
    1. Ridge Regression (L2) con optimización de alpha por CV
    2. Lasso Regression (L1) con optimización de alpha por CV
    3. Random Forest Regressor
    4. XGBoost Regressor
    """
    set_seed(42)

    # 1. Ridge con GridSearchCV
    ridge_params = {'alpha': [0.01, 0.1, 1.0, 10.0, 50.0, 100.0]}
    ridge_cv = GridSearchCV(Ridge(random_state=42), ridge_params, cv=5, scoring='r2', n_jobs=-1)
    ridge_cv.fit(X_train_proc, y_train_reg)
    best_ridge = ridge_cv.best_estimator_

    # 2. Lasso con GridSearchCV
    lasso_params = {'alpha': [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]}
    lasso_cv = GridSearchCV(Lasso(random_state=42, max_iter=5000), lasso_params, cv=5, scoring='r2', n_jobs=-1)
    lasso_cv.fit(X_train_proc, y_train_reg)
    best_lasso = lasso_cv.best_estimator_

    # 3. Random Forest Regressor
    rf_reg = RandomForestRegressor(
        n_estimators=120, max_depth=9, min_samples_split=4, random_state=42, n_jobs=-1
    )
    rf_reg.fit(X_train_proc, y_train_reg)

    # 4. XGBoost Regressor
    xgb_reg = XGBRegressor(
        n_estimators=120, max_depth=4, learning_rate=0.07, subsample=0.85,
        colsample_bytree=0.85, random_state=42, n_jobs=-1
    )
    xgb_reg.fit(X_train_proc, y_train_reg)

    models = {
        'Ridge Regression (L2)': best_ridge,
        'Lasso Regression (L1)': best_lasso,
        'Random Forest Regressor': rf_reg,
        'XGBoost Regressor': xgb_reg
    }

    results = []
    for name, model in models.items():
        for split_name, X_eval, y_eval in [('Val', X_val_proc, y_val_reg), ('Test', X_test_proc, y_test_reg)]:
            y_pred = model.predict(X_eval)
            r2 = r2_score(y_eval, y_pred)
            rmse = np.sqrt(mean_squared_error(y_eval, y_pred))
            mae = mean_absolute_error(y_eval, y_pred)

            results.append({
                'Modelo': name,
                'Split': split_name,
                'R2 Score': r2,
                'RMSE (USD)': rmse,
                'MAE (USD)': mae
            })

    metrics_df = pd.DataFrame(results)
    payload = {
        'models': models,
        'best_ridge_alpha': ridge_cv.best_params_['alpha'],
        'best_lasso_alpha': lasso_cv.best_params_['alpha'],
        'y_test_reg': y_test_reg,
        'X_test_proc': X_test_proc
    }
    return payload, metrics_df


# -----------------------------------------------------------------------------
# 2. BLOQUE 2: CLASIFICACIÓN AVANZADA Y PERCEPTRÓN MULTICAPA (PyTorch MLP)
# -----------------------------------------------------------------------------

class ChurnMLPNet(nn.Module):
    """
    Arquitectura de Perceptrón Multicapa (MLP) en PyTorch para clasificación binaria:
    Linear -> LayerNorm -> ReLU -> Dropout -> Linear -> ReLU -> Dropout -> Linear (Logits).
    """
    def __init__(self, input_dim: int, hidden_dim1: int = 64, hidden_dim2: int = 32, dropout_rate: float = 0.25):
        super(ChurnMLPNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.LayerNorm(hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def train_pytorch_mlp(
    X_train_proc: np.ndarray, y_train: np.ndarray,
    X_val_proc: np.ndarray, y_val: np.ndarray,
    epochs: int = 50, batch_size: int = 64, lr: float = 0.002
) -> Tuple[ChurnMLPNet, Dict[str, List[float]]]:
    """Entrena la red neuronal MLP en PyTorch registrando la curva de pérdida por época."""
    set_seed(42)

    X_tr_t = torch.tensor(X_train_proc, dtype=torch.float32)
    y_tr_t = torch.tensor(y_train, dtype=torch.float32)
    X_va_t = torch.tensor(X_val_proc, dtype=torch.float32)
    y_va_t = torch.tensor(y_val, dtype=torch.float32)

    train_dataset = TensorDataset(X_tr_t, y_tr_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    input_dim = X_train_proc.shape[1]
    model = ChurnMLPNet(input_dim=input_dim, hidden_dim1=64, hidden_dim2=32, dropout_rate=0.25)

    # Ponderación de clase para desbalance: ratio ~ 2.77
    pos_weight = torch.tensor([2.77], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_x.size(0)

        epoch_train_loss = running_loss / len(train_dataset)

        # Evaluación en validación
        model.eval()
        with torch.no_grad():
            val_logits = model(X_va_t)
            val_loss = criterion(val_logits, y_va_t).item()

        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(val_loss)

    return model, history


def train_and_evaluate_classification_advanced(
    X_train_proc: np.ndarray, y_train: np.ndarray,
    X_val_proc: np.ndarray, y_val: np.ndarray,
    X_test_proc: np.ndarray, y_test: np.ndarray,
    feature_names: List[str]
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Entrena y compara:
    1. Random Forest Classifier (con ajuste de balanceo)
    2. XGBoost Classifier (con scale_pos_weight)
    3. Perceptrón Multicapa (MLP) en PyTorch
    """
    set_seed(42)

    # 1. Random Forest Classifier
    rf_clf = RandomForestClassifier(
        n_estimators=150, max_depth=7, min_samples_split=5,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    rf_clf.fit(X_train_proc, y_train)

    # 2. XGBoost Classifier
    xgb_clf = XGBClassifier(
        n_estimators=140, max_depth=4, learning_rate=0.06,
        subsample=0.85, colsample_bytree=0.85, scale_pos_weight=2.77,
        random_state=42, eval_metric='logloss', n_jobs=-1
    )
    xgb_clf.fit(X_train_proc, y_train)

    # 3. PyTorch MLP
    mlp_model, mlp_history = train_pytorch_mlp(X_train_proc, y_train, X_val_proc, y_val, epochs=50)

    # Función auxiliar para predecir probabilidades en PyTorch
    def predict_mlp_proba(model: nn.Module, X: np.ndarray) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            t_in = torch.tensor(X, dtype=torch.float32)
            logits = model(t_in)
            probs = torch.sigmoid(logits).numpy()
        return probs

    models_dict = {
        'Random Forest (Balanced)': (rf_clf, lambda m, X: m.predict_proba(X)[:, 1]),
        'XGBoost Classifier': (xgb_clf, lambda m, X: m.predict_proba(X)[:, 1]),
        'PyTorch MLP': (mlp_model, lambda m, X: predict_mlp_proba(m, X))
    }

    results = []
    for name, (m_obj, prob_fn) in models_dict.items():
        for split_name, X_eval, y_eval in [('Val', X_val_proc, y_val), ('Test', X_test_proc, y_test)]:
            probs = prob_fn(m_obj, X_eval)
            preds = (probs >= 0.50).astype(int)

            acc = accuracy_score(y_eval, preds)
            prec = precision_score(y_eval, preds, zero_division=0)
            rec = recall_score(y_eval, preds, zero_division=0)
            f1_pos = f1_score(y_eval, preds, zero_division=0)
            f1_macro = f1_score(y_eval, preds, average='macro', zero_division=0)
            auc_roc = roc_auc_score(y_eval, probs)
            pr_auc = average_precision_score(y_eval, probs)

            results.append({
                'Modelo': name,
                'Split': split_name,
                'Accuracy': acc,
                'Precision': prec,
                'Recall': rec,
                'F1-Score (Pos)': f1_pos,
                'F1-Score (Macro)': f1_macro,
                'AUC-ROC': auc_roc,
                'PR-AUC': pr_auc
            })

    metrics_df = pd.DataFrame(results)
    payload = {
        'rf_clf': rf_clf,
        'xgb_clf': xgb_clf,
        'mlp_model': mlp_model,
        'mlp_history': mlp_history,
        'feature_names': feature_names,
        'X_test_proc': X_test_proc,
        'y_test': y_test
    }
    return payload, metrics_df


# -----------------------------------------------------------------------------
# 3. BLOQUE 3: APRENDIZAJE NO SUPERVISADO (K-MEANS Y JERÁRQUICO)
# -----------------------------------------------------------------------------

def analyze_clustering_unsupervised(df_featured: pd.DataFrame) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Ejecuta el análisis de segmentación no supervisada:
    1. Estandarización de variables clave de comportamiento y consumo.
    2. Evaluación de K-Means para K en [2, 8] calculando Inercia (Codo) y Coeficiente de Silueta.
    3. Selección del K óptimo y ajuste del modelo final K-Means.
    4. Implementación de Agglomerative Clustering (Jerárquico) con Ward Linkage.
    5. Perfilamiento estadístico de negocio de los clústeres.
    """
    set_seed(42)

    clustering_cols = ['tenure', 'MonthlyCharges', 'TotalCharges_Clean', 'TotalServices', 'FeeDiscrepancyRatio']
    X_clust = df_featured[clustering_cols].values
    scaler = StandardScaler()
    X_clust_scaled = scaler.fit_transform(X_clust)

    k_range = list(range(2, 9))
    inertias = []
    silhouette_scores = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_clust_scaled)
        inertias.append(km.inertia_)
        sil = silhouette_score(X_clust_scaled, km.labels_)
        silhouette_scores.append(sil)

    # Selección de K óptimo: K = 3 o 4 (en telecomunicaciones K=3 balancea silueta y parsimonia operativa)
    optimal_k = 3
    final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=15)
    kmeans_labels = final_kmeans.fit_predict(X_clust_scaled)

    # Segundo Algoritmo: Clustering Jerárquico Aglomerativo
    agg_clust = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward')
    hierarchical_labels = agg_clust.fit_predict(X_clust_scaled)
    sil_hierarchical = silhouette_score(X_clust_scaled, hierarchical_labels)

    # Asignar clústeres al dataframe para perfilamiento
    df_clustered = df_featured.copy()
    df_clustered['KMeans_Cluster'] = kmeans_labels
    df_clustered['Hierarchical_Cluster'] = hierarchical_labels

    # Perfiles promedio de negocio por Clúster K-Means
    profile_df = df_clustered.groupby('KMeans_Cluster').agg(
        Clientes=('customerID', 'count'),
        Tenure_Promedio=('tenure', 'mean'),
        MonthlyCharges_Promedio=('MonthlyCharges', 'mean'),
        TotalCharges_Promedio=('TotalCharges_Clean', 'mean'),
        TotalServices_Promedio=('TotalServices', 'mean'),
        FeeDiscrepancy_Promedio=('FeeDiscrepancyRatio', 'mean'),
        Tasa_Desercion_Churn=('Churn', lambda s: (s == 'Yes').mean() * 100),
        Adopcion_PagoAuto=('AutomaticPayment', lambda s: s.mean() * 100),
        Contrato_LargoPlazo=('LongTermContract', lambda s: s.mean() * 100)
    ).reset_index()

    # Reducción PCA para visualización 2D
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_clust_scaled)
    df_clustered['PCA1'] = X_pca[:, 0]
    df_clustered['PCA2'] = X_pca[:, 1]

    payload = {
        'k_range': k_range,
        'inertias': inertias,
        'silhouette_scores': silhouette_scores,
        'optimal_k': optimal_k,
        'final_kmeans': final_kmeans,
        'kmeans_labels': kmeans_labels,
        'sil_hierarchical': sil_hierarchical,
        'df_clustered': df_clustered,
        'profile_df': profile_df,
        'pca': pca,
        'clustering_cols': clustering_cols
    }

    return payload, profile_df


# -----------------------------------------------------------------------------
# 4. BLOQUE 4: INTERPRETABILIDAD CON SHAP (XAI)
# -----------------------------------------------------------------------------

def compute_shap_explanations(
    xgb_clf: XGBClassifier,
    X_test_proc: np.ndarray,
    feature_names: List[str]
) -> Tuple[np.ndarray, shap.Explainer]:
    """Calcula los valores SHAP sobre el modelo de gradiente impulsado XGBoost en el conjunto Test."""
    explainer = shap.TreeExplainer(xgb_clf)
    shap_values = explainer.shap_values(X_test_proc)
    return shap_values, explainer


# -----------------------------------------------------------------------------
# 5. GENERACIÓN DE FIGURAS ANALÍTICAS DE ALTA RESOLUCIÓN (Fase 3)
# -----------------------------------------------------------------------------

def generate_fase3_figures(
    reg_payload: Dict[str, Any],
    clf_payload: Dict[str, Any],
    clust_payload: Dict[str, Any],
    shap_values: np.ndarray,
    output_dir: str = "reports/fase3/figures"
):
    """Genera y guarda las 6 figuras diagnósticas requeridas para el informe y notebook."""
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # --- FIGURA 1: COMPARATIVA DE REGRESIÓN (R2 Y RMSE) ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    models_reg = reg_payload['models']
    y_test = reg_payload['y_test_reg']
    X_test = reg_payload['X_test_proc']

    names = list(models_reg.keys())
    r2_vals = [r2_score(y_test, m.predict(X_test)) for m in models_reg.values()]
    rmse_vals = [np.sqrt(mean_squared_error(y_test, m.predict(X_test))) for m in models_reg.values()]

    # R2 Barplot
    sns.barplot(x=r2_vals, y=names, hue=names, ax=axes[0], palette=['#002B5B', '#2A6F97', '#014F86', '#C5A059'], legend=False)
    axes[0].set_title(r'Coeficiente de Determinación $R^2$ en Conjunto de Prueba', fontweight='bold')
    axes[0].set_xlabel(r'$R^2$ Score')
    for i, v in enumerate(r2_vals):
        axes[0].text(v - 0.08, i, f'{v:.4f}', va='center', color='white', fontweight='bold', fontsize=10)

    # RMSE Barplot
    sns.barplot(x=rmse_vals, y=names, hue=names, ax=axes[1], palette=['#002B5B', '#2A6F97', '#014F86', '#C5A059'], legend=False)
    axes[1].set_title('Raíz del Error Cuadrático Medio (RMSE) en Prueba', fontweight='bold')
    axes[1].set_xlabel('USD')
    for i, v in enumerate(rmse_vals):
        axes[1].text(v - 90, i, f'${v:.1f}', va='center', color='white', fontweight='bold', fontsize=10)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig1_regression_comparison.png'), dpi=300)
    plt.close(fig)

    # --- FIGURA 2: CURVAS ROC Y PRECISION-RECALL (CLASIFICACIÓN) ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    y_test_clf = clf_payload['y_test']
    X_test_clf = clf_payload['X_test_proc']

    # Obtener probabilidades
    rf_probs = clf_payload['rf_clf'].predict_proba(X_test_clf)[:, 1]
    xgb_probs = clf_payload['xgb_clf'].predict_proba(X_test_clf)[:, 1]
    
    clf_payload['mlp_model'].eval()
    with torch.no_grad():
        mlp_probs = torch.sigmoid(clf_payload['mlp_model'](torch.tensor(X_test_clf, dtype=torch.float32))).numpy()

    curves_data = {
        'Random Forest (Balanced)': (rf_probs, '#002B5B'),
        'XGBoost Classifier': (xgb_probs, '#C5A059'),
        'PyTorch MLP': (mlp_probs, '#D9534F')
    }

    for label, (probs, col) in curves_data.items():
        fpr, tpr, _ = roc_curve(y_test_clf, probs)
        auc_score = roc_auc_score(y_test_clf, probs)
        axes[0].plot(fpr, tpr, label=f'{label} (AUC = {auc_score:.3f})', color=col, lw=2.2)

        prec_line, rec_line, _ = precision_recall_curve(y_test_clf, probs)
        pr_score = average_precision_score(y_test_clf, probs)
        axes[1].plot(rec_line, prec_line, label=f'{label} (PR-AUC = {pr_score:.3f})', color=col, lw=2.2)

    axes[0].plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.7)
    axes[0].set_title('Curvas ROC (Receiver Operating Characteristic) - Test', fontweight='bold')
    axes[0].set_xlabel('Tasa de Falsos Positivos')
    axes[0].set_ylabel('Sensibilidad / Recall')
    axes[0].legend(loc='lower right', frameon=True)

    no_skill = np.sum(y_test_clf) / len(y_test_clf)
    axes[1].axhline(y=no_skill, color='k', linestyle='--', lw=1, alpha=0.7, label=f'No-Skill ({no_skill:.3f})')
    axes[1].set_title('Curvas Precision-Recall - Test', fontweight='bold')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].legend(loc='upper right', frameon=True)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig2_classification_roc_pr.png'), dpi=300)
    plt.close(fig)

    # --- FIGURA 3: CURVA DE PÉRDIDA EN PYTORCH (TRAIN VS VALIDATION) ---
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    history = clf_payload['mlp_history']
    epochs_arr = range(1, len(history['train_loss']) + 1)

    ax.plot(epochs_arr, history['train_loss'], label='Pérdida Entrenamiento (Train Loss)', color='#002B5B', lw=2.2)
    ax.plot(epochs_arr, history['val_loss'], label='Pérdida Validación (Validation Loss)', color='#D9534F', lw=2.2, linestyle='--')
    ax.set_title('Curva de Convergencia de Pérdida (BCE Loss) en Perceptrón Multicapa PyTorch', fontweight='bold')
    ax.set_xlabel('Época de Entrenamiento')
    ax.set_ylabel('Función de Pérdida (Binary Cross Entropy con Pesos)')
    ax.legend(frameon=True, loc='upper right')
    ax.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig3_pytorch_loss_curve.png'), dpi=300)
    plt.close(fig)

    # --- FIGURA 4: K-MEANS - MÉTODO DEL CODO Y COEFICIENTE DE SILUETA ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    k_range = clust_payload['k_range']
    inertias = clust_payload['inertias']
    sil_scores = clust_payload['silhouette_scores']

    # Codo
    axes[0].plot(k_range, inertias, 'o-', color='#002B5B', lw=2.2, markersize=7)
    axes[0].axvline(x=3, color='#D9534F', linestyle='--', label='K óptimo = 3 (Punto de inflexión)')
    axes[0].set_title('Método del Codo (Inercia WCSS vs. K)', fontweight='bold')
    axes[0].set_xlabel('Número de Clústeres (K)')
    axes[0].set_ylabel('Inercia Intra-Clúster (WCSS)')
    axes[0].legend(frameon=True)

    # Silueta
    axes[1].plot(k_range, sil_scores, 's-', color='#C5A059', lw=2.2, markersize=7)
    axes[1].axvline(x=3, color='#D9534F', linestyle='--', label='K óptimo = 3 (Máxima compacidad)')
    axes[1].set_title('Coeficiente Promedio de Silueta vs. K', fontweight='bold')
    axes[1].set_xlabel('Número de Clústeres (K)')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].legend(frameon=True)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig4_clustering_elbow_silhouette.png'), dpi=300)
    plt.close(fig)

    # --- FIGURA 5: MAPA 2D PCA DE CLÚSTERES Y PERFILES DE NEGOCIO ---
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), dpi=300)
    df_clust = clust_payload['df_clustered']
    prof_df = clust_payload['profile_df']

    # Scatter PCA 2D
    scatter_palette = ['#002B5B', '#C5A059', '#D9534F']
    sns.scatterplot(
        data=df_clust, x='PCA1', y='PCA2', hue='KMeans_Cluster',
        palette=scatter_palette, alpha=0.5, edgecolor='none', ax=axes[0]
    )
    axes[0].set_title('Espacio Reducido PCA (2D) de Segmentación K-Means', fontweight='bold')
    axes[0].set_xlabel(f"Componente Principal 1 ({clust_payload['pca'].explained_variance_ratio_[0]*100:.1f}% Varianza)")
    axes[0].set_ylabel(f"Componente Principal 2 ({clust_payload['pca'].explained_variance_ratio_[1]*100:.1f}% Varianza)")
    axes[0].legend(title='Clúster', frameon=True)

    # Perfiles en gráfico de barras
    cluster_labels = [f"Clúster {r['KMeans_Cluster']}\n({r['Clientes']} cli)" for _, r in prof_df.iterrows()]
    bar_width = 0.35
    x_idx = np.arange(len(cluster_labels))

    ax2_twin = axes[1].twinx()
    b1 = axes[1].bar(x_idx - bar_width/2, prof_df['Tenure_Promedio'], width=bar_width, color='#002B5B', label='Antigüedad (Meses)')
    b2 = ax2_twin.bar(x_idx + bar_width/2, prof_df['Tasa_Desercion_Churn'], width=bar_width, color='#D9534F', label='Tasa Deserción (%)')

    axes[1].set_xticks(x_idx)
    axes[1].set_xticklabels(cluster_labels, fontweight='bold')
    axes[1].set_ylabel('Antigüedad Media (Meses)', color='#002B5B', fontweight='bold')
    ax2_twin.set_ylabel('Tasa de Deserción (% Churn)', color='#D9534F', fontweight='bold')
    axes[1].set_title('Perfil Comparativo de Negocio: Antigüedad vs. Riesgo de Churn', fontweight='bold')

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig5_cluster_profiles.png'), dpi=300)
    plt.close(fig)

    # --- FIGURA 6: INTERPRETABILIDAD SHAP (BEESWARM SUMMARY PLOT) ---
    fig = plt.figure(figsize=(10, 6), dpi=300)
    feature_names = clf_payload['feature_names']
    X_test_df = pd.DataFrame(clf_payload['X_test_proc'], columns=feature_names)
    
    shap.summary_plot(
        shap_values, X_test_df, max_display=12,
        plot_type="dot", show=False, color_bar=True
    )
    plt.title('Importancia Global de Características e Impacto Marginal (SHAP Summary Plot)', fontweight='bold', fontsize=11, pad=15)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig6_shap_summary.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"[OK] 6 figuras analíticas de Fase 3 guardadas exitosamente en '{output_dir}'.")
