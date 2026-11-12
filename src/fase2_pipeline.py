"""
src/fase2_pipeline.py
======================
Módulo de Preprocesamiento, Ingeniería de Características, Prevención de Data Leakage
y Modelado Base (Baseline) para el Proyecto Integrador de Ciencia de Datos (Fase 2).

Universidad de La Salle - Maestría en Inteligencia Artificial
Asignatura: Ciencia de Datos
Docente: Fabián Camilo Castro Riveros
Estudiante: Yerson León Montañez
"""

import os
from typing import Dict, Tuple, Any, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score,
    roc_curve, precision_recall_curve
)
from imblearn.over_sampling import SMOTE


# -----------------------------------------------------------------------------
# 1. CARGA, AUDITORÍA DE NULOS Y LIMPIEZA
# -----------------------------------------------------------------------------

def load_and_audit_raw_data(filepath: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carga el dataset crudo y extrae los registros con valores en blanco en TotalCharges
    para documentar rigurosamente el mecanismo de falta (MAR vs MCAR/MNAR).
    """
    df = pd.read_csv(filepath)
    # Detectar espacios en blanco o cadenas vacías en TotalCharges
    blank_mask = df['TotalCharges'].astype(str).str.strip() == ''
    missing_records = df[blank_mask].copy()
    
    # Imputación determinista justificada por dominio: si tenure == 0 -> TotalCharges = 0.0
    df['TotalCharges_Clean'] = df['TotalCharges'].replace(r'^\s*$', np.nan, regex=True).astype(float)
    df['TotalCharges_Clean'] = df['TotalCharges_Clean'].fillna(0.0)
    
    return df, missing_records


# -----------------------------------------------------------------------------
# 2. INGENIERÍA DE CARACTERÍSTICAS (DOMAIN FEATURE ENGINEERING)
# -----------------------------------------------------------------------------

class DomainFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Transformador personalizado para generación de variables derivadas respaldadas
    en el conocimiento del dominio de telecomunicaciones:
    1. TotalServices: Conteo de servicios activos de suscripción (1 a 9).
    2. FeeDiscrepancyRatio: Razón entre el cobro mensual actual y la tarifa promedio histórica.
    3. AutomaticPayment: Indicador binario si el método de pago es domiciliado/automático.
    4. LongTermContract: Indicador si posee contrato a 1 o 2 años vs mensual.
    """
    def __init__(self, epsilon: float = 1e-5):
        self.epsilon = epsilon
        self.service_columns = [
            'PhoneService', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies'
        ]

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        
        # 1. TotalServices: Conteo de servicios activos contratados
        def count_services(row):
            count = 0
            if row.get('PhoneService') == 'Yes':
                count += 1
            if row.get('MultipleLines') == 'Yes':
                count += 1
            if row.get('InternetService') in ['DSL', 'Fiber optic']:
                count += 1
            for col in ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                        'TechSupport', 'StreamingTV', 'StreamingMovies']:
                if row.get(col) == 'Yes':
                    count += 1
            return count

        X_out['TotalServices'] = X_out.apply(count_services, axis=1)
        
        # 2. Tarifa Histórica Promedio y Ratio de Discrepancia Tarifaria
        effective_tenure = np.maximum(X_out['tenure'].values, 1.0)
        avg_historical = np.where(
            X_out['tenure'].values == 0,
            X_out['MonthlyCharges'].values,
            X_out['TotalCharges_Clean'].values / effective_tenure
        )
        X_out['AvgHistoricalFee'] = avg_historical
        ratio = X_out['MonthlyCharges'].values / (avg_historical + self.epsilon)
        X_out['FeeDiscrepancyRatio'] = np.clip(ratio, 0.2, 5.0)
        
        # 3. AutomaticPayment: 1 si es transferencia bancaria o tarjeta de crédito automática
        auto_methods = ['Bank transfer (automatic)', 'Credit card (automatic)']
        X_out['AutomaticPayment'] = X_out['PaymentMethod'].isin(auto_methods).astype(int)
        
        # 4. LongTermContract: 1 si es One year o Two year, 0 si es Month-to-month
        X_out['LongTermContract'] = X_out['Contract'].isin(['One year', 'Two year']).astype(int)
        
        return X_out


# -----------------------------------------------------------------------------
# 3. ESTRATEGIA DE PARTICIÓN SIN FUGA DE DATOS (DATA LEAKAGE PREVENTION)
# -----------------------------------------------------------------------------

def split_dataset_stratified(
    df: pd.DataFrame,
    target_col: str = 'Churn',
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Divide el dataset en tres particiones estrictas:
    - Entrenamiento: 70%
    - Validación: 15%
    - Prueba: 15%
    Aplica estratificación sobre la variable objetivo para preservar el ratio de clases.
    """
    y = df[target_col].copy()
    if target_col == 'Churn':
        y_binary = (y == 'Yes').astype(int)
    else:
        y_binary = y

    # Primera partición: Train+Val (85%) vs Test (15%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        df, y_binary,
        test_size=test_size,
        random_state=random_state,
        stratify=y_binary if target_col == 'Churn' else None
    )

    # Segunda partición: Train (70% del total = 0.70/0.85 de train_val) vs Val (15% del total = 0.15/0.85)
    relative_val_size = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=relative_val_size,
        random_state=random_state,
        stratify=y_train_val if target_col == 'Churn' else None
    )

    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test
    }


# -----------------------------------------------------------------------------
# 4. CONSTRUCCIÓN DE PIPELINES DE CODIFICACIÓN Y ESCALAMIENTO
# -----------------------------------------------------------------------------

def build_feature_preprocessor(scaler_type: str = 'standard') -> ColumnTransformer:
    """
    Construye el ColumnTransformer de scikit-learn garantizando:
    - Estandarización o escalamiento robusto de cuantitativas ajustado únicamente en Train.
    - One-Hot Encoding de nominales multiclase sin multicolinealidad (drop='first').
    - Codificación ordinal para la jerarquía de contratos.
    - Passthrough para variables binarias ya codificadas.
    """
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges_Clean', 'TotalServices', 'AvgHistoricalFee', 'FeeDiscrepancyRatio']
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


def preprocess_dataframe_binary_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Mapea variables binarias de texto ('Yes'/'No', 'Female'/'Male') a 0/1."""
    df_out = df.copy()
    binary_cols = {
        'Partner': 'Partner_Binary',
        'Dependents': 'Dependents_Binary',
        'PhoneService': 'PhoneService_Binary',
        'PaperlessBilling': 'PaperlessBilling_Binary'
    }
    for col, new_col in binary_cols.items():
        if col in df_out.columns:
            df_out[new_col] = (df_out[col] == 'Yes').astype(int)
    return df_out


# -----------------------------------------------------------------------------
# 5. MODELADO BASELINE Y EVALUACIÓN EXHAUSTIVA
# -----------------------------------------------------------------------------

def train_and_evaluate_classification_baselines(
    X_train_df: pd.DataFrame, y_train: np.ndarray,
    X_val_df: pd.DataFrame, y_val: np.ndarray,
    X_test_df: pd.DataFrame, y_test: np.ndarray
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Entrena y compara exhaustivamente cuatro configuraciones de referencia:
    1. Dummy Classifier (Estratificado aleatorio)
    2. Dummy Classifier (Mayoría - Most Frequent)
    3. Regresión Logística Estándar (Sin balanceo)
    4. Regresión Logística con Ponderación de Clases (class_weight='balanced')
    5. Regresión Logística entrenada con sobremuestreo SMOTE en Train
    """
    engineer = DomainFeatureEngineer()
    
    # Aplicar ingeniería de variables de manera independiente por conjunto
    X_tr_feat = engineer.transform(preprocess_dataframe_binary_cols(X_train_df))
    X_va_feat = engineer.transform(preprocess_dataframe_binary_cols(X_val_df))
    X_te_feat = engineer.transform(preprocess_dataframe_binary_cols(X_test_df))

    # Preprocesador
    preprocessor = build_feature_preprocessor(scaler_type='standard')
    
    # Ajustar ÚNICAMENTE en Train
    X_tr_proc = preprocessor.fit_transform(X_tr_feat)
    X_va_proc = preprocessor.transform(X_va_feat)
    X_te_proc = preprocessor.transform(X_te_feat)

    # Nombres de columnas transformadas para trazabilidad
    nom_cols = preprocessor.named_transformers_['nom'].get_feature_names_out().tolist()
    feature_names = (
        ['tenure', 'MonthlyCharges', 'TotalCharges_Clean', 'TotalServices', 'AvgHistoricalFee', 'FeeDiscrepancyRatio'] +
        ['Contract_Ordinal'] +
        nom_cols +
        ['SeniorCitizen', 'Partner_Binary', 'Dependents_Binary', 'PhoneService_Binary', 
         'PaperlessBilling_Binary', 'AutomaticPayment', 'LongTermContract']
    )

    # Balanceo SMOTE aplicado exclusivamente en X_train procesado
    smote = SMOTE(random_state=42)
    X_tr_proc_smote, y_tr_smote = smote.fit_resample(X_tr_proc, y_train)

    models = {
        'Dummy (Stratified)': DummyClassifier(strategy='stratified', random_state=42),
        'Dummy (Most Frequent)': DummyClassifier(strategy='most_frequent'),
        'Logistic Regression (Standard)': LogisticRegression(max_iter=1000, random_state=42),
        'Logistic Regression (Balanced)': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
        'Logistic Regression (SMOTE)': LogisticRegression(max_iter=1000, random_state=42)
    }

    results = []
    trained_models = {}

    for name, model in models.items():
        if name == 'Logistic Regression (SMOTE)':
            model.fit(X_tr_proc_smote, y_tr_smote)
        else:
            model.fit(X_tr_proc, y_train)

        trained_models[name] = model

        # Evaluar en Val y Test
        for split_name, X_eval, y_eval in [('Val', X_va_proc, y_val), ('Test', X_te_proc, y_test)]:
            y_pred = model.predict(X_eval)
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X_eval)[:, 1]
                auc_roc = roc_auc_score(y_eval, y_prob)
                avg_prec = average_precision_score(y_eval, y_prob)
            else:
                y_prob = y_pred
                auc_roc = 0.5
                avg_prec = np.mean(y_eval)

            acc = accuracy_score(y_eval, y_pred)
            prec = precision_score(y_eval, y_pred, zero_division=0)
            rec = recall_score(y_eval, y_pred, zero_division=0)
            f1_pos = f1_score(y_eval, y_pred, zero_division=0)
            f1_macro = f1_score(y_eval, y_pred, average='macro', zero_division=0)

            results.append({
                'Modelo': name,
                'Split': split_name,
                'Accuracy': acc,
                'Precision': prec,
                'Recall': rec,
                'F1-Score (Pos)': f1_pos,
                'F1-Score (Macro)': f1_macro,
                'AUC-ROC': auc_roc,
                'PR-AUC': avg_prec
            })

    metrics_df = pd.DataFrame(results)

    data_payload = {
        'trained_models': trained_models,
        'feature_names': feature_names,
        'preprocessor': preprocessor,
        'X_train_proc': X_tr_proc,
        'y_train': y_train,
        'X_val_proc': X_va_proc,
        'y_val': y_val,
        'X_test_proc': X_te_proc,
        'y_test': y_test
    }

    return data_payload, metrics_df


def train_and_evaluate_regression_baselines(
    X_train_df: pd.DataFrame,
    X_val_df: pd.DataFrame,
    X_test_df: pd.DataFrame
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Entrena modelos de referencia para el problema de regresión continua (Predicción de TotalCharges):
    1. Dummy Regressor (Estrategia: Media)
    2. Regresión Lineal Simple Univariada (TotalCharges ~ tenure)
    3. Regresión Lineal Multivariable (TotalCharges ~ tenure + MonthlyCharges + servicios)
    """
    y_train = X_train_df['TotalCharges_Clean'].values
    y_val = X_val_df['TotalCharges_Clean'].values
    y_test = X_test_df['TotalCharges_Clean'].values

    engineer = DomainFeatureEngineer()
    X_tr_feat = engineer.transform(preprocess_dataframe_binary_cols(X_train_df))
    X_va_feat = engineer.transform(preprocess_dataframe_binary_cols(X_val_df))
    X_te_feat = engineer.transform(preprocess_dataframe_binary_cols(X_test_df))

    # Regresor Dummy
    dummy_reg = DummyRegressor(strategy='mean')
    dummy_reg.fit(X_tr_feat, y_train)

    # Regresor Lineal Simple (Univariado: solo tenure)
    lin_simple = LinearRegression()
    lin_simple.fit(X_tr_feat[['tenure']].values, y_train)

    # Regresor Lineal Multivariable (tenure + MonthlyCharges + TotalServices)
    multiv_cols = ['tenure', 'MonthlyCharges', 'TotalServices']
    scaler = StandardScaler()
    X_tr_multi = scaler.fit_transform(X_tr_feat[multiv_cols])
    X_va_multi = scaler.transform(X_va_feat[multiv_cols])
    X_te_multi = scaler.transform(X_te_feat[multiv_cols])

    lin_multi = LinearRegression()
    lin_multi.fit(X_tr_multi, y_train)

    reg_models = {
        'Dummy Regressor (Mean)': (dummy_reg, lambda X: X),
        'Linear Regression (Univariate: tenure)': (lin_simple, lambda X: X[['tenure']].values),
        'Linear Regression (Multivariate)': (lin_multi, lambda X: scaler.transform(X[multiv_cols]))
    }

    results = []
    for name, (model, prep_fn) in reg_models.items():
        for split_name, X_split, y_split in [('Val', X_va_feat, y_val), ('Test', X_te_feat, y_test)]:
            X_in = prep_fn(X_split)
            y_pred = model.predict(X_in)
            r2 = r2_score(y_split, y_pred)
            rmse = np.sqrt(mean_squared_error(y_split, y_pred))
            mae = mean_absolute_error(y_split, y_pred)

            results.append({
                'Modelo': name,
                'Split': split_name,
                'R2 Score': r2,
                'RMSE (USD)': rmse,
                'MAE (USD)': mae
            })

    metrics_df = pd.DataFrame(results)
    payload = {
        'models': {
            'dummy': dummy_reg,
            'simple': lin_simple,
            'multivariate': lin_multi
        },
        'scaler': scaler,
        'multiv_cols': multiv_cols,
        'X_val_feat': X_va_feat,
        'y_val': y_val,
        'X_test_feat': X_te_feat,
        'y_test': y_test
    }
    return payload, metrics_df


# -----------------------------------------------------------------------------
# 6. GENERACIÓN DE GRÁFICOS ANALÍTICOS DE ALTA RESOLUCIÓN
# -----------------------------------------------------------------------------

def generate_fase2_figures(
    df: pd.DataFrame,
    missing_df: pd.DataFrame,
    class_payload: Dict[str, Any],
    reg_payload: Dict[str, Any],
    output_dir: str = "reports/fase2/figures"
):
    """Genera y guarda todas las figuras diagnósticas requeridas para el informe y notebook."""
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    palette = ['#002B5B', '#C5A059', '#D9534F', '#5CB85C']

    # --- FIGURA 1: DIAGNÓSTICO DEL MECANISMO DE FALTA (MAR) ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    
    # Distribución de tenure por estado de missingness en TotalCharges
    df_diag = df.copy()
    df_diag['TotalCharges_Status'] = np.where(df_diag['TotalCharges'].astype(str).str.strip() == '', 'Faltante (N=11)', 'Observado (N=7032)')
    
    sns.countplot(
        data=df_diag, x='TotalCharges_Status', hue='TotalCharges_Status',
        ax=axes[0], palette=['#D9534F', '#002B5B'], legend=False
    )
    axes[0].set_title('Conteo de Observaciones en TotalCharges', fontweight='bold', fontsize=12)
    axes[0].set_xlabel('Estado de la Variable')
    axes[0].set_ylabel('Frecuencia Absoluta')
    for p in axes[0].patches:
        axes[0].annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='bottom', fontsize=10, xytext=(0, 3), textcoords='offset points')

    # Distribución de tenure en los registros faltantes vs observados
    sns.boxplot(
        data=df_diag, x='TotalCharges_Status', y='tenure', hue='TotalCharges_Status',
        ax=axes[1], palette=['#D9534F', '#002B5B'], width=0.4, legend=False
    )
    axes[1].set_title('Distribución de Antigüedad (tenure) según Estado de TotalCharges', fontweight='bold', fontsize=12)
    axes[1].set_xlabel('Estado de TotalCharges')
    axes[1].set_ylabel('Antigüedad (Meses)')
    axes[1].annotate(
        '100% de faltantes tienen tenure = 0\n' + r'$\rightarrow$ Mecanismo Estrictamente MAR',
        xy=(0, 0), xytext=(0.15, 25),
        arrowprops=dict(facecolor='black', shrink=0.08, width=1.5, headwidth=8),
        fontsize=11, bbox=dict(boxstyle="round,pad=0.4", fc="#FFF3CD", ec="#856404", lw=1.2)
    )

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig1_missing_mar_tenure.png'), dpi=300)
    plt.close(fig)

    # --- FIGURA 2: EVALUACIÓN DE OUTLIERS Y TRANSFORMACIÓN LOGARÍTMICA ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)
    total_charges = df['TotalCharges_Clean'].values
    
    # 1. TotalCharges Original
    sns.histplot(total_charges, kde=True, ax=axes[0], color='#002B5B', bins=35)
    axes[0].set_title(f'TotalCharges Original\n(Asimetría g1 = {pd.Series(total_charges).skew():.3f})', fontweight='bold')
    axes[0].set_xlabel('USD')
    axes[0].set_ylabel('Densidad / Conteo')

    # 2. Transformación Logarítmica log(1 + x)
    log_charges = np.log1p(total_charges)
    sns.histplot(log_charges, kde=True, ax=axes[1], color='#C5A059', bins=35)
    axes[1].set_title(f'Transformación Logarítmica log1p\n(Asimetría g1 = {pd.Series(log_charges).skew():.3f})', fontweight='bold')
    axes[1].set_xlabel('log(1 + USD)')
    axes[1].set_ylabel('')

    # 3. Recorte por Percentiles (Winsorization p1 - p99)
    p1, p99 = np.percentile(total_charges, [1, 99])
    winsor_charges = np.clip(total_charges, p1, p99)
    sns.histplot(winsor_charges, kde=True, ax=axes[2], color='#5CB85C', bins=35)
    axes[2].set_title(f'Winsorización [P1, P99]\n(Asimetría g1 = {pd.Series(winsor_charges).skew():.3f})', fontweight='bold')
    axes[2].set_xlabel('USD (Recortado)')
    axes[2].set_ylabel('')

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig2_outliers_transformations.png'), dpi=300)
    plt.close(fig)

    # --- FIGURA 3: VARIABLES DERIVADAS E IMPACTO EN DESERCIÓN ---
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), dpi=300)
    eng_df = DomainFeatureEngineer().transform(preprocess_dataframe_binary_cols(df))

    # TotalServices vs Churn
    sns.countplot(data=eng_df, x='TotalServices', hue='Churn', ax=axes[0], palette=['#002B5B', '#D9534F'])
    axes[0].set_title('Deserción según TotalServices Contratados', fontweight='bold')
    axes[0].set_xlabel('Número Total de Servicios')
    axes[0].set_ylabel('Clientes')

    # FeeDiscrepancyRatio vs Churn (Boxplot)
    sns.boxplot(data=eng_df, x='Churn', y='FeeDiscrepancyRatio', hue='Churn', ax=axes[1], palette=['#002B5B', '#D9534F'], width=0.4, showfliers=False, legend=False)
    axes[1].set_title('Ratio de Discrepancia Tarifaria vs Churn\n(Cobro Mensual / Cobro Promedio Histórico)', fontweight='bold')
    axes[1].set_xlabel('Churn')
    axes[1].set_ylabel('FeeDiscrepancyRatio')

    # AutomaticPayment vs Churn
    payment_churn = pd.crosstab(eng_df['AutomaticPayment'], eng_df['Churn'], normalize='index') * 100
    payment_churn.plot(kind='bar', stacked=True, ax=axes[2], color=['#002B5B', '#D9534F'], edgecolor='none', rot=0)
    axes[2].set_title('Tasa de Deserción por Método de Pago\n(0: Manual/Cheque, 1: Domiciliado/Auto)', fontweight='bold')
    axes[2].set_xlabel('Pago Automático')
    axes[2].set_ylabel('Porcentaje (%)')
    axes[2].legend(title='Churn', frameon=True)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig3_engineered_features.png'), dpi=300)
    plt.close(fig)

    # --- FIGURA 4: MATRICES DE CONFUSIÓN (BASELINES DE CLASIFICACIÓN) ---
    models_to_plot = [
        ('Dummy (Stratified)', class_payload['trained_models']['Dummy (Stratified)']),
        ('Logistic Regression (Standard)', class_payload['trained_models']['Logistic Regression (Standard)']),
        ('Logistic Regression (Balanced)', class_payload['trained_models']['Logistic Regression (Balanced)']),
        ('Logistic Regression (SMOTE)', class_payload['trained_models']['Logistic Regression (SMOTE)'])
    ]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5), dpi=300)
    for i, (m_name, m_obj) in enumerate(models_to_plot):
        y_pred_test = m_obj.predict(class_payload['X_test_proc'])
        cm = confusion_matrix(class_payload['y_test'], y_pred_test)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False,
                    xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'],
                    annot_kws={'size': 13, 'weight': 'bold'})
        axes[i].set_title(f'{m_name}\n(Test Split)', fontweight='bold', fontsize=11)
        axes[i].set_xlabel('Predicción')
        if i == 0:
            axes[i].set_ylabel('Valor Real')
        else:
            axes[i].set_ylabel('')

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig4_confusion_matrices.png'), dpi=300)
    plt.close(fig)

    # --- FIGURA 5: CURVAS ROC Y PRECISION-RECALL (BASELINES) ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    test_models = {
        'Dummy Stratified': class_payload['trained_models']['Dummy (Stratified)'],
        'LR Standard': class_payload['trained_models']['Logistic Regression (Standard)'],
        'LR Balanced': class_payload['trained_models']['Logistic Regression (Balanced)'],
        'LR SMOTE': class_payload['trained_models']['Logistic Regression (SMOTE)']
    }
    colors_dict = {'Dummy Stratified': '#777777', 'LR Standard': '#002B5B', 'LR Balanced': '#D9534F', 'LR SMOTE': '#5CB85C'}

    for m_label, m_inst in test_models.items():
        if hasattr(m_inst, 'predict_proba'):
            probs = m_inst.predict_proba(class_payload['X_test_proc'])[:, 1]
            fpr, tpr, _ = roc_curve(class_payload['y_test'], probs)
            auc_val = roc_auc_score(class_payload['y_test'], probs)
            prec_vals, rec_vals, _ = precision_recall_curve(class_payload['y_test'], probs)
            pr_auc_val = average_precision_score(class_payload['y_test'], probs)

            axes[0].plot(fpr, tpr, label=f'{m_label} (AUC = {auc_val:.3f})', color=colors_dict[m_label], lw=2)
            axes[1].plot(rec_vals, prec_vals, label=f'{m_label} (PR-AUC = {pr_auc_val:.3f})', color=colors_dict[m_label], lw=2)

    axes[0].plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.7)
    axes[0].set_title('Curvas ROC (Receiver Operating Characteristic) - Test', fontweight='bold', fontsize=12)
    axes[0].set_xlabel('Tasa de Falsos Positivos (1 - Especificidad)')
    axes[0].set_ylabel('Tasa de Verdaderos Positivos (Sensibilidad / Recall)')
    axes[0].legend(loc='lower right', frameon=True)

    no_skill = np.sum(class_payload['y_test']) / len(class_payload['y_test'])
    axes[1].axhline(y=no_skill, color='k', linestyle='--', lw=1, alpha=0.7, label=f'Línea Base No-Skill ({no_skill:.3f})')
    axes[1].set_title('Curvas Precision-Recall (Evaluación con Desbalance) - Test', fontweight='bold', fontsize=12)
    axes[1].set_xlabel('Recall (Sensibilidad)')
    axes[1].set_ylabel('Precision (Valor Predictivo Positivo)')
    axes[1].legend(loc='upper right', frameon=True)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig5_roc_pr_curves.png'), dpi=300)
    plt.close(fig)

    # --- FIGURA 6: DIAGNÓSTICO DE REGRESIÓN (REAL VS PREDICHO Y RESIDUOS) ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    lin_multi = reg_payload['models']['multivariate']
    scaler = reg_payload['scaler']
    multiv_cols = reg_payload['multiv_cols']
    X_test_mat = scaler.transform(reg_payload['X_test_feat'][multiv_cols])
    y_test_reg = reg_payload['y_test']
    y_pred_reg = lin_multi.predict(X_test_mat)
    residuals = y_test_reg - y_pred_reg

    # Real vs Predicho
    axes[0].scatter(y_test_reg, y_pred_reg, alpha=0.35, color='#002B5B', edgecolor='none')
    min_val = min(y_test_reg.min(), y_pred_reg.min())
    max_val = max(y_test_reg.max(), y_pred_reg.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Predicción Perfecta (y = x)')
    r2_test = r2_score(y_test_reg, y_pred_reg)
    axes[0].set_title(f'Regresión Lineal Multivariable: Real vs. Predicho\n(Test $R^2 = {r2_test:.4f}$)', fontweight='bold', fontsize=12)
    axes[0].set_xlabel('TotalCharges Real (USD)')
    axes[0].set_ylabel('TotalCharges Predicho (USD)')
    axes[0].legend(loc='upper left', frameon=True)

    # Gráfico de Residuos
    axes[1].scatter(y_pred_reg, residuals, alpha=0.35, color='#C5A059', edgecolor='none')
    axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1].set_title('Gráfico de Dispersión de Residuos (Homocedasticidad)', fontweight='bold', fontsize=12)
    axes[1].set_xlabel('Valor Predicho (USD)')
    axes[1].set_ylabel('Residuo ($y_{real} - y_{pred}$)')

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig6_regression_residuals.png'), dpi=300)
    plt.close(fig)

    print(f"[OK] 6 figuras analíticas generadas exitosamente en '{output_dir}'.")
