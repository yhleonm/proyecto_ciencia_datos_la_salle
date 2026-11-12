"""
Módulo de Utilidades para Análisis Exploratorio de Datos (EDA)
Proyecto Integrador - Ciencia de Datos e Inteligencia Artificial
Universidad de La Salle
Autor: Yerson León Montañez (yeleon36@unisalle.edu.co)
"""

import os
import json
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración estética de gráficos
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['figure.dpi'] = 300


class TelcoEDA:
    """Clase principal para procesamiento estadístico y generación de reportes EDA."""

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.raw_df = pd.read_csv(data_path)
        self.df = self.raw_df.copy()
        self._prepare_data()

    def _prepare_data(self):
        """Conversión de tipos y detección de nulos encubiertos."""
        # TotalCharges contiene 11 espacios en blanco ' ' cuando tenure == 0
        self.df['TotalCharges_cleaned'] = pd.to_numeric(self.df['TotalCharges'].replace(' ', np.nan), errors='coerce')
        self.num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges_cleaned']
        self.cat_cols = [
            'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
            'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
            'Contract', 'PaperlessBilling', 'PaymentMethod', 'Churn'
        ]

    def run_data_quality_audit(self) -> dict:
        """Auditoría exhaustiva de calidad: nulos, duplicados y anomalías."""
        n_rows = len(self.raw_df)
        duplicates = int(self.raw_df.duplicated().sum())
        id_duplicates = int(self.raw_df['customerID'].duplicated().sum())
        
        # Conteo de nulos y espacios en blanco
        missing_report = {}
        for col in self.raw_df.columns:
            if col == 'TotalCharges':
                whitespace_count = int((self.raw_df[col].astype(str).str.strip() == '').sum())
                nan_count = int(self.raw_df[col].isna().sum()) + whitespace_count
            else:
                nan_count = int(self.raw_df[col].isna().sum())
            missing_report[col] = {
                'missing_count': nan_count,
                'missing_pct': round((nan_count / n_rows) * 100, 4)
            }
            
        return {
            'total_records': n_rows,
            'total_columns': len(self.raw_df.columns),
            'duplicate_records': duplicates,
            'duplicate_customer_ids': id_duplicates,
            'missing_report': missing_report
        }

    def compute_univariate_quantitative(self) -> pd.DataFrame:
        """Métricas de tendencia central, dispersión, asimetría y curtosis."""
        metrics = []
        for col in self.num_cols:
            series = self.df[col].dropna()
            mean_val = series.mean()
            median_val = series.median()
            std_val = series.std(ddof=1)
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr_val = q3 - q1
            min_val = series.min()
            max_val = series.max()
            skew_val = stats.skew(series, bias=False)
            kurt_val = stats.kurtosis(series, bias=False)  # Exceso de curtosis (0 = normal)
            
            metrics.append({
                'Variable': col.replace('_cleaned', ''),
                'N_Validos': len(series),
                'Media': round(mean_val, 2),
                'Mediana': round(median_val, 2),
                'Desv_Std': round(std_val, 2),
                'Q1': round(q1, 2),
                'Q3': round(q3, 2),
                'IQR': round(iqr_val, 2),
                'Min': round(min_val, 2),
                'Max': round(max_val, 2),
                'Asimetria': round(skew_val, 4),
                'Curtosis': round(kurt_val, 4)
            })
        return pd.DataFrame(metrics)

    def detect_outliers(self) -> pd.DataFrame:
        """Detección de outliers mediante rango intercuartílico (IQR) y Z-Score (|Z| > 3)."""
        outlier_data = []
        for col in self.num_cols:
            series = self.df[col].dropna()
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower_iqr = q1 - 1.5 * iqr
            upper_iqr = q3 + 1.5 * iqr
            outliers_iqr = series[(series < lower_iqr) | (series > upper_iqr)]
            
            z_scores = np.abs(stats.zscore(series))
            outliers_z = series[z_scores > 3.0]
            
            outlier_data.append({
                'Variable': col.replace('_cleaned', ''),
                'IQR_Lim_Inf': round(lower_iqr, 2),
                'IQR_Lim_Sup': round(upper_iqr, 2),
                'IQR_Atipicos_N': len(outliers_iqr),
                'IQR_Atipicos_Pct': round(len(outliers_iqr) / len(series) * 100, 3),
                'Z_Score_Atipicos_N': len(outliers_z),
                'Z_Score_Atipicos_Pct': round(len(outliers_z) / len(series) * 100, 3)
            })
        return pd.DataFrame(outlier_data)

    def compute_bivariate_categorical(self, target_col: str = 'Churn') -> pd.DataFrame:
        """Pruebas de independencia Chi-cuadrado y cálculo de V de Cramér."""
        results = []
        for col in self.cat_cols:
            if col == target_col:
                continue
            contingency = pd.crosstab(self.df[col], self.df[target_col])
            chi2, p_val, dof, _ = stats.chi2_contingency(contingency)
            n = contingency.sum().sum()
            min_dim = min(contingency.shape) - 1
            cramer_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0.0
            
            results.append({
                'Variable': col,
                'Chi2_Stat': round(chi2, 2),
                'Grados_Libertad': dof,
                'p_valor': p_val,
                'Cramer_V': round(cramer_v, 4),
                'Asociacion': 'Fuerte' if cramer_v > 0.3 else ('Moderada' if cramer_v > 0.15 else 'Debil')
            })
        df_chi = pd.DataFrame(results).sort_values(by='Cramer_V', ascending=False)
        return df_chi

    def generate_all_figures(self, output_dir: str):
        """Generación y guardado de visualizaciones para el informe y notebook."""
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Distribución Univariada Cuantitativa (Histogramas + KDE)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        col_names = [('tenure', 'Antigüedad (Meses)'), 
                     ('MonthlyCharges', 'Cargos Mensuales (USD)'), 
                     ('TotalCharges_cleaned', 'Cargos Totales (USD)')]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for ax, (col, title), color in zip(axes, col_names, colors):
            series = self.df[col].dropna()
            sns.histplot(series, kde=True, ax=ax, color=color, stat='density', alpha=0.5, edgecolor=color)
            ax.axvline(series.mean(), color='red', linestyle='--', linewidth=1.5, label=f'Media: {series.mean():.1f}')
            ax.axvline(series.median(), color='black', linestyle=':', linewidth=1.5, label=f'Mediana: {series.median():.1f}')
            ax.set_title(title, fontweight='bold')
            ax.set_xlabel('Valor')
            ax.set_ylabel('Densidad')
            ax.legend(frameon=True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'fig1_univariate_numerical.png'), dpi=300)
        plt.close()

        # 2. Boxplots para diagnóstico de Outliers (IQR)
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for ax, (col, title), color in zip(axes, col_names, colors):
            sns.boxplot(y=self.df[col], ax=ax, color=color, width=0.35, fliersize=4)
            ax.set_title(f'Diagrama de Caja: {title}', fontweight='bold')
            ax.set_ylabel('Magnitud')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'fig2_outliers_boxplots.png'), dpi=300)
        plt.close()

        # 3. Matriz de Correlación Numérica (Pearson vs Spearman)
        num_clean = self.df[['tenure', 'MonthlyCharges', 'TotalCharges_cleaned']].dropna()
        num_clean.columns = ['tenure', 'MonthlyCharges', 'TotalCharges']
        
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        sns.heatmap(num_clean.corr(method='pearson'), annot=True, fmt='.3f', cmap='Blues', ax=axes[0], vmin=-1, vmax=1)
        axes[0].set_title('Correlación Lineal de Pearson', fontweight='bold')
        sns.heatmap(num_clean.corr(method='spearman'), annot=True, fmt='.3f', cmap='Greens', ax=axes[1], vmin=-1, vmax=1)
        axes[1].set_title('Correlación Monótona de Spearman', fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'fig3_correlation_matrices.png'), dpi=300)
        plt.close()

        # 4. Tasa de Deserción (Churn) según factores estructurales clave
        key_cats = ['Contract', 'InternetService', 'TechSupport', 'PaymentMethod']
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for ax, col in zip(axes, key_cats):
            ct = pd.crosstab(self.df[col], self.df['Churn'], normalize='index') * 100
            ct[['No', 'Yes']].plot(kind='bar', stacked=True, ax=ax, color=['#4CAF50', '#E53935'], alpha=0.85)
            ax.set_title(f'Tasa de Deserción por {col}', fontweight='bold')
            ax.set_ylabel('Porcentaje (%)')
            ax.set_xlabel(col)
            ax.legend(['Retenido (No)', 'Desertor (Yes)'], title='Churn', frameon=True)
            ax.tick_params(axis='x', rotation=25)
            for p in ax.patches:
                width, height = p.get_width(), p.get_height()
                if height > 5:
                    x, y = p.get_xy() 
                    ax.text(x + width/2, y + height/2, f'{height:.1f}%', ha='center', va='center', color='white', fontweight='bold', fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'fig4_churn_categorical_distribution.png'), dpi=300)
        plt.close()

        # 5. Diagrama Bivariado de Dispersión tenure vs TotalCharges segmentado por Churn
        fig, ax = plt.subplots(figsize=(8, 5.5))
        sns.scatterplot(
            data=self.df, x='tenure', y='TotalCharges_cleaned', hue='Churn',
            palette={'No': '#2b83ba', 'Yes': '#d7191c'}, alpha=0.6, ax=ax, s=25
        )
        ax.set_title('Relación no lineal: Antigüedad (tenure) vs Cargos Totales (TotalCharges)', fontweight='bold')
        ax.set_xlabel('Antigüedad en Meses (tenure)')
        ax.set_ylabel('Cargos Totales Acumulados (USD)')
        ax.legend(title='Estado de Churn', frameon=True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'fig5_bivariate_scatter.png'), dpi=300)
        plt.close()

        # 6. Fuerza de Asociación V de Cramér para todas las variables categóricas
        df_chi = self.compute_bivariate_categorical()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=df_chi, x='Cramer_V', y='Variable', hue='Variable', palette='crest', legend=False, ax=ax)
        ax.axvline(0.15, color='orange', linestyle='--', label='Umbral Asociación Moderada (0.15)')
        ax.axvline(0.30, color='red', linestyle='--', label='Umbral Asociación Fuerte (0.30)')
        ax.set_title('Fuerza de Asociación con la Deserción (V de Cramér / χ²)', fontweight='bold')
        ax.set_xlabel('Coeficiente V de Cramér')
        ax.legend(frameon=True, loc='lower right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'fig6_cramer_v_association.png'), dpi=300)
        plt.close()
        
        print(f"Figuras generadas exitosamente en: {output_dir}")
