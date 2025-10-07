# ml_pipeline.py

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier


class DataLoader:
    def __init__(self, file_path=None):
        self.file_path = file_path
        self.df = None

    def load_csv(self, file_path=None):
        if file_path is not None:
            self.file_path = file_path
        if self.file_path is None:
            raise FileNotFoundError("No se especificó ruta del archivo CSV.")
        self.df = pd.read_csv(self.file_path)
        print(f"Archivo '{self.file_path}' cargado correctamente.")
        return self.df


class DataExplorer:
    def __init__(self, df):
        self.df = df

    def describe_data(self):
        print("Estadísticas descriptivas:")
        print(self.df.describe())
        print("\nValores únicos por columna:")
        print(self.df.nunique())
        print("\nValores faltantes por columna:")
        print(self.df.isnull().sum())

    def plot_class_distribution(self):
        sns.countplot(x='Class', data=self.df)
        plt.title("Distribución de clases")
        plt.savefig("reports/figures/plot_class_distribution.png")

    def plot_histograms(self):
        self.df.hist(bins=30, sharey=True, figsize=(16, 12))
        plt.tight_layout(pad=3.0)
        plt.savefig("reports/figures/plot_histograms.png")

    def cross_tabs(self):
        for column in self.df.select_dtypes(include=['object', 'bool']).columns:
            print(f"\nCrosstab para columna: {column}")
            print(pd.crosstab(index=self.df[column], columns='% observaciones', normalize='columns') * 100)


class Preprocessor:
    def __init__(self, df):
        self.df = df.copy()
        self.label_encoder = LabelEncoder()
        self.numeric_cols = None

    def encode_labels(self):
        self.df['Class'] = self.label_encoder.fit_transform(self.df['Class'])

    def drop_duplicates(self):
        dup_count = self.df.duplicated().sum()
        print(f"Duplicados encontrados: {dup_count}")
        if dup_count > 0:
            self.df = self.df.drop_duplicates()

    def drop_constant_columns(self):
        stds = self.df.select_dtypes(include=[np.number]).std()
        zero_std = stds[stds == 0].index.tolist()
        print(f"Columnas constantes eliminadas: {zero_std}")
        if zero_std:
            self.df = self.df.drop(columns=zero_std)

    def detect_outliers(self):
        self.numeric_cols = self.df.select_dtypes(include=[np.number]).columns.drop('Class', errors='ignore')
        z_scores = np.abs(stats.zscore(self.df[self.numeric_cols], nan_policy='omit'))
        outlier_counts = (z_scores > 3).sum(axis=0)
        
        # Convertimos el arreglo numpy a pd.Series para poder usar sort_values
        outlier_counts = pd.Series(outlier_counts, index=self.numeric_cols)
        
        print("Conteo de outliers por columna (z-score > 3):")
        print(outlier_counts.sort_values(ascending=False).head(10))

    def iqr_outliers(self):
        # Detect outliers using the IQR method for each numeric column except 'Class'
        self.numeric_cols = self.df.select_dtypes(include=[np.number]).columns.drop('Class', errors='ignore')
        for col in self.numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((self.df[col] < (Q1 - 1.5 * IQR)) | (self.df[col] > (Q3 + 1.5 * IQR))).sum()
            print(f"Columna '{col}': {outliers} outliers detectados (IQR)")


    def get_clean_df(self):
        return self.df


class Visualizer:
    def __init__(self, df):
        self.df = df

    def plot_correlations(self):
        numeric_df = self.df.select_dtypes(include=[np.number])
        corr = numeric_df.corr()
        plt.figure(figsize=(18, 12))
        sns.heatmap(corr, annot=True, linewidths=0.5)
        plt.title("Mapa de calor de correlaciones")
        plt.savefig("reports/figures/plot_correlations.png")

    def plot_boxplots(self):
        if 'Class' not in self.df.columns:
            print("La columna 'Class' no existe.")
            return

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.drop('Class', errors='ignore')
        for col in numeric_cols[:5]:
            plt.figure(figsize=(6, 3))
            sns.boxplot(x='Class', y=col, data=self.df)
            plt.title(f"Boxplot de {col} por clase")
            plt.tight_layout()
            plt.savefig(f"reports/figures/plot_boxplots{col}.png")

    def plot_densities(self):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.drop('Class', errors='ignore')
        for col in numeric_cols[:5]:
            plt.figure(figsize=(5, 3))
            sns.kdeplot(data=self.df, x=col, hue="Class", fill=True)
            plt.title(f"Densidad de {col}")
            plt.tight_layout()
            plt.savefig(f"reports/figures/plot_densities{col}.png")


class ModelTrainer:
    def __init__(self, df):
        self.df = df
        self.X = self.df.drop(columns='Class')
        self.Y = self.df['Class']
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None
        self.scaler = StandardScaler()

    def split_and_scale(self, test_size=0.3, random_state=42):
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            self.X, self.Y, test_size=test_size, random_state=random_state)
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

    def train_logistic(self):
        model = LogisticRegression(random_state=0)
        model.fit(self.X_train, self.Y_train)
        y_pred = model.predict(self.X_test)
        print("Logistic Regression Accuracy:", accuracy_score(self.Y_test, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(self.Y_test, y_pred))

    def train_knn(self, metric='minkowski'):
        model = KNeighborsClassifier(metric=metric)
        model.fit(self.X_train, self.Y_train)
        y_pred = model.predict(self.X_test)
        print(f"KNN ({metric}) Accuracy:", accuracy_score(self.Y_test, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(self.Y_test, y_pred))

    def get_split_data(self):
        return self.X_train, self.X_test, self.Y_train, self.Y_test


def save_datasets(raw_df, cleaned_df, X_train, X_test, Y_train, Y_test):
    import os
    os.makedirs("data", exist_ok=True)

    raw_df.to_csv("data/turkish_music_emotion_raw.csv", index=False)
    cleaned_df.to_csv("data/interim/turkish_music_emotion_cleaned.csv", index=False)

    pd.DataFrame(X_train).to_csv("data/interim/X_train.csv", index=False)
    pd.DataFrame(X_test).to_csv("data/processed//X_test.csv", index=False)
    pd.DataFrame(Y_train).to_csv("data/interim/Y_train.csv", index=False)
    pd.DataFrame(Y_test).to_csv("data/processed/Y_test.csv", index=False)
    print("Datasets guardados en la carpeta 'data/' para DVC.")


if __name__ == "__main__":
 
    loader = DataLoader("data/raw/acoustic_features.csv")
    df = loader.load_csv()

    # Exploración
    explorer = DataExplorer(df)
    explorer.describe_data()
    explorer.plot_class_distribution()
    explorer.plot_histograms()
    explorer.cross_tabs()

    # Preprocesamiento
    prep = Preprocessor(df)
    prep.encode_labels()
    prep.drop_duplicates()
    prep.drop_constant_columns()
    prep.detect_outliers()
    prep.iqr_outliers()
    df_clean = prep.get_clean_df()

    # Visualización
    viz = Visualizer(df_clean)
    viz.plot_correlations()
    viz.plot_boxplots()
    viz.plot_densities()

    # Entrenamiento
    trainer = ModelTrainer(df_clean)
    trainer.split_and_scale()
    trainer.train_logistic()
    trainer.train_knn()
    trainer.train_knn(metric="manhattan")

    # Guardado para DVC
    X_train, X_test, Y_train, Y_test = trainer.get_split_data()
    save_datasets(df, df_clean, X_train, X_test, Y_train, Y_test)