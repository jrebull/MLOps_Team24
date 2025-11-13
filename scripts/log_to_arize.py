import dvc.api
import pandas as pd
import joblib
from arize.api import Client
import arize.log
from arize.utils.types import ModelTypes, Environments
from arize.pandas.logger import Schema
import warnings

# --- 1. CONFIGURACIÓN (¡Pega tus Keys aquí!) ---
print("🚀 Iniciando script de monitoreo Arize...")
warnings.filterwarnings("ignore")

SPACE_ID = "TurkishArize"
API_KEY = "ak-ef3560c9-cb86-4ae1-af85-3d79e926f4d5-YBos3QR9GXY6-avKbyMitrFv2o5cRw55"
MODEL_ID = "turkish-music-emotion"
MODEL_VERSION = "1.0" # Puedes cambiar esto (ej. "2.0") cuando re-entrenes

# --- 2. CONSTANTES DEL REPOSITORIO (Detectadas automáticamente) ---
REPO_URL = "https://github.com/jrebull/MLOps_Team24.git"
PATH_X_TRAIN = "data/processed/X_train.csv"
PATH_Y_TRAIN = "data/processed/y_train.csv"
PATH_X_TEST = "data/processed/X_test.csv"
PATH_Y_TEST = "data/processed/y_test.csv"
PATH_MODEL = "models/optimized/production_model.pkl" # Usando tu modelo de prod

# Nombres de columnas detectados de tus archivos
FEATURE_NAMES = [
    '_RMSenergy_Mean', '_Lowenergy_Mean', '_Fluctuation_Mean', '_Tempo_Mean', 
    '_MFCC_Mean_1', '_MFCC_Mean_2', '_MFCC_Mean_3', '_MFCC_Mean_4', 
    '_MFCC_Mean_5', '_MFCC_Mean_6', '_MFCC_Mean_7', '_MFCC_Mean_8', 
    '_MFCC_Mean_9', '_MFCC_Mean_10', '_MFCC_Mean_11', '_MFCC_Mean_12', 
    '_MFCC_Mean_13', '_Roughness_Mean', '_Roughness_Slope', 
    '_Zero-crossingrate_Mean', '_AttackTime_Mean', '_AttackTime_Slope', 
    '_Rolloff_Mean', '_Eventdensity_Mean', '_Pulseclarity_Mean', 
    '_Brightness_Mean', '_Spectralcentroid_Mean', '_Spectralspread_Mean', 
    '_Spectralskewness_Mean', '_Spectralkurtosis_Mean', '_Spectralflatness_Mean', 
    '_EntropyofSpectrum_Mean', '_Chromagram_Mean_1', '_Chromagram_Mean_2', 
    '_Chromagram_Mean_3', '_Chromagram_Mean_4', '_Chromagram_Mean_5', 
    '_Chromagram_Mean_6', '_Chromagram_Mean_7', '_Chromagram_Mean_8', 
    '_Chromagram_Mean_9', '_Chromagram_Mean_10', '_Chromagram_Mean_11', 
    '_Chromagram_Mean_12', '_HarmonicChangeDetectionFunction_Mean', 
    '_HarmonicChangeDetectionFunction_Std', '_HarmonicChangeDetectionFunction_Slope', 
    '_HarmonicChangeDetectionFunction_PeriodFreq', 
    '_HarmonicChangeDetectionFunction_PeriodAmp', 
    '_HarmonicChangeDetectionFunction_PeriodEntropy'
]
PREDICTION_COLUMN = "prediction" # Crearemos esta columna
ACTUAL_COLUMN = "Class"          # Esta es tu etiqueta real (y_train/y_test)

# --- 3. CONECTAR Y CARGAR DATOS ---
try:
    arize_client = Client(space_id=SPACE_ID, api_key=API_KEY)
    print("✅ Conexión con Arize exitosa.")
except Exception as e:
    print(f"❌ Error conectando a Arize: {e}")
    exit()

print("⬇️ Cargando datos y modelo desde DVC/S3... (Esto puede tardar)")

# Cargar datasets
with dvc.api.open(path=PATH_X_TRAIN, repo=REPO_URL, mode='r') as f:
    X_train = pd.read_csv(f)
with dvc.api.open(path=PATH_Y_TRAIN, repo=REPO_URL, mode='r') as f:
    y_train = pd.read_csv(f)
with dvc.api.open(path=PATH_X_TEST, repo=REPO_URL, mode='r') as f:
    X_test = pd.read_csv(f)
with dvc.api.open(path=PATH_Y_TEST, repo=REPO_URL, mode='r') as f:
    y_test = pd.read_csv(f)

# Cargar modelo
with dvc.api.open(path=PATH_MODEL, repo=REPO_URL, mode='rb') as f:
    model = joblib.load(f)

print("✅ Datos y modelo cargados.")

# --- 4. GENERAR PREDICCIONES ---
print("🧠 Generando predicciones...")
preds_train = model.predict(X_train)
preds_test = model.predict(X_test)

# --- 5. PREPARAR DATAFRAMES PARA ARIZE ---
# Arize necesita (features, predicción, etiqueta_real) en un solo DataFrame

# DataFrame de Entrenamiento
df_train = X_train.copy()
df_train[PREDICTION_COLUMN] = preds_train
df_train[ACTUAL_COLUMN] = y_train[ACTUAL_COLUMN]

# DataFrame de Producción (Test)
df_prod = X_test.copy()
df_prod[PREDICTION_COLUMN] = preds_test
df_prod[ACTUAL_COLUMN] = y_test[ACTUAL_COLUMN]

print("✅ DataFrames listos para enviar.")

# --- 6. DEFINIR EL SCHEMA ---
schema = Schema(
    feature_column_names=FEATURE_NAMES,
    prediction_label_column_name=PREDICTION_COLUMN,
    actual_label_column_name=ACTUAL_COLUMN 
)

# --- 7. ENVIAR DATOS A ARIZE ---

# Loggear el BASELINE (Entrenamiento)
print(f"📤 Enviando datos de TRAINING (Baseline) para '{MODEL_ID}' v{MODEL_VERSION}...")
try:
    arize.log.log(client=arize_client, 
        dataframe=df_train,
        model_id=MODEL_ID,
        model_version=MODEL_VERSION,
        model_type=ModelTypes.SCORE_CATEGORICAL,
        environment=Environments.TRAINING,
        schema=schema
    )
    print("✅ Datos de TRAINING enviados.")
except Exception as e:
    print(f"❌ Error enviando datos de TRAINING: {e}")

# Loggear los datos de PRODUCCIÓN
print(f"📤 Enviando datos de PRODUCTION para '{MODEL_ID}' v{MODEL_VERSION}...")
try:
    arize.log.log(client=arize_client, 
        dataframe=df_prod,
        model_id=MODEL_ID,
        model_version=MODEL_VERSION,
        model_type=ModelTypes.SCORE_CATEGORICAL,
        environment=Environments.PRODUCTION,
        schema=schema
    )
    print("✅ Datos de PRODUCTION enviados.")
except Exception as e:
    print(f"❌ Error enviando datos de PRODUCTION: {e}")

print(f"\n🎉 ¡Proceso completo! Revisa tu modelo '{MODEL_ID}' en el dashboard de Arize en 5-10 minutos.")