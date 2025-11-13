import dvc.api
import pandas as pd
import joblib
import time
from arize.pandas.logger import Client, Schema
from arize.utils.types import ModelTypes, Environments
import warnings
# --- ADICIONES: Importa estas librerías al inicio de tu script ---
import uuid
import numpy as np
from datetime import datetime, timedelta

# --- 1. CONFIGURACIÓN ---
print("🚀 Iniciando script de monitoreo Arize (Versión con Timestamp)...")
warnings.filterwarnings("ignore")

# !!! PEGA TUS LLAVES AQUI ABAJO DESPUES DE CREAR EL ARCHIVO !!!
SPACE_ID = "U3BhY2U6MzIyODU6QkU4Mg==" 
API_KEY = "ak-ef3560c9-cb86-4ae1-af85-3d79e926f4d5-YBos3QR9GXY6-avKbyMitrFv2o5cRw55"
MODEL_ID = "turkish-music-emotion"
MODEL_VERSION = "1.4-con-timestamp" # <--- Versión nueva para empezar limpio


# --- 2. CONSTANTES DEL REPOSITORIO ---
REPO_URL = "https://github.com/jrebull/MLOps_Team24.git"
PATH_X_TRAIN = "data/processed/X_train.csv"
PATH_Y_TRAIN = "data/processed/y_train.csv"
PATH_X_TEST = "data/processed/X_test.csv"
PATH_Y_TEST = "data/processed/y_test.csv"
PATH_MODEL = "models/optimized/production_model.pkl"

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

# --- NUEVAS CONSTANTES PARA LA DEMO ---
PREDICTION_COLUMN = "prediction"
ACTUAL_COLUMN = "Class"
TIMESTAMP_COLUMN = "prediction_ts"
PREDICTION_ID_COLUMN = "prediction_id" # <-- CLAVE 1

# --- 3. CONECTAR ---
try:
    arize_client = Client(space_id=SPACE_ID, api_key=API_KEY)
    print(f"✅ Conexión con Arize exitosa. Iniciando demo para {MODEL_ID} (v{MODEL_VERSION})")
except Exception as e:
    print(f"❌ Error conectando a Arize: {e}")
    exit()

# --- 4. CARGAR DATOS ---
print("⬇️ Cargando datos desde DVC...")
with dvc.api.open(path=PATH_X_TRAIN, repo=REPO_URL, mode='r') as f:
    X_train = pd.read_csv(f)
with dvc.api.open(path=PATH_Y_TRAIN, repo=REPO_URL, mode='r') as f:
    y_train = pd.read_csv(f)
with dvc.api.open(path=PATH_X_TEST, repo=REPO_URL, mode='r') as f:
    X_test = pd.read_csv(f)
with dvc.api.open(path=PATH_Y_TEST, repo=REPO_URL, mode='r') as f:
    y_test = pd.read_csv(f)
with dvc.api.open(path=PATH_MODEL, repo=REPO_URL, mode='rb') as f:
    model = joblib.load(f)

# --- 5. PREPARAR Y ENVIAR EL TRAINING (LÍNEA BASE) ---
print("🧠 Generando predicciones de Training...")
preds_train = model.predict(X_train)
now = datetime.now()
training_timestamp = int((now - timedelta(days=30)).timestamp()) # Simula que el training es de hace 30 días

df_train = X_train.copy()
df_train[PREDICTION_ID_COLUMN] = [str(uuid.uuid4()) for _ in range(len(df_train))]
df_train[TIMESTAMP_COLUMN] = training_timestamp
df_train[PREDICTION_COLUMN] = preds_train
df_train[ACTUAL_COLUMN] = y_train[ACTUAL_COLUMN]

# Define el Schema de Training (tiene todo)
schema_training = Schema(
    prediction_id_column_name=PREDICTION_ID_COLUMN,
    timestamp_column_name=TIMESTAMP_COLUMN,
    feature_column_names=FEATURE_NAMES,
    prediction_label_column_name=PREDICTION_COLUMN,
    actual_label_column_name=ACTUAL_COLUMN
)

print(f"📤 Enviando TRAINING (v{MODEL_VERSION}) como línea base...")
res = arize_client.log(
    dataframe=df_train,
    model_id=MODEL_ID,
    model_version=MODEL_VERSION,
    model_type=ModelTypes.SCORE_CATEGORICAL,
    environment=Environments.TRAINING,
    schema=schema_training
)
if res.status_code != 200: print(f"❌ Error Training: {res.content}")
else: print("✅ Training enviado.")


# --- 6. SIMULACIÓN DE DRIFT GRADUAL (10 DÍAS) ---
# Definimos schemas separados para producción (realista)
schema_produccion = Schema(
    prediction_id_column_name=PREDICTION_ID_COLUMN,
    timestamp_column_name=TIMESTAMP_COLUMN,
    feature_column_names=FEATURE_NAMES,
    prediction_label_column_name=PREDICTION_COLUMN
)
schema_actuals = Schema(
    prediction_id_column_name=PREDICTION_ID_COLUMN,
    actual_label_column_name=ACTUAL_COLUMN
)

print("\n🚀 Iniciando Simulación de Drift Gradual y Caída de Performance (10 días)...")

for i in range(1, 31): # Simula 10 días, del 1 al 10
    current_time = now - timedelta(days=i)
    current_timestamp = int(current_time.timestamp())
    print(f"\n--- Día {i}/10 ({current_time.strftime('%Y-%m-%d')}) ---")
    
    # 1. Copia los datos de test (producción)
    df_prod = X_test.copy()
    
    # 2. 🔥 SIMULACIÓN DE DRIFT "IMPRESIONANTE" 🔥
    print("🔥 Simulando drift gradual...")
    # El drift se hace más fuerte cada día (i)
    df_prod['_Tempo_Mean'] = df_prod['_Tempo_Mean'] * (1 + i * 0.2) # Aumenta 20% por día
    df_prod['_MFCC_Mean_1'] = df_prod['_MFCC_Mean_1'] + (i * 1.5) # Suma 1.5 cada día
    df_prod['_Fluctuation_Mean'] = df_prod['_Fluctuation_Mean'] * (1 + i * 0.1) # Aumenta 10% por día
    df_prod['_Spectralcentroid_Mean'] = df_prod['_Spectralcentroid_Mean'] + (i * 5) # Suma 5 cada día
    df_prod['_RMSenergy_Mean'] = df_prod['_RMSenergy_Mean'] - (i * 0.05) # Resta 0.05 cada día

    # 3. Generar predicciones (con los datos sucios)
    print("🧠 Generando predicciones de producción...")
    preds_prod = model.predict(df_prod[FEATURE_NAMES])
    
    df_prod[PREDICTION_ID_COLUMN] = [str(uuid.uuid4()) for _ in range(len(df_prod))]
    df_prod[TIMESTAMP_COLUMN] = current_timestamp
    df_prod[PREDICTION_COLUMN] = preds_prod
    
    # 4. Enviar PREDICCIONES (aún no sabemos la respuesta real)
    print("📤 Enviando PREDICCIONES a Arize...")
    res_pred = arize_client.log(
        dataframe=df_prod,
        model_id=MODEL_ID,
        model_version=MODEL_VERSION,
        model_type=ModelTypes.SCORE_CATEGORICAL,
        environment=Environments.PRODUCTION,
        schema=schema_produccion
    )
    if res_pred.status_code != 200: print(f"❌ Error Predicciones Día {i}: {res_pred.content}")
    else: print("✅ Predicciones enviadas.")

    # 5. 📉 SIMULACIÓN DE CAÍDA DE PERFORMANCE 📉
    print("📉 Simulando llegada de 'Actuals' (con errores)...")
    # A medida que el drift (i) aumenta, la tasa de error también
    tasa_de_error = min(0.05 + (i * 0.08), 1.0) # Empieza en 5% y sube 8% por día
    print(f"-> Tasa de error para hoy: {tasa_de_error:.0%}")
    
    df_actuals_prod = y_test.copy()
    
    # Introducir errores
    num_errores = int(len(df_actuals_prod) * tasa_de_error)
    indices_error = np.random.choice(df_actuals_prod.index, num_errores, replace=False)
    
    # Cambia la etiqueta real por una incorrecta
    # (Esto simula que nuestro modelo predijo mal Y el usuario nos da la etiqueta real)
    # Lógica simple: cambia la etiqueta a (etiqueta % 8) + 1
    etiquetas_reales_con_error = df_actuals_prod[ACTUAL_COLUMN].copy()
    etiquetas_reales_con_error.loc[indices_error] = (etiquetas_reales_con_error.loc[indices_error] % 8) + 1
    
    # 6. Preparar y enviar los ACTUALS (las respuestas reales)
    df_actuals_final = pd.DataFrame({
        PREDICTION_ID_COLUMN: df_prod[PREDICTION_ID_COLUMN], # <-- CLAVE 2: Une con la predicción
        ACTUAL_COLUMN: etiquetas_reales_con_error
    })
    
    print("📤 Enviando ACTUALS a Arize para unir...")
    res_actual = arize_client.log(
        dataframe=df_actuals_final,
        model_id=MODEL_ID,
        model_version=MODEL_VERSION,
        environment=Environments.PRODUCTION,
        schema=schema_actuals,
        model_type=ModelTypes.SCORE_CATEGORICAL # <--- AGREGADO
    )
    if res_actual.status_code != 200: print(f"❌ Error Actuals Día {i}: {res_actual.content}")
    else: print("✅ Actuals enviados.")


print(f"\n🎉 ¡DEMO COMPLETA! Ve a Arize y busca la versión {MODEL_VERSION}")
print("🔍 En el dashboard, ajusta el tiempo a 'Last 24 hours' y observa:")
print("  1. Pestaña 'Drift': Verás el PSI subiendo día a día.")
print("  2. Pestaña 'Performance': Verás la 'Accuracy' bajando día a día.")
print("  3. Haz clic en un día con drift y verás el 'Drift Breakdown' LLENO de datos.")