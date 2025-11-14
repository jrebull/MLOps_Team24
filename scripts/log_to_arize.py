# --- INICIO DEL SCRIPT ---
import time
import uuid
import warnings
from datetime import datetime, timedelta

import dvc.api
import joblib
import numpy as np
import pandas as pd

# ----------------- ¡¡¡ESTA ES LA IMPORTACIÓN DE LA API PANDAS!!! -----------------
from arize.api import ModelTypes, Environments 
from arize.pandas.logger import Client, Schema # ¡Volvemos a Client y Schema de pandas.logger!
# -------------------------------------------------------------------------------------

print("🚀 Iniciando script de monitoreo Arize (v5.0 - Prueba Aislada)...")
warnings.filterwarnings("ignore")

# --- 1. CONFIGURACIÓN ---
SPACE_ID = "U3BhY2U6MzIyODU6QkU4Mg==" 
API_KEY = "ak-ef3560c9-cb86-4ae1-af85-3d79e926f4d5-YBos3QR9GXY6-avKbyMitrFv2o5cRw55"


# --- 2. CONSTANTES DEL REPOSITORIO ---
REPO_URL = "https://github.com/jrebull/MLOps_Team24.git"
PATH_X_TRAIN = "data/processed/X_train.csv"
PATH_Y_TRAIN = "data/processed/y_train.csv"
PATH_X_TEST = "data/processed/X_test.csv"
PATH_Y_TEST = "data/processed/y_test.csv"
PATH_MODEL = "models/optimized/production_model.pkl"

EMOTION_MAP = {
    0: "Angry 😠",
    1: "Happy 😊",
    2: "Relax 😎",
    3: "Sad 😥"
}

PREDICTION_COLUMN = "prediction"
ACTUAL_COLUMN = "Class"
TIMESTAMP_COLUMN = "prediction_ts"
PREDICTION_ID_COLUMN = "prediction_id"

MODEL_ID = "turkish-music-emotion" 
MODEL_VERSION = "5.0-prueba-aislada" # <-- ¡Versión 5.0!


# --- 3. CONECTAR A ARIZE ---
try:
    arize_client = Client(space_id=SPACE_ID, api_key=API_KEY)
    print(f"✅ Conexión con Arize exitosa. Iniciando demo para {MODEL_ID} (v{MODEL_VERSION})")
except Exception as e:
    print(f"❌ Error conectando a Arize: {e}")
    exit()

# --- 4. CARGAR DATOS Y MODELO ---
print("⬇️ Cargando datos desde DVC...")
try:
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
    print("✅ Datos y modelo cargados.")
except Exception as e:
    print(f"❌ Error cargando datos o modelo: {e}")
    exit()

# ¡¡¡OJO!!! NO creamos FEATURE_NAMES de 50 columnas aquí.

if isinstance(y_train, pd.DataFrame) and ACTUAL_COLUMN in y_train.columns:
    y_train = y_train[ACTUAL_COLUMN]
if isinstance(y_test, pd.DataFrame) and ACTUAL_COLUMN in y_test.columns:
    y_test = y_test[ACTUAL_COLUMN]

# --- 5. SELECCIÓN DINÁMICA DE FEATURES PARA DRIFT ---
# (Tu lógica original para seleccionar features)
todas_las_features = list(X_test.columns)
features_con_datos = []
for col in todas_las_features:
    if col in X_train.columns and col in X_test.columns:
        if X_train[col].nunique() > 1 and X_test[col].nunique() > 1 and \
           X_train[col].isnull().sum() == 0 and X_test[col].isnull().sum() == 0:
            features_con_datos.append(col)

features_para_drift = features_con_datos[:5] if len(features_con_datos) >= 5 else features_con_datos

# --- ¡¡¡CAMBIO CLAVE 1: LA PRUEBA DE AISLAMIENTO!!! ---
# En lugar de las 50, solo usaremos la lista de 5 features que SÍ usamos.
FEATURE_NAMES = features_para_drift
print(f"✅ PRUEBA: Solo se enviarán estas {len(FEATURE_NAMES)} features a Arize: {FEATURE_NAMES}")
# ------------------------------------------------------

# --- 6. PREPARAR Y ENVIAR EL TRAINING (LÍNEA BASE) ---
print("\n🧠 Generando predicciones de Training...")
preds_train = model.predict(X_train[todas_las_features]) # El modelo necesita las 50
now = datetime.now()
training_timestamp = int((now - timedelta(days=30)).timestamp())

df_train = X_train.copy()
df_train[PREDICTION_ID_COLUMN] = [str(uuid.uuid4()) for _ in range(len(df_train))]
df_train[TIMESTAMP_COLUMN] = training_timestamp
df_train[PREDICTION_COLUMN] = pd.Series(preds_train).map(EMOTION_MAP)
df_train[ACTUAL_COLUMN] = y_train.map(EMOTION_MAP)

# --- ¡¡¡CAMBIO CLAVE 2: VOLVEMOS AL ARGUMENTO CORRECTO!!! ---
schema_training = Schema(
    prediction_id_column_name=PREDICTION_ID_COLUMN,
    timestamp_column_name=TIMESTAMP_COLUMN,
    feature_column_names=FEATURE_NAMES, # <--- ¡¡LA CORRECCIÓN!! (usando la lista de 5)
    prediction_label_column_name=PREDICTION_COLUMN,
    actual_label_column_name=ACTUAL_COLUMN
)

print(f"📤 Enviando TRAINING (v{MODEL_VERSION}) como línea base...")
# Usamos el método .log() y el argumento 'dataframe='
res = arize_client.log(
    dataframe=df_train, # <--- ¡¡EL ARGUMENTO ORIGINAL!!
    schema=schema_training,
    model_id=MODEL_ID,
    model_version=MODEL_VERSION,
    model_type=ModelTypes.SCORE_CATEGORICAL,
    environment=Environments.TRAINING
)
if res.status_code != 200: print(f"❌ Error Training: {res.content}")
else: print("✅ Training enviado.")


# --- 7. SIMULACIÓN DE DRIFT GRADUAL (30 DÍAS AL PASADO) ---

# --- ¡¡¡CAMBIO CLAVE 2 (Repetido)!!! ---
schema_produccion = Schema(
    prediction_id_column_name=PREDICTION_ID_COLUMN,
    timestamp_column_name=TIMESTAMP_COLUMN,
    feature_column_names=FEATURE_NAMES, # <--- ¡¡LA CORRECCIÓN!! (usando la lista de 5)
    prediction_label_column_name=PREDICTION_COLUMN
)
schema_actuals = Schema(
    prediction_id_column_name=PREDICTION_ID_COLUMN,
    actual_label_column_name=ACTUAL_COLUMN,
    timestamp_column_name=TIMESTAMP_COLUMN
)

print(f"\n🚀 Iniciando Simulación de Drift Gradual (30 días) para v{MODEL_VERSION}...")

for i in range(1, 31):
    current_time = now - timedelta(days=i)
    current_timestamp = int(current_time.timestamp())
    print(f"\n--- Día {i}/30 ({current_time.strftime('%Y-%m-%d')}) ---")
    
    df_prod = X_test.copy()
    print("🔥 Simulando drift gradual...")
    # 'features_para_drift' ya es la misma lista que 'FEATURE_NAMES'
    for idx, feature in enumerate(features_para_drift):
        if idx % 5 == 0: df_prod[feature] = df_prod[feature] * (1 + i * 0.05)
        elif idx % 5 == 1: df_prod[feature] = df_prod[feature] + (i * 0.02)
        elif idx % 5 == 2: df_prod[feature] = df_prod[feature] * (1 - i * 0.03)
        elif idx % 5 == 3: df_prod[feature] = df_prod[feature] - (i * 0.01)
        else: df_prod[feature] = df_prod[feature] + (i * 0.015)

    print("🧠 Generando predicciones de producción...")
    preds_prod = model.predict(df_prod[todas_las_features]) # El modelo necesita las 50
    df_prod[PREDICTION_ID_COLUMN] = [str(uuid.uuid4()) for _ in range(len(df_prod))]
    df_prod[TIMESTAMP_COLUMN] = current_timestamp
    df_prod[PREDICTION_COLUMN] = pd.Series(preds_prod).map(EMOTION_MAP)
    
    print("📤 Enviando PREDICCIONES a Arize...")
    res_pred = arize_client.log(
        dataframe=df_prod, # <--- ¡¡EL ARGUMENTO ORIGINAL!!
        schema=schema_produccion,
        model_id=MODEL_ID,
        model_version=MODEL_VERSION,
        model_type=ModelTypes.SCORE_CATEGORICAL,
        environment=Environments.PRODUCTION
    )
    if res_pred.status_code != 200: print(f"❌ Error Predicciones Día {i}: {res_pred.content}")
    else: print("✅ Predicciones enviadas.")

    print("📉 Simulando llegada de 'Actuals' (con errores)...")
    tasa_de_error = min(0.05 + (i * 0.03), 0.95)
    print(f"-> Tasa de error para hoy: {tasa_de_error:.0%}")
    df_actuals_prod = y_test.copy()
    num_errores = int(len(df_actuals_prod) * tasa_de_error)
    indices_error = np.random.choice(df_actuals_prod.index, num_errores, replace=False)
    etiquetas_reales_con_error = df_actuals_prod.copy()
    etiquetas_reales_con_error.loc[indices_error] = (etiquetas_reales_con_error.loc[indices_error] + 1) % 4
    
    df_actuals_final = pd.DataFrame({
        PREDICTION_ID_COLUMN: df_prod[PREDICTION_ID_COLUMN], 
        ACTUAL_COLUMN: etiquetas_reales_con_error, 
        TIMESTAMP_COLUMN: current_timestamp
    })
    
    df_actuals_final[ACTUAL_COLUMN] = df_actuals_final[ACTUAL_COLUMN].map(EMOTION_MAP)
    
    print("📤 Enviando ACTUALS a Arize para unir...")
    res_actual = arize_client.log(
        dataframe=df_actuals_final, # <--- ¡¡EL ARGUMENTO ORIGINAL!!
        schema=schema_actuals,
        model_id=MODEL_ID,
        model_version=MODEL_VERSION,
        model_type=ModelTypes.SCORE_CATEGORICAL,
        environment=Environments.PRODUCTION
    )
    if res_actual.status_code != 200: print(f"❌ Error Actuals Día {i}: {res_actual.content}")
    else: print("✅ Actuals enviados.")

print(f"\n🎉 ¡DEMO COMPLETA! Ve a Arize y busca la versión {MODEL_VERSION}")
print("🔍 En el dashboard, revisa si ahora sí puedes ver las 5 features ('_RMSenergy_Mean', etc.) con datos.")