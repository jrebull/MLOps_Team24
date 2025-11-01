import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("turkish-music-emotion-recognition")

# Cargar datos ORIGINALES (sin normalizar)
df = pd.read_csv('data/processed/turkish_music_emotion_v1_original.csv')
X = df.drop(['Class'], axis=1)
y = df['Class'].str.strip().str.lower()

# Remover NaN
mask = ~y.isna()
X = X[mask]
y = y[mask]

print(f"📊 Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"   Clases: {y.unique()}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\n🚀 Entrenando ENSEMBLE (RAW features)...")

with mlflow.start_run(run_name="ensemble_raw_features"):
    rf = RandomForestClassifier(n_estimators=300, max_depth=25, min_samples_split=3, random_state=42, n_jobs=-1)
    gb = GradientBoostingClassifier(n_estimators=200, max_depth=8, learning_rate=0.05, random_state=42)
    et = ExtraTreesClassifier(n_estimators=300, max_depth=25, min_samples_split=3, random_state=42, n_jobs=-1)
    
    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('gb', gb), ('et', et)],
        voting='soft',
        weights=[2, 1, 2],
        n_jobs=-1
    )
    
    print("  → Training...")
    ensemble.fit(X_train, y_train)
    
    y_pred = ensemble.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\n✅ Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print("\n" + classification_report(y_test, y_pred))
    
    mlflow.log_params({
        "model_type": "voting_ensemble",
        "features": "raw_unnormalized",
        "n_estimators": "300,200,300"
    })
    
    mlflow.log_metric("test_accuracy", acc)
    
    report = classification_report(y_test, y_pred, output_dict=True)
    for emotion in y.unique():
        if emotion in report:
            mlflow.log_metric(f"{emotion}_recall", report[emotion]['recall'])
    
    mlflow.sklearn.log_model(ensemble, "model", input_example=X_train[:5])
    
    run_id = mlflow.active_run().info.run_id
    print(f"\n🎯 NEW RUN_ID: {run_id}")
    
print("\n✅ DONE!")
