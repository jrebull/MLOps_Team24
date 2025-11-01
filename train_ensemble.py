import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.metrics import classification_report, accuracy_score

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("turkish-music-emotion-recognition")

# Usar splits ya limpios
X_train = pd.read_csv('data/processed/X_train.csv')
X_test = pd.read_csv('data/processed/X_test.csv')
y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()

print(f"📊 Train: {X_train.shape} | Test: {X_test.shape}")
print(f"   Clases: {pd.Series(y_train).unique()}")

with mlflow.start_run(run_name="ensemble_voting_best"):
    # Ensemble con mejores hiperparámetros
    rf = RandomForestClassifier(
        n_estimators=300, 
        max_depth=25, 
        min_samples_split=3,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42, 
        n_jobs=-1
    )
    
    gb = GradientBoostingClassifier(
        n_estimators=200, 
        max_depth=8, 
        learning_rate=0.05,
        subsample=0.8,
        random_state=42
    )
    
    et = ExtraTreesClassifier(
        n_estimators=300, 
        max_depth=25, 
        min_samples_split=3,
        random_state=42, 
        n_jobs=-1
    )
    
    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('gb', gb), ('et', et)],
        voting='soft',
        weights=[2, 1, 2],  # RF y ET más peso
        n_jobs=-1
    )
    
    print("  → Entrenando ensemble optimizado...")
    ensemble.fit(X_train, y_train)
    
    y_pred = ensemble.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\n🎯 Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print("\n" + classification_report(y_test, y_pred))
    
    mlflow.log_params({
        "model_type": "voting_ensemble",
        "voting": "soft",
        "weights": "2,1,2",
        "rf_n_estimators": 300,
        "gb_n_estimators": 200,
        "et_n_estimators": 300
    })
    
    mlflow.log_metric("test_accuracy", acc)
    
    report = classification_report(y_test, y_pred, output_dict=True)
    for emotion in pd.Series(y_train).unique():
        if emotion in report:
            mlflow.log_metric(f"{emotion}_precision", report[emotion]['precision'])
            mlflow.log_metric(f"{emotion}_recall", report[emotion]['recall'])
            mlflow.log_metric(f"{emotion}_f1", report[emotion]['f1-score'])
    
    mlflow.sklearn.log_model(ensemble, "model", input_example=X_train[:5])
    
    run_id = mlflow.active_run().info.run_id
    print(f"\n✅ RUN_ID: {run_id}")
    
print("\n🚀 LISTO!")
