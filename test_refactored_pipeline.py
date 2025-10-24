"""Quick test of refactored modeling pipeline (without MLflow)"""
import sys
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from acoustic_ml.dataset import DatasetManager
from acoustic_ml.modeling.train import BaseModelTrainer, ModelConfig, TrainingConfig
from acoustic_ml.modeling.predict import ModelPredictor
from acoustic_ml.modeling.evaluate import ModelEvaluator
from acoustic_ml.config import MODELS_DIR
from sklearn.ensemble import RandomForestClassifier

def test_pipeline():
    """Test the complete refactored pipeline"""
    
    print("🧪 Testing Refactored Pipeline (MLflow disabled)\n")
    print("=" * 50)
    
    # 1. Load data
    print("\n📊 Step 1: Loading processed data...")
    try:
        manager = DatasetManager()
        X_train, X_test, y_train, y_test = manager.load_train_test_split()
        print(f"✅ Data loaded successfully")
        print(f"   - Train: {X_train.shape}, Test: {X_test.shape}")
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return False
    
    # 2. Train model WITHOUT MLflow
    print("\n🎯 Step 2: Training model (MLflow disabled)...")
    try:
        # Create config without MLflow
        model_config = ModelConfig(
            name="RandomForest_Baseline_Test",
            model_class=RandomForestClassifier,
            hyperparameters={
                "n_estimators": 200,
                "max_depth": None,
                "random_state": 42,
                "n_jobs": -1
            }
        )
        training_config = TrainingConfig(
            mlflow_tracking_uri=None  # Disable MLflow
        )
        
        trainer = BaseModelTrainer(model_config, training_config)
        model = trainer.train(X_train, y_train)
        print(f"✅ Model trained successfully")
        
        # Get metrics from trainer
        metrics = trainer.evaluate(X_train, y_train)
        print(f"   - CV Accuracy (mean): {metrics['accuracy_mean']:.4f}")
        print(f"   - CV Accuracy (std): {metrics['accuracy_std']:.4f}")
        
        # Save model for testing
        test_model_path = MODELS_DIR / "test_model.pkl"
        with open(test_model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"   - Model saved to {test_model_path}")
        
    except Exception as e:
        print(f"❌ Failed to train model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 3. Test predictor
    print("\n🔮 Step 3: Testing predictor...")
    try:
        predictor = ModelPredictor("test_model.pkl")
        predictor.load_model()
        
        predictions = predictor.predict(X_test[:10])
        print(f"✅ Batch prediction successful")
        print(f"   - Predicted {len(predictions)} samples")
        
        single_pred = predictor.predict_single(X_test.iloc[0])
        print(f"✅ Single prediction successful: {single_pred}")
        
    except Exception as e:
        print(f"❌ Failed prediction: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 4. Test evaluator
    print("\n📈 Step 4: Testing evaluator...")
    try:
        evaluator = ModelEvaluator("test_model.pkl")
        
        metrics = evaluator.evaluate(X_test, y_test)
        print(f"✅ Evaluation successful")
        print(f"   - Accuracy: {metrics['accuracy']:.4f}")
        print(f"   - F1 (macro): {metrics['f1_macro']:.4f}")
        
        report = evaluator.generate_classification_report(X_test, y_test, save=False)
        print(f"✅ Classification report generated")
        
        fig = evaluator.plot_confusion_matrix(X_test, y_test, save=False)
        print(f"✅ Confusion matrix plotted")
        
    except Exception as e:
        print(f"❌ Failed evaluation: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 50)
    print("🎉 All tests passed successfully!")
    print("=" * 50)
    return True

if __name__ == "__main__":
    success = test_pipeline()
    sys.exit(0 if success else 1)
