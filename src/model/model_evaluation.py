import os
import json
import logging
from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, roc_auc_score, 
    confusion_matrix, f1_score, precision_recall_curve, roc_curve
)
import mlflow
import mlflow.tensorflow
import seaborn as sns
from src.model.mlflow_utils import MLFlowManager

logging.basicConfig(level=logging.INFO)

class ModelEvaluator:
    def __init__(self, model_path='models/rnn_model.h5', mlflow_config_path='mlflow_config.yaml'):
        self.model_path = model_path
        self.mlflow_manager = MLFlowManager(mlflow_config_path)
        
        # Ensure reports directory exists
        os.makedirs('reports/figures', exist_ok=True)

    @staticmethod
    def load_data(file_path):
        df = pd.read_csv(file_path)
        logging.info("Data loaded from %s", file_path)
        return df

    def evaluate_model(self, model, X_test, y_test):
        """
        Evaluate model performance with comprehensive metrics and visualizations.
        """
        # Get predictions
        y_pred = model.predict(X_test)
        y_pred_binary = (y_pred > 0.5).astype(int)
        
        # Calculate confusion matrix elements
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_binary).ravel()
        
        # Calculate comprehensive metrics
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred_binary)),
            'precision': float(precision_score(y_test, y_pred_binary)),
            'recall': float(recall_score(y_test, y_pred_binary)),
            'f1_score': float(f1_score(y_test, y_pred_binary)),
            'roc_auc': float(roc_auc_score(y_test, y_pred)),
            'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,  # True Negative Rate
            'false_positive_rate': float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
            'negative_predictive_value': float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0,
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn)
        }
        
        # Generate evaluation plots
        self._generate_evaluation_plots(y_test, y_pred, y_pred_binary)
        
        return metrics
        
    def _generate_evaluation_plots(self, y_test, y_pred, y_pred_prob):
        """
        Generate and save evaluation plots.
        """
        # Create reports directory if it doesn't exist
        os.makedirs('reports/figures', exist_ok=True)
        
        # Convert continuous predictions to binary for confusion matrix
        y_pred_binary = (y_pred_prob > 0.5).astype(int)
        
        # 1. Confusion Matrix Heatmap
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred_binary)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('reports/figures/confusion_matrix.png')
        plt.close()
        
        # 2. ROC Curve
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.savefig('reports/figures/roc_curve.png')
        plt.close()
        
        # 3. Precision-Recall Curve
        plt.figure(figsize=(8, 6))
        precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
        plt.plot(recall, precision)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.savefig('reports/figures/precision_recall_curve.png')
        plt.close()
        
        # 4. Prediction Distribution
        plt.figure(figsize=(10, 6))
        plt.hist(y_pred_prob[y_test == 0], bins=50, alpha=0.5, label='Actual Negative')
        plt.hist(y_pred_prob[y_test == 1], bins=50, alpha=0.5, label='Actual Positive')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Count')
        plt.title('Prediction Probability Distribution')
        plt.legend()
        plt.savefig('reports/figures/prediction_distribution.png')
        plt.close()

    def save_metrics(self, metrics, file_path='reports/metrics.json'):
        """Save metrics to a JSON file and log to MLFlow."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        logging.info("Metrics saved to %s", file_path)

    def load_model(self):
        """Load model from local path or MLFlow."""
        try:
            # First try loading from MLFlow if run_id is available
            with open('reports/experiment_info.json', 'r') as f:
                experiment_info = json.load(f)
                run_id = experiment_info.get('run_id')
                
            if run_id:
                model_uri = f"runs:/{run_id}/model"
                try:
                    model = mlflow.tensorflow.load_model(model_uri)
                    logging.info(f"Model loaded from MLFlow run: {run_id}")
                    return model
                except Exception as e:
                    logging.warning(f"Could not load model from MLFlow: {e}")
            
            # Fall back to local path
            model = tf.keras.models.load_model(self.model_path)
            logging.info("Model loaded from %s", self.model_path)
            return model
            
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise

    def full_evaluation_pipeline(self, test_file='./datas/processed/test_features.csv'):
        """
        Full model evaluation pipeline with MLFlow tracking.
        """
        try:
            # Load existing experiment info
            with open('reports/experiment_info.json', 'r') as f:
                experiment_info = json.load(f)
            
            # MLFlow tracking is already set up in the manager's __init__ method
            
            # Start a new MLFlow run for evaluation
            with mlflow.start_run(run_name=f"model_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
                logging.info(f"Started MLFlow run: {run.info.run_id}")
                
                # Load and evaluate model
                model = self.load_model()
                df = self.load_data(test_file)
                
                # Data preprocessing
                for col in df.select_dtypes(include=['bool']).columns:
                    df[col] = df[col].astype(int)
                df['Risk'] = df['Risk'].map({'good': 1, 'bad': 0})
                
                X_test = df.iloc[:, :-1].values
                y_test = df.iloc[:, -1].values
                X_test = X_test.reshape((X_test.shape[0], 24, 1))
                
                logging.info(f"Test data shapes - X: {X_test.shape}, y: {y_test.shape}")
                
                # Log test dataset info
                mlflow.log_params({
                    'test_samples': len(X_test),
                    'feature_dim': X_test.shape[2],
                    'timesteps': X_test.shape[1]
                })
                
                # Evaluate model and get metrics
                metrics = self.evaluate_model(model, X_test, y_test)
                
                # Save metrics locally
                self.save_metrics(metrics)
                
                # Log metrics to MLFlow
                self.mlflow_manager.log_metrics(metrics)
                
                # Log evaluation plots as artifacts
                mlflow.log_artifacts('reports/figures', 'evaluation_plots')
                
                # Update experiment info
                experiment_info['evaluation'] = {
                    'run_id': run.info.run_id,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'metrics': metrics
                }
                
                with open('reports/experiment_info.json', 'w') as f:
                    json.dump(experiment_info, f, indent=4)
                
                # Log the updated experiment info
                mlflow.log_artifact('reports/experiment_info.json')
                
                # Check if model meets production criteria
                if self.mlflow_manager.should_register_model(metrics):
                    logging.info("Model meets production criteria")
                    # Get the model version from training run
                    try:
                        model_version = self.mlflow_manager.get_latest_production_model(experiment_info['model_name'])
                        if model_version:
                            logging.info(f"Current production model: {model_version}")
                        else:
                            logging.info("No current production model found")
                    except Exception as e:
                        logging.warning(f"Could not get production model info: {e}")
                
                logging.info(f"Evaluation completed. MLFlow run ID: {run.info.run_id}")
                print(f"Updated experiment info with evaluation metrics: {metrics}")
                
        except Exception as e:
            logging.error(f"Error in evaluation pipeline: {e}")
            raise

if __name__ == '__main__':
    evaluator = ModelEvaluator()
    evaluator.full_evaluation_pipeline()
