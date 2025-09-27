import numpy as np
import pandas as pd
import tensorflow as tf
import mlflow
import json
import os
import logging

logging.basicConfig(level=logging.INFO)

class ModelEvaluator:
    def __init__(self, model_path='models/rnn_model.h5'):
        self.model_path = model_path

    @staticmethod
    def load_data(file_path):
        df = pd.read_csv(file_path)
        logging.info("Data loaded from %s", file_path)
        return df

    @staticmethod
    def evaluate_model(model, X_test, y_test):
        y_pred_prob = model.predict(X_test).ravel()
        y_pred = (y_pred_prob > 0.5).astype(int)
        accuracy = np.mean(y_pred == y_test)
        precision = np.sum((y_pred==1) & (y_test==1)) / max(1, np.sum(y_pred==1))
        recall = np.sum((y_pred==1) & (y_test==1)) / max(1, np.sum(y_test==1))
        auc = tf.keras.metrics.AUC()(y_test, y_pred_prob).numpy()
        metrics = {'accuracy': float(accuracy), 'precision': float(precision),
                   'recall': float(recall), 'auc': float(auc)}
        logging.info("Evaluation metrics: %s", metrics)
        return metrics

    @staticmethod
    def save_metrics(metrics, file_path='reports/metrics.json'):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        logging.info("Metrics saved to %s", file_path)

    def load_model(self):
        model = tf.keras.models.load_model(self.model_path)
        logging.info("Model loaded from %s", self.model_path)
        return model

    def full_evaluation_pipeline(self, test_file='./data/processed/test_features'
    '.csv'):
        mlflow.set_experiment("rnn_evaluation")
        with mlflow.start_run():
            model = self.load_model()
            df = self.load_data(test_file)
            
            # Convert boolean columns to integers (same as training)
            for col in df.select_dtypes(include=['bool']).columns:
                df[col] = df[col].astype(int)
            
            # Convert target variable to numeric (good=1, bad=0)
            df['Risk'] = df['Risk'].map({'good': 1, 'bad': 0})
            
            X_test = df.iloc[:, :-1].values
            y_test = df.iloc[:, -1].values
            
            # Reshape for LSTM: (samples, timesteps, features) - same as training
            X_test = X_test.reshape((X_test.shape[0], 24, 1))
            
            print(f"X_test shape: {X_test.shape}")
            print(f"y_test shape: {y_test.shape}")
            
            metrics = self.evaluate_model(model, X_test, y_test)
            self.save_metrics(metrics)
            for k,v in metrics.items():
                mlflow.log_metric(k,v)
            mlflow.keras.log_model(model, "model")

if __name__ == '__main__':
    evaluator = ModelEvaluator()
    evaluator.full_evaluation_pipeline()
