import numpy as np
import pandas as pd
import tensorflow as tf
import mlflow
import mlflow.keras
import yaml
import os
import logging

logging.basicConfig(level=logging.INFO)

class ModelPredictor:
    def __init__(self, model_path=None, mlflow_model_uri=None, config_path='config.yaml'):
        """
        model_path: local h5 file path
        mlflow_model_uri: MLflow model URI (for logged model)
        """
        self.model_path = model_path
        self.mlflow_model_uri = mlflow_model_uri
        self.config_path = config_path
        self.config = self.load_config()
        self.model = self.load_model()

    def load_config(self, config_path=None):
        """Load hyperparameters/config from YAML"""
        if config_path is None:
            config_path = self.config_path
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logging.info("Config loaded from %s", config_path)
            return config
        logging.warning("Config file not found, using default settings")
        return {}

    def load_model(self):
        """Load trained model: either from local path or MLflow"""
        if self.mlflow_model_uri:
            logging.info("Loading model from MLflow URI: %s", self.mlflow_model_uri)
            model = mlflow.keras.load_model(self.mlflow_model_uri)
        elif self.model_path:
            logging.info("Loading model from local path: %s", self.model_path)
            model = tf.keras.models.load_model(self.model_path)
        else:
            raise ValueError("Either model_path or mlflow_model_uri must be provided")
        logging.info("Model loaded successfully")
        return model

    @staticmethod
    def load_data(file_path):
        """Load new data for prediction"""
        df = pd.read_csv(file_path)
        logging.info("Prediction data loaded from %s", file_path)
        return df

    @staticmethod
    def preprocess_data(df):
        """Reshape data for RNN input: (samples, timesteps, features)"""
        # Convert boolean columns to integers (same as training)
        bool_cols = df.select_dtypes(include=['bool']).columns
        df[bool_cols] = df[bool_cols].astype(int)
        
        # Drop the target column if it exists (we don't need it for prediction)
        if 'Risk' in df.columns:
            X = df.drop('Risk', axis=1).values
        else:
            X = df.values
            
        # Reshape for LSTM: (samples, timesteps, features)
        X_reshaped = X.reshape((X.shape[0], 24, 1))
        return X_reshaped

    def predict(self, df):
        """Predict class labels and probabilities"""
        X = self.preprocess_data(df)
        y_prob = self.model.predict(X).ravel()
        y_pred = (y_prob > 0.5).astype(int)
        results = pd.DataFrame({
            'prediction': y_pred,
            'probability': y_prob
        })
        return results

    def save_predictions(self, results, file_path='reports/predictions.csv'):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        results.to_csv(file_path, index=False)
        logging.info("Predictions saved to %s", file_path)

    def run_pipeline(self, data_file):
        df = self.load_data(data_file)
        results = self.predict(df)
        self.save_predictions(results)
        return results

if __name__ == '__main__':
    # Example usage:
    # Use the local model file we trained
    predictor = ModelPredictor(model_path='models/rnn_model.h5')
    
    # Or use an MLflow logged model (if you have the run ID):
    # predictor = ModelPredictor(mlflow_model_uri='runs:/<RUN_ID>/model')

    predictions = predictor.run_pipeline('./data/processed/test_features.csv')
    print(predictions.head())
