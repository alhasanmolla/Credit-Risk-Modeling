"""
MLFlow Utilities for Model Management
This module provides utility functions for MLFlow configuration and setup.
"""

import os
import yaml
import mlflow
from mlflow.tracking import MlflowClient
import mlflow.exceptions
import logging
from typing import Dict, Optional, Any

class MLFlowManager:
    def __init__(self, config_path: str = "mlflow_config.yaml"):
        """Initialize MLFlow configuration and setup tracking."""
        self.config = self._load_config(config_path)
        self.client = MlflowClient()
        self._setup_mlflow()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load MLFlow configuration from YAML file."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"MLFlow config file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
        
    def _setup_mlflow(self):
        """Set up MLFlow tracking and experiment."""
        # Set tracking URI
        mlflow.set_tracking_uri(self.config['tracking_server']['uri'])
        
        # Set up experiment
        experiment_name = self.config['experiment']['name']
        experiment = mlflow.get_experiment_by_name(experiment_name)
        
        if experiment is None:
            experiment_id = mlflow.create_experiment(
                experiment_name,
                tags=self.config['experiment']['tags']
            )
        else:
            experiment_id = experiment.experiment_id
            
        mlflow.set_experiment(experiment_name)
        self.experiment_id = experiment_id
        
    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None) -> mlflow.ActiveRun:
        """Start a new MLFlow run with optional name and tags."""
        return mlflow.start_run(
            run_name=run_name,
            tags=tags,
            experiment_id=self.experiment_id
        )
        
    def log_model(self, model, artifact_path: str = "model", registered_model_name: Optional[str] = None):
        """Log a model to MLFlow with optional registration."""
        mlflow.tensorflow.log_model(
            model,
            artifact_path=artifact_path,
            registered_model_name=registered_model_name if registered_model_name else self.config['model_registry']['model_name']
        )
        
    def register_model(self, model_uri: str, name: Optional[str] = None) -> str:
        """Register a model in the MLFlow registry."""
        model_name = name if name else self.config['model_registry']['model_name']
        result = mlflow.register_model(model_uri, model_name)
        return result.version
        
    def transition_model_stage(self, model_name: str, version: str, stage: str):
        """Transition a model version to a new stage."""
        if stage not in self.config['model_registry']['stages']:
            raise ValueError(f"Invalid stage: {stage}. Must be one of {self.config['model_registry']['stages']}")
            
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage
        )
        
    def get_latest_production_model(self, model_name: Optional[str] = None) -> Optional[str]:
        """Get the latest production model URI."""
        model_name = model_name if model_name else self.config['model_registry']['model_name']
        
        try:
            latest_version = self.client.get_latest_versions(model_name, stages=["Production"])
            if latest_version:
                return f"models:/{model_name}/{latest_version[0].version}"
            return None
        except Exception as e:
            logging.error(f"Error getting latest production model: {e}")
            return None
            
    def should_register_model(self, metrics: Dict[str, float]) -> bool:
        """Check if model meets criteria for registration."""
        criteria = self.config['versioning']['promotion_criteria']
        
        if not metrics:
            return False
            
        meets_criteria = (
            metrics.get('accuracy', 0) >= criteria['accuracy_threshold'] and
            metrics.get('auc', 0) >= criteria['auc_threshold']
        )
        
        return meets_criteria and self.config['versioning']['auto_register']
        
    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None):
        """Log artifacts to MLFlow."""
        if self.config['artifacts']['log_artifacts']:
            mlflow.log_artifacts(local_dir, artifact_path)
            
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to MLFlow."""
        if self.config['artifacts']['log_metrics']:
            mlflow.log_metrics(metrics, step=step)
            
    def log_params(self, params: Dict[str, Any]):
        """Log parameters to MLFlow, handling conflicts gracefully."""
        if self.config['artifacts']['log_params']:
            try:
                mlflow.log_params(params)
            except mlflow.exceptions.MlflowException as e:
                if "Changing param values is not allowed" in str(e):
                    logging.warning(f"Skipping parameter logging - some parameters already exist: {e}")
                else:
                    raise