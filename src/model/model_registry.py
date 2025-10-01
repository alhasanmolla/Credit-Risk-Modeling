"""
Model Registry Utilities
This module provides utilities for managing models in the MLFlow registry.
"""

import os
import yaml
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

class ModelRegistry:
    def __init__(self, config_path: str = "mlflow_config.yaml"):
        """Initialize the model registry with configuration."""
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
        """Set up MLFlow tracking."""
        mlflow.set_tracking_uri(self.config['tracking_server']['uri'])
        
    def list_registered_models(self) -> List[Dict[str, Any]]:
        """List all registered models with their details."""
        models = self.client.search_registered_models()
        return [{
            'name': model.name,
            'creation_timestamp': datetime.fromtimestamp(model.creation_timestamp/1000.0),
            'last_updated_timestamp': datetime.fromtimestamp(model.last_updated_timestamp/1000.0),
            'description': model.description,
            'latest_versions': [
                {
                    'version': version.version,
                    'stage': version.current_stage,
                    'status': version.status,
                    'run_id': version.run_id
                }
                for version in model.latest_versions
            ]
        } for model in models]
        
    def get_model_versions(self, model_name: str) -> List[Dict[str, Any]]:
        """Get all versions of a specific model."""
        versions = self.client.search_model_versions(f"name='{model_name}'")
        return [{
            'version': version.version,
            'stage': version.current_stage,
            'status': version.status,
            'run_id': version.run_id,
            'creation_timestamp': datetime.fromtimestamp(version.creation_timestamp/1000.0),
            'last_updated_timestamp': datetime.fromtimestamp(version.last_updated_timestamp/1000.0),
            'description': version.description,
            'source': version.source,
            'metrics': self._get_version_metrics(version.run_id)
        } for version in versions]
        
    def _get_version_metrics(self, run_id: str) -> Dict[str, float]:
        """Get metrics for a specific model version."""
        try:
            run = self.client.get_run(run_id)
            return run.data.metrics
        except Exception as e:
            logging.warning(f"Could not fetch metrics for run {run_id}: {e}")
            return {}
            
    def transition_model_version(self, model_name: str, version: str, stage: str) -> None:
        """Transition a model version to a new stage."""
        if stage not in self.config['model_registry']['stages']:
            raise ValueError(f"Invalid stage: {stage}. Must be one of {self.config['model_registry']['stages']}")
            
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage
        )
        logging.info(f"Transitioned model {model_name} version {version} to {stage}")
        
    def archive_old_versions(self, model_name: str, keep_versions: int = 5) -> None:
        """Archive old versions of a model, keeping the most recent ones."""
        versions = self.get_model_versions(model_name)
        versions.sort(key=lambda x: int(x['version']), reverse=True)
        
        # Keep the specified number of most recent versions and production version
        keep_indices = set(range(keep_versions))
        production_versions = [i for i, v in enumerate(versions) if v['stage'] == 'Production']
        keep_indices.update(production_versions)
        
        for i, version in enumerate(versions):
            if i not in keep_indices and version['stage'] not in ['Archived', 'Production']:
                self.transition_model_version(model_name, version['version'], 'Archived')
                
    def delete_model_version(self, model_name: str, version: str) -> None:
        """Delete a specific version of a model."""
        self.client.delete_model_version(
            name=model_name,
            version=version
        )
        logging.info(f"Deleted model {model_name} version {version}")
        
    def get_latest_versions(self, model_name: str, stages: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Get the latest versions of a model, optionally filtered by stages."""
        versions = self.client.get_latest_versions(model_name, stages=stages)
        return [{
            'version': version.version,
            'stage': version.current_stage,
            'status': version.status,
            'run_id': version.run_id,
            'creation_timestamp': datetime.fromtimestamp(version.creation_timestamp/1000.0),
            'source': version.source
        } for version in versions]
        
    def compare_model_versions(self, model_name: str, version1: str, version2: str) -> Dict[str, Any]:
        """Compare two versions of a model."""
        v1_details = self.client.get_model_version(model_name, version1)
        v2_details = self.client.get_model_version(model_name, version2)
        
        v1_metrics = self._get_version_metrics(v1_details.run_id)
        v2_metrics = self._get_version_metrics(v2_details.run_id)
        
        return {
            'version1': {
                'version': version1,
                'stage': v1_details.current_stage,
                'metrics': v1_metrics,
                'creation_timestamp': datetime.fromtimestamp(v1_details.creation_timestamp/1000.0)
            },
            'version2': {
                'version': version2,
                'stage': v2_details.current_stage,
                'metrics': v2_metrics,
                'creation_timestamp': datetime.fromtimestamp(v2_details.creation_timestamp/1000.0)
            },
            'metric_differences': {
                k: v2_metrics.get(k, 0) - v1_metrics.get(k, 0)
                for k in set(v1_metrics) | set(v2_metrics)
            }
        }
        
    def get_model_lineage(self, model_name: str) -> List[Dict[str, Any]]:
        """Get the lineage of a model including all versions and their transitions."""
        versions = self.get_model_versions(model_name)
        versions.sort(key=lambda x: int(x['version']))
        
        lineage = []
        for version in versions:
            run = self.client.get_run(version['run_id'])
            lineage.append({
                'version': version['version'],
                'stage': version['stage'],
                'creation_time': version['creation_timestamp'],
                'metrics': version['metrics'],
                'parameters': run.data.params,
                'tags': run.data.tags
            })
            
        return lineage
        
    def promote_model_version(self, model_name: str, version: str, 
                            target_stage: str = 'Production',
                            archive_existing: bool = True) -> None:
        """Promote a model version to a target stage, optionally archiving existing versions in that stage."""
        if target_stage not in self.config['model_registry']['stages']:
            raise ValueError(f"Invalid target stage: {target_stage}")
            
        # Archive existing versions in target stage if requested
        if archive_existing:
            existing_versions = self.get_latest_versions(model_name, stages=[target_stage])
            for v in existing_versions:
                self.transition_model_version(model_name, v['version'], 'Archived')
                
        # Promote the specified version
        self.transition_model_version(model_name, version, target_stage)
        logging.info(f"Promoted model {model_name} version {version} to {target_stage}")
        
    def get_model_dependencies(self, model_name: str, version: str) -> Dict[str, Any]:
        """Get the dependencies and environment information for a specific model version."""
        model_version = self.client.get_model_version(model_name, version)
        run = self.client.get_run(model_version.run_id)
        
        # Get the model artifacts
        artifacts = self.client.list_artifacts(run.info.run_id)
        
        return {
            'run_id': run.info.run_id,
            'artifacts': [a.path for a in artifacts],
            'environment': {
                key: value
                for key, value in run.data.tags.items()
                if key.startswith('mlflow.')
            }
        }