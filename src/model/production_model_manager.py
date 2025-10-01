# production_model_manager.py
"""
Production Model Manager for MLflow Model Registry
This script handles model promotion to production and retrieval of latest production models.
"""

import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
import logging
from src.logger import logging
import os
import json
from datetime import datetime
import warnings
from typing import Dict, List, Optional, Any
from src.model.model_registry import ModelRegistry

warnings.filterwarnings("ignore")

class ProductionModelManager:
    """Manages model lifecycle for production deployment."""
    
    def __init__(self, config_path: str = "mlflow_config.yaml"):
        """Initialize the production model manager."""
        self.registry = ModelRegistry(config_path)
        self.client = self.registry.client
        self.config = self.registry.config
        self.tracking_uri = self.config['tracking_server']['uri']
        mlflow.set_tracking_uri(self.tracking_uri)
        
    def get_latest_model_version(self, model_name="credit_risk_rnn", stage="Staging"):
        """
        Get the latest version of a model in a specific stage.
        
        Args:
            model_name (str): Name of the registered model
            stage (str): Stage name (Staging, Production, Archived)
            
        Returns:
            dict: Model version information including version number, run_id, and metadata
        """
        try:
            # Get all versions of the model
            model_versions = self.client.search_model_versions(f"name='{model_name}'")
            
            if not model_versions:
                logging.error(f"No versions found for model: {model_name}")
                return None
            
            # Filter versions by stage and get the latest
            stage_versions = [mv for mv in model_versions if mv.current_stage == stage]
            
            if not stage_versions:
                logging.warning(f"No versions found in stage '{stage}' for model: {model_name}")
                # Return the latest version regardless of stage
                latest_version = max(model_versions, key=lambda x: int(x.version))
                logging.info(f"Returning latest version regardless of stage: {latest_version.version}")
            else:
                latest_version = max(stage_versions, key=lambda x: int(x.version))
                logging.info(f"Found latest version in {stage}: {latest_version.version}")
            
            # Extract model information
            model_info = {
                "version": int(latest_version.version),
                "run_id": latest_version.run_id,
                "stage": latest_version.current_stage,
                "status": latest_version.status,
                "creation_timestamp": datetime.fromtimestamp(latest_version.creation_timestamp / 1000).strftime('%Y-%m-%d %H:%M:%S'),
                "last_updated_timestamp": datetime.fromtimestamp(latest_version.last_updated_timestamp / 1000).strftime('%Y-%m-%d %H:%M:%S'),
                "description": latest_version.description or "No description",
                "source": latest_version.source,
                "model_uri": f"models:/{model_name}/{latest_version.version}"
            }
            
            logging.info(f"Latest model version retrieved successfully:")
            logging.info(f"  Model: {model_name}")
            logging.info(f"  Version: {model_info['version']}")
            logging.info(f"  Stage: {model_info['stage']}")
            logging.info(f"  Run ID: {model_info['run_id']}")
            logging.info(f"  Model URI: {model_info['model_uri']}")
            
            return model_info
            
        except Exception as e:
            logging.error(f"Error retrieving latest model version: {e}")
            raise
    
    def promote_model_to_production(self, model_name="credit_risk_rnn", version=None):
        """
        Promote a model version to production.
        
        Args:
            model_name (str): Name of the registered model
            version (int): Version number to promote. If None, promotes the latest Staging version.
            
        Returns:
            bool: True if promotion successful, False otherwise
        """
        try:
            if version is None:
                # Get the latest version from Staging
                latest_info = self.get_latest_model_version(model_name, stage="Staging")
                if latest_info is None:
                    logging.error("No model found in Staging to promote to Production")
                    return False
                version = latest_info["version"]
            
            # Transition the model to Production
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage="Production",
                archive_existing_versions=True  # Archive current production versions
            )
            
            logging.info(f"Model {model_name} version {version} successfully promoted to Production")
            
            # Get updated model info
            production_info = self.get_latest_model_version(model_name, stage="Production")
            return production_info is not None
            
        except Exception as e:
            logging.error(f"Error promoting model to production: {e}")
            return False
    
    def get_production_model_info(self, model_name="credit_risk_rnn"):
        """
        Get detailed information about the current production model.
        
        Args:
            model_name (str): Name of the registered model
            
        Returns:
            dict: Production model information or None if no production model exists
        """
        return self.get_latest_model_version(model_name, stage="Production")
    
    def load_production_model(self, model_name="credit_risk_rnn"):
        """
        Load the current production model for inference.
        
        Args:
            model_name (str): Name of the registered model
            
        Returns:
            mlflow.pyfunc.PyFuncModel: Loaded production model
        """
        try:
            # Get production model info
            prod_info = self.get_production_model_info(model_name)
            if prod_info is None:
                logging.error("No production model found")
                return None
            
            # Load the model using the model URI
            model_uri = prod_info["model_uri"]
            model = mlflow.pyfunc.load_model(model_uri)
            
            logging.info(f"Production model loaded successfully from {model_uri}")
            return model
            
        except Exception as e:
            logging.error(f"Error loading production model: {e}")
            return None
    
    def list_all_model_versions(self, model_name="credit_risk_rnn"):
        """
        List all versions of a model with their stages.
        
        Args:
            model_name (str): Name of the registered model
            
        Returns:
            list: List of model version information dictionaries
        """
        try:
            model_versions = self.client.search_model_versions(f"name='{model_name}'")
            
            versions_info = []
            for mv in model_versions:
                version_info = {
                    "version": int(mv.version),
                    "stage": mv.current_stage,
                    "status": mv.status,
                    "run_id": mv.run_id,
                    "creation_timestamp": datetime.fromtimestamp(mv.creation_timestamp / 1000).strftime('%Y-%m-%d %H:%M:%S'),
                    "model_uri": f"models:/{model_name}/{mv.version}"
                }
                versions_info.append(version_info)
            
            # Sort by version number (descending)
            versions_info.sort(key=lambda x: x["version"], reverse=True)
            
            return versions_info
            
        except Exception as e:
            logging.error(f"Error listing model versions: {e}")
            return []

    def compare_model_performance(self, model_name: str, version1: str, version2: str) -> Dict[str, Any]:
        """
        Compare performance metrics between two model versions.
        
        Args:
            model_name (str): Name of the registered model
            version1 (str): First version number
            version2 (str): Second version number
            
        Returns:
            dict: Comparison results including metrics differences
        """
        return self.registry.compare_model_versions(model_name, version1, version2)
    
    def get_model_lineage(self, model_name: str) -> List[Dict[str, Any]]:
        """
        Get the complete lineage of a model including all versions and transitions.
        
        Args:
            model_name (str): Name of the registered model
            
        Returns:
            list: Model lineage information
        """
        return self.registry.get_model_lineage(model_name)
    
    def get_model_dependencies(self, model_name: str, version: str) -> Dict[str, Any]:
        """
        Get dependencies and environment information for a specific model version.
        
        Args:
            model_name (str): Name of the registered model
            version (str): Version number
            
        Returns:
            dict: Model dependencies and environment information
        """
        return self.registry.get_model_dependencies(model_name, version)
    
    def archive_old_versions(self, model_name: str, keep_versions: int = 5) -> None:
        """
        Archive old model versions while keeping the most recent ones.
        
        Args:
            model_name (str): Name of the registered model
            keep_versions (int): Number of recent versions to keep
        """
        self.registry.archive_old_versions(model_name, keep_versions)
    
    def validate_model_promotion(self, model_name: str, version: str, 
                               target_stage: str = "Production") -> bool:
        """
        Validate if a model version meets the criteria for promotion to target stage.
        
        Args:
            model_name (str): Name of the registered model
            version (str): Version number to validate
            target_stage (str): Target stage for promotion
            
        Returns:
            bool: True if model meets promotion criteria, False otherwise
        """
        try:
            # Get model metrics
            model_version = self.client.get_model_version(model_name, version)
            run = self.client.get_run(model_version.run_id)
            metrics = run.data.metrics
            
            # Get promotion criteria from config
            criteria = self.config['model_registry'].get('promotion_criteria', {}).get(target_stage, {})
            
            if not criteria:
                logging.warning(f"No promotion criteria defined for stage {target_stage}")
                return True
            
            # Validate metrics against criteria
            for metric_name, threshold in criteria.items():
                if metric_name not in metrics:
                    logging.warning(f"Required metric {metric_name} not found in model metrics")
                    return False
                
                if metrics[metric_name] < threshold:
                    logging.info(f"Model did not meet {metric_name} threshold: {metrics[metric_name]} < {threshold}")
                    return False
            
            logging.info("Model meets all promotion criteria")
            return True
            
        except Exception as e:
            logging.error(f"Error validating model promotion: {e}")
            return False
    
    def rollback_production_model(self, model_name: str) -> bool:
        """
        Rollback to the previous production model version.
        
        Args:
            model_name (str): Name of the registered model
            
        Returns:
            bool: True if rollback successful, False otherwise
        """
        try:
            # Get all versions in production stage
            prod_versions = self.registry.get_model_versions(model_name)
            prod_versions = [v for v in prod_versions if v['stage'] == 'Production']
            
            if len(prod_versions) < 2:
                logging.error("Not enough production versions for rollback")
                return False
            
            # Sort by version number (descending)
            prod_versions.sort(key=lambda x: int(x['version']), reverse=True)
            
            # Archive current production version
            current_version = prod_versions[0]['version']
            self.registry.transition_model_version(model_name, current_version, 'Archived')
            
            # Promote previous version back to production
            previous_version = prod_versions[1]['version']
            self.registry.transition_model_version(model_name, previous_version, 'Production')
            
            logging.info(f"Successfully rolled back from version {current_version} to {previous_version}")
            return True
            
        except Exception as e:
            logging.error(f"Error during rollback: {e}")
            return False

def main():
    """Main function to demonstrate production model management."""
    
    # Initialize the production model manager
    manager = ProductionModelManager()
    
    model_name = "credit_risk_rnn"
    
    print("=" * 60)
    print("PRODUCTION MODEL MANAGEMENT DASHBOARD")
    print("=" * 60)
    
    # 1. Get latest staging model
    print("\n1. LATEST STAGING MODEL:")
    print("-" * 30)
    staging_info = manager.get_latest_model_version(model_name, stage="Staging")
    if staging_info:
        print(f"Version: {staging_info['version']}")
        print(f"Run ID: {staging_info['run_id']}")
        print(f"Created: {staging_info['creation_timestamp']}")
        print(f"Model URI: {staging_info['model_uri']}")
    else:
        print("No staging model found")
    
    # 2. Get current production model
    print("\n2. CURRENT PRODUCTION MODEL:")
    print("-" * 30)
    prod_info = manager.get_production_model_info(model_name)
    if prod_info:
        print(f"Version: {prod_info['version']}")
        print(f"Run ID: {prod_info['run_id']}")
        print(f"Created: {prod_info['creation_timestamp']}")
        print(f"Model URI: {prod_info['model_uri']}")
        
        # Show model dependencies
        deps = manager.get_model_dependencies(model_name, prod_info['version'])
        print("\nModel Dependencies:")
        print(f"Environment: {deps['environment'].get('mlflow.source.name', 'Unknown')}")
        print("Artifacts:", ", ".join(deps['artifacts']))
    else:
        print("No production model found")
    
    # 3. List all model versions with lineage
    print("\n3. MODEL LINEAGE:")
    print("-" * 30)
    lineage = manager.get_model_lineage(model_name)
    for version in lineage:
        print(f"Version {version['version']} ({version['stage']}):")
        print(f"  Created: {version['creation_time']}")
        print(f"  Metrics: {version['metrics']}")
        print(f"  Parameters: {version['parameters']}")
    
    # 4. Compare staging and production models (if both exist)
    if staging_info and prod_info:
        print("\n4. MODEL COMPARISON:")
        print("-" * 30)
        comparison = manager.compare_model_performance(model_name, 
                                                     staging_info['version'],
                                                     prod_info['version'])
        print("Metric Differences (Staging - Production):")
        for metric, diff in comparison['metric_differences'].items():
            print(f"  {metric}: {diff:+.4f}")
    
    # 5. Validate and promote staging to production (optional)
    if staging_info:
        print("\n5. VALIDATING STAGING MODEL FOR PRODUCTION:")
        print("-" * 30)
        is_valid = manager.validate_model_promotion(model_name, staging_info['version'])
        if is_valid:
            print("Model meets promotion criteria")
            if not prod_info:
                print("Promoting to production...")
                success = manager.promote_model_to_production(model_name)
                if success:
                    print("Promotion successful!")
        else:
            print("Model does not meet promotion criteria")
            print("Promotion failed!")
    
    print("\n" + "=" * 60)
    print("DASHBOARD COMPLETE")
    print("=" * 60)
    
    # Save production model to expected location for DVC
    if prod_info:
        print("\nSaving production model to models/production_model.h5...")
        try:
            # Load the production model using mlflow.keras
            import mlflow.keras
            model_uri = prod_info["model_uri"]
            keras_model = mlflow.keras.load_model(model_uri)
            
            # Save the model to the expected location
            keras_model.save('models/production_model.h5')
            print("Production model saved successfully!")
            
        except Exception as e:
            print(f"Error saving production model: {e}")
            # Fallback: copy the existing model file if available
            try:
                import shutil
                if os.path.exists('models/rnn_model.h5'):
                    shutil.copy('models/rnn_model.h5', 'models/production_model.h5')
                    print("Production model copied from rnn_model.h5")
                else:
                    print("No model file available to copy")
            except Exception as fallback_error:
                print(f"Fallback also failed: {fallback_error}")

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    main()