# register_model.py

import json
import mlflow
import logging
from src.logger import logging
import os
import warnings


warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")


class ModelRegistry():

    @staticmethod
    def load_model_info(file_path: str) -> dict:
        """Load the model info from a JSON file."""
        try:
            with open(file_path, 'r') as file:
                model_info = json.load(file)
            logging.debug('Model info loaded from %s', file_path)
            return model_info
        except FileNotFoundError:
            logging.error('File not found: %s', file_path)
            raise
        except Exception as e:
            logging.error('Unexpected error occurred while loading the model info: %s', e)
            raise


    @staticmethod
    def register_model(model_name: str, model_info: dict):
        """Register the model to the MLflow Model Registry."""
        try:
            # URI for model from previous run
            model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"

            # Register the model in MLflow Model Registry
            model_version = mlflow.register_model(model_uri, model_name)

            # Transition the model to "Staging" stage
            client = mlflow.tracking.MlflowClient()
            try:
                client.transition_model_version_stage(
                    name=model_name,
                    version=model_version.version,
                    stage="Staging"
                )
                logging.info(
                    f'Model {model_name} version {model_version.version} registered '
                    f'and transitioned to Staging.'
                )
            except Exception as transition_error:
                # If transition fails, just log that registration succeeded
                logging.warning(
                    f'Model {model_name} version {model_version.version} registered '
                    f'but transition to Staging failed: {transition_error}'
                )
                logging.info(
                    f'Model {model_name} version {model_version.version} is ready in the registry.'
                )

        except Exception as e:
            logging.error('Error during model registration: %s', e)
            raise


    @staticmethod
    def main():
        try:
            # Use local MLflow backend (file-based tracking)
            tracking_uri = "file:///A:/end-to-end-data-scientist-project/Credit-Risk-Modeling/mlruns"
            mlflow.set_tracking_uri(tracking_uri)

            model_info_path = 'reports/experiment_info.json'
            model_info = ModelRegistry.load_model_info(model_info_path)

            # Model registry name
            model_name = "credit_risk_rnn"

            ModelRegistry.register_model(model_name, model_info)

        except Exception as e:
            logging.error('Failed to complete the model registration process: %s', e)
            print(f"Error: {e}")


if __name__ == '__main__':
        ModelRegistry.main()
