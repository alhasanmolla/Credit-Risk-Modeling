from src.data.data_ingestion import DataIngestion
from src.data.data_preprocessing import DataPreprocessing
from src.features.feature_engineering import FeatureEngineering
from src.model.model_building import ModelBuilding
from src.model.model_evaluation import ModelEvaluator
from src.model.register_model import ModelRegistry
from src.model.predict_model import ModelPredictor
from os.path import join, dirname, realpath
import logging

logging.basicConfig(level=logging.INFO)





if __name__ == "__main__":
    logging.info("The execution has started")

    try:
        # Data Ingestion
        data_ingestion = DataIngestion()
        data_ingestion.main()
        logging.info("Data ingestion completed")

        # Data Preprocessing
        data_preprocessor = DataPreprocessing()
        data_preprocessor.main()
        logging.info("Data preprocessing completed")

        # Feature Engineering
        feature_engineer = FeatureEngineering()
        feature_engineer.main()
        logging.info("Feature engineering completed")

        # Model Building
        model_builder = ModelBuilding()
        model_builder.full_train_pipeline()
        logging.info("Model building completed")

        # Model Evaluation
        model_evaluator = ModelEvaluator()
        model_evaluator.full_evaluation_pipeline()
        logging.info("Model evaluation completed")

        # Model Register
        model_register = ModelRegistry()
        model_register.main()
        logging.info("Model registration completed")

        # Model Predictor
        model_predictor = ModelPredictor(model_path='models/rnn_model.h5')
        model_predictor.run_pipeline('data/processed/test_features.csv')
        logging.info("Model prediction completed")


    except Exception as e:
        logging.error("An error occurred: %s", e)
        raise
    finally:
        logging.info("The execution has finished")
        


