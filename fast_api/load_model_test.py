import tensorflow as tf
import pickle
import os
import logging
from datetime import datetime

# Set up logging to file
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'api_test.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def test_model_loading():
    """Test loading the trained model and scaler."""
    try:
        # Test loading the Keras model
        model_path = './models/rnn_model.h5'
        scaler_path = './models/feature_scaler.pkl'
        
        logger.info("Starting model loading test...")
        
        # Check and load the Keras model
        if os.path.exists(model_path):
            logger.info(f"Keras model file found: {model_path}")
            try:
                model = tf.keras.models.load_model(model_path)
                logger.info("Keras model loaded successfully!")
                logger.info(f"Model input shape: {model.input_shape}")
                logger.info(f"Model output shape: {model.output_shape}")
            except Exception as e:
                logger.error(f"Error loading the Keras model: {e}")
                return False
        else:
            logger.error(f"Keras model file not found: {model_path}")
            return False
        
        logger.info("="*50)
        
        # Check and load the pickle scaler file
        if os.path.exists(scaler_path):
            logger.info(f"Scaler pickle file found: {scaler_path}")
            try:
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                logger.info("Scaler loaded successfully!")
                logger.info(f"Scaler type: {type(scaler)}")
                if hasattr(scaler, 'feature_range'):
                    logger.info(f"Scaler feature range: {scaler.feature_range}")
                else:
                    logger.info(f"Scaler mean: {scaler.mean_[:5]}...")  # Show first 5 features
                    logger.info(f"Scaler scale: {scaler.scale_[:5]}...")  # Show first 5 features
            except Exception as e:
                logger.error(f"Error loading the scaler pickle file: {e}")
                return False
        else:
            logger.error(f"Scaler pickle file not found: {scaler_path}")
            return False
        
        logger.info("Model loading test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Unexpected error during model loading test: {e}")
        return False

if __name__ == "__main__":
    success = test_model_loading()
    if success:
        logger.info("All model loading tests passed!")
    else:
        logger.error("Some model loading tests failed!")
        exit(1)