# Feature engineering for credit risk modeling
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from src.logger import logging
import yaml
import pickle

class FeatureEngineering:

    @staticmethod  
    def load_params(params_path: str) -> dict:
        """Load parameters from a YAML file."""
        try:
            with open(params_path, 'r') as file:
                params = yaml.safe_load(file)
            logging.debug('Parameters retrieved from %s', params_path)
            return params
        except FileNotFoundError:
            logging.error('File not found: %s', params_path)
            raise
        except yaml.YAMLError as e:
            logging.error('YAML error: %s', e)
            raise
        except Exception as e:
            logging.error('Unexpected error: %s', e)
            raise

    @staticmethod    
    def load_data(file_path: str) -> pd.DataFrame:
        """Load data from a CSV file."""
        try:
            df = pd.read_csv(file_path)
            df.fillna('', inplace=True)
            logging.info('Data loaded and NaNs filled from %s', file_path)
            return df
        except pd.errors.ParserError as e:
            logging.error('Failed to parse the CSV file: %s', e)
            raise
        except Exception as e:
            logging.error('Unexpected error occurred while loading the data: %s', e)
            raise

    @staticmethod
    def feature_engineer(df):
        """
        Perform feature engineering: one-hot encode categoricals, scale numerics, separate target.
        Returns: (X, y)
        """
        # Separate features and target
        X = df.drop('Risk', axis=1)
        y = df['Risk']

        # Categorical and numeric columns
        categorical_cols = ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']
        numeric_cols = ['Age', 'Job', 'Credit amount', 'Duration']

        # One-hot encode categorical columns
        X_encoded = pd.get_dummies(X, columns=categorical_cols)

        # Scale numeric columns
        scaler = StandardScaler()
        X_encoded[numeric_cols] = scaler.fit_transform(X_encoded[numeric_cols])

        # Save scaler for future use
        os.makedirs("models", exist_ok=True)
        with open('models/feature_scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        logging.info('Scaler saved as feature_scaler.pkl')

        return X_encoded, y

    @staticmethod
    def save_data(df, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logging.info('Data saved to %s', file_path)

    @staticmethod
    def main():
        try:
            # Load train and test data
            train_data = pd.read_csv('./data/interim/train_processed.csv')
            test_data = pd.read_csv('./data/interim/test_processed.csv')
            logging.info('Train and test data loaded')

            # Feature engineering
            X_train, y_train = FeatureEngineering.feature_engineer(train_data)
            X_test, y_test = FeatureEngineering.feature_engineer(test_data)

            # Combine features and target for saving
            train_df = X_train.copy()
            train_df['Risk'] = y_train
            test_df = X_test.copy()
            test_df['Risk'] = y_test

            # Debugging: print head of processed data
            print("\n===== Train Features (head) =====")
            print(train_df.head())
            print("\n===== Test Features (head) =====")
            print(test_df.head())

            # Save processed data
            FeatureEngineering.save_data(train_df, os.path.join('./data', 'processed', 'train_features.csv'))
            FeatureEngineering.save_data(test_df, os.path.join('./data', 'processed', 'test_features.csv'))
            logging.info('Feature engineering completed and files saved')
        except Exception as e:
            logging.error('Feature engineering failed: %s', e)
            print(f"Error: {e}")

if __name__ == '__main__':
    FeatureEngineering.main()
