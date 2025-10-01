import numpy as np
import pandas as pd
import os
import re
import logging

class DataPreprocessing():

    @staticmethod
    def preprocess_dataframe(df):
        # Example preprocessing steps:
        # 1. Drop missing values
        df = df.dropna()
        # 2. Convert categorical columns to dummy variables
        # df = pd.get_dummies(df)
        # 3. Scale numerical columns (optional)
        # from sklearn.preprocessing import StandardScaler
        # scaler = StandardScaler()
        # df[df.select_dtypes(include=np.number).columns] = scaler.fit_transform(df.select_dtypes(include=np.number))
        return df

    @staticmethod
    def main():
        try:
            # Fetch the data from data/raw
            train_data = pd.read_csv('./datas/raw/train.csv')
            test_data = pd.read_csv('./datas/raw/test.csv')
            logging.info('data loaded properly')

            # Transform the data
            train_processed_data = DataPreprocessing.preprocess_dataframe(train_data)
            test_processed_data = DataPreprocessing.preprocess_dataframe(test_data)

            # Store the data inside data/processed
            data_path = os.path.join("./datas", "interim")
            os.makedirs(data_path, exist_ok=True)
        
            train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
            test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)
        
            logging.info('Processed data saved to %s', data_path)
        except Exception as e:
            logging.error('Failed to complete the data transformation process: %s', e)
            print(f"Error: {e}")

if __name__ == '__main__':
    DataPreprocessing.main()