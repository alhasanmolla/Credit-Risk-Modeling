import numpy as np
import pandas as pd
import yaml
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization, Input, TimeDistributed, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import regularizers, optimizers, initializers
import optuna
import logging
import os

logging.basicConfig(level=logging.INFO)

class ModelBuilding:
    def __init__(self, config_path='config.yaml', model_path='models/rnn_model.h5'):
        self.config_path = config_path
        self.model_path = model_path
        self.config = self.load_config(config_path)

    @staticmethod
    def load_config(config_path):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logging.info("Config loaded successfully from %s", config_path)
        return config

    @staticmethod
    def load_data(file_path):
        df = pd.read_csv(file_path)
        logging.info("Data loaded from %s", file_path)
        return df

    def build_model(self, input_shape, trial=None):
        cfg = self.config
        if trial:
            units = trial.suggest_int('units', 32, 128)
            dropout_rate = trial.suggest_uniform('dropout', 0.0, 0.5)
            activation = trial.suggest_categorical('activation', ['tanh', 'relu'])
            rnn_type = trial.suggest_categorical('rnn_type', ['SimpleRNN', 'LSTM'])
            lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)
            clipnorm = trial.suggest_uniform('clipnorm', 0.0, 5.0)
            weight_decay = trial.suggest_loguniform('l2', 1e-6, 1e-2)
        else:
            units = cfg['units']
            dropout_rate = cfg['dropout']
            activation = cfg['activation']
            rnn_type = cfg['rnn_type']
            lr = cfg['lr']
            clipnorm = cfg['clipnorm']
            weight_decay = cfg['l2']

        reg = regularizers.l2(weight_decay)
        initializer = initializers.glorot_uniform()

        model = Sequential()
        model.add(Input(shape=input_shape))

        # Use LSTM first to handle 3D input: (timesteps, features)
        if rnn_type == "LSTM":
            model.add(LSTM(units, activation=activation, kernel_regularizer=reg,
                           kernel_initializer=initializer, return_sequences=False))
        else:
            model.add(tf.keras.layers.SimpleRNN(units, activation=activation,
                                                kernel_regularizer=reg,
                                                kernel_initializer=initializer,
                                                return_sequences=False))
        model.add(Dropout(dropout_rate))
        model.add(BatchNormalization())

        # Add dense layers after LSTM
        model.add(Dense(units // 2, activation=activation, kernel_regularizer=reg,
                        kernel_initializer=initializer))
        model.add(Dropout(dropout_rate))
        model.add(BatchNormalization())

        model.add(Dense(units // 4, activation=activation, kernel_regularizer=reg,
                        kernel_initializer=initializer))
        model.add(Dropout(dropout_rate))
        model.add(BatchNormalization())

        model.add(Dense(1, activation='sigmoid', kernel_regularizer=reg, kernel_initializer=initializer))

        opt = optimizers.Adam(learning_rate=lr, clipnorm=clipnorm)
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train_model(self, X_train, y_train, X_val, y_val, trial=None):
        model = self.build_model(input_shape=(X_train.shape[1], 1), trial=trial)
        callbacks_list = [
            EarlyStopping(monitor='val_loss', patience=self.config['patience'], restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
        ]
        model.fit(X_train, y_train, validation_data=(X_val, y_val),
                  epochs=self.config['epochs'], batch_size=self.config['batch_size'],
                  callbacks=callbacks_list, verbose=1)
        return model

    def save_model(self, model):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        model.save(self.model_path)
        logging.info("Model saved to %s", self.model_path)

    def run_optuna(self, X_train, y_train, X_val, y_val, n_trials=10):
        def objective(trial):
            model = self.build_model(input_shape=(X_train.shape[1], 1), trial=trial)
            model.fit(X_train, y_train, validation_data=(X_val, y_val),
                      epochs=self.config.get("epochs",10), batch_size=self.config.get("batch_size",32), verbose=0)
            val_acc = max(model.history.history['val_accuracy'])
            return 1.0 - val_acc

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        logging.info(f"Best Optuna params: {study.best_trial.params}")
        return study.best_trial

    def full_train_pipeline(self, train_file='./data/processed/train_features.csv'):
        df = self.load_data(train_file)
        
        # Convert boolean columns to integers
        for col in df.select_dtypes(include=['bool']).columns:
            df[col] = df[col].astype(int)
        
        # Convert target variable to numeric (good=1, bad=0)
        df['Risk'] = df['Risk'].map({'good': 1, 'bad': 0})
        
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        X_val, y_val = X[:100], y[:100]
        X_train, y_train = X[100:], y[100:]
        
        # Reshape for LSTM: (samples, timesteps, features)
        # Use 24 timesteps with 1 feature each (or vice versa)
        X_train = X_train.reshape((X_train.shape[0], 24, 1))
        X_val = X_val.reshape((X_val.shape[0], 24, 1))
        
        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"X_val shape: {X_val.shape}")
        print(f"y_val shape: {y_val.shape}")

        best_trial = self.run_optuna(X_train, y_train, X_val, y_val)
        final_model = self.train_model(X_train, y_train, X_val, y_val, trial=best_trial)
        self.save_model(final_model)

if __name__ == '__main__':
    trainer = ModelBuilding()
    trainer.full_train_pipeline()