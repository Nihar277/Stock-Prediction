import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    data = data[['Close']]
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    return scaled_data, scaler

def create_sequences(data, time_step=60):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

def build_and_train_model(X_train, y_train, X_test, y_test):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=4, batch_size=32, validation_data=(X_test, y_test), verbose=1)
    
    return model

def train_and_save_models(data_folder, models_folder):
    os.makedirs(models_folder, exist_ok=True)
    
    for file in os.listdir(data_folder):
        if file.endswith('.csv'):
            symbol = os.path.splitext(file)[0]
            file_path = os.path.join(data_folder, file)
            
            scaled_data, scaler = load_and_preprocess_data(file_path)
            time_step = 60
            X, y = create_sequences(scaled_data, time_step)
            
            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
            
            model = build_and_train_model(X_train, y_train, X_test, y_test)
            model_path = os.path.join(models_folder, f'{symbol}.h5')
            model.save(model_path)
            print(f'Model saved successfully: {model_path}')

def main():
    data_folder = 'Data'
    models_folder = 'models'
    train_and_save_models(data_folder, models_folder)

if __name__ == '__main__':
    main()
