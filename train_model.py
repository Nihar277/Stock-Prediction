import os
import numpy as np
import pandas as pd
import tensorflow as tf
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from datetime import datetime

# Paths
DATA_PATH = "./Data/"
MODEL_PATH = "./models/"

# Ensure directories exist
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)

# Check if data is up to date
def is_data_up_to_date(file):
    df = pd.read_csv(os.path.join(DATA_PATH, file))
    df['Date'] = pd.to_datetime(df['Date'])
    latest_date = df['Date'].max().date()
    return latest_date == datetime.today().date()

# Load stock data
def load_data(file):
    df = pd.read_csv(os.path.join(DATA_PATH, file))
    df = df[['Date', 'Close']].dropna()
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df

# Data Preprocessing
def preprocess_data(df):
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = scaler.fit_transform(df)
    return df_scaled, scaler

# Create sequences for LSTM
def create_sequences(data, seq_length=50):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

# Train Model for Each Stock
def train_all_models():
    files = [f for f in os.listdir(DATA_PATH) if f.endswith('.csv')]
    
    for file in files:
        stock = file.split('.')[0]

        # Check if data is up to date
        if is_data_up_to_date(file):
            print(f"✅ {stock} data is already up to date! Skipping training.")
            continue

        print(f"🔄 Training model for {stock}...")

        df = load_data(file)
        data_scaled, scaler = preprocess_data(df)
        X, y = create_sequences(data_scaled)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # Build LSTM Model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
            LSTM(50),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=1)

        # Save Model & Scaler
        model.save(f'{MODEL_PATH}{stock}_model.h5')
        np.save(f'{MODEL_PATH}{stock}_scaler.npy', scaler)

        print(f"✅ Model trained & saved for {stock}!")

if __name__ == "__main__":
    train_all_models()