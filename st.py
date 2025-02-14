import os
import pandas as pd
import numpy as np
import logging
from flask import Flask, request, jsonify
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import pickle  # Add this line

# Setup logging
logging.basicConfig(
    filename='stock_prediction.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Define folders
data_folder = "./Data"
model_folder = "./models"
os.makedirs(data_folder, exist_ok=True)
os.makedirs(model_folder, exist_ok=True)

sequence_length = 60  # Look-back period

# Function to prepare data for training
def prepare_data(data, target_column='Close'):
    dataset = data[target_column].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    return X, y, scaler

# Function to build LSTM model
def build_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    # model.compile(optimizer='adam', loss='mse')
    model.compile(optimizer='adam', loss='mean_squared_error')  # Correct loss function

    return model

# Function to train models for all CSV files in data folder
def train_all_models():
    for file in os.listdir(data_folder):
        if file.endswith(".csv"):
            symbol = file.replace(".csv", "")
            print(f"Training model for {symbol}...")
            try:
                data = pd.read_csv(os.path.join(data_folder, file), index_col=0)
                X, y, scaler = prepare_data(data)
                
                model = build_model((sequence_length, 1))
                model.fit(X, y, epochs=5, batch_size=32, validation_split=0.1, verbose=1)
                
                model_path = os.path.join(model_folder, f"{symbol}.h5")
                model.save(model_path)  # Save as .h5
                scaler_path = os.path.join(model_folder, f"{symbol}_scaler.pkl")
                with open(scaler_path, "wb") as f:
                    pickle.dump(scaler, f)
                
                logging.info(f"Model trained and saved for {symbol} at {model_path}")
            except Exception as e:
                logging.error(f"Error training {symbol}: {str(e)}")

# Flask app to serve prediction requests
app = Flask(__name__)

# Endpoint to get predictions for a stock symbol
# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Parse request data
#         data = request.get_json()
#         symbol = data.get('symbol')
#         days_ahead = data.get('days_ahead', 2)  # Default is 2 days ahead
        
#         # Load trained model and scaler
#         model_path = os.path.join(model_folder, f"{symbol}.h5")
#         scaler_path = os.path.join(model_folder, f"{symbol}_scaler.pkl")
        
#         model = load_model(model_path)
#         with open(scaler_path, 'rb') as f:
#             scaler = pickle.load(f)
        
#         # Load stock data
#         stock_data = pd.read_csv(os.path.join(data_folder, f"{symbol}.csv"), index_col=0)
        
#         # Prepare data for prediction
#         X, _, _ = prepare_data(stock_data)
        
#         # Make predictions
#         predictions = []
#         for _ in range(days_ahead):
#             predicted_value = model.predict(X[-1].reshape(1, X.shape[1], 1))
#             predictions.append(predicted_value[0][0])
            
#             # Update X with the predicted value for the next prediction
#             X = np.roll(X, -1, axis=1)
#             X[0, -1, 0] = predicted_value
        
#         # Inverse scale the predictions
#         predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
#         return jsonify({"predictions": predictions.flatten().tolist()})
    
#     except Exception as e:
#         logging.error(f"Error during prediction: {str(e)}")
#         return jsonify({"error": "An error occurred during prediction."}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse request data
        data = request.get_json()
        symbol = data.get('symbol')
        days_ahead = data.get('days_ahead', 2)  # Default is 2 days ahead
        
        # Load trained model and scaler
        model_path = os.path.join(model_folder, f"{symbol}.h5")
        scaler_path = os.path.join(model_folder, f"{symbol}_scaler.pkl")
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            return jsonify({"error": f"Model or scaler file not found for symbol {symbol}"}), 400
        
        model = load_model(model_path)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        # Load stock data
        stock_data_path = os.path.join(data_folder, f"{symbol}.csv")
        if not os.path.exists(stock_data_path):
            return jsonify({"error": f"Stock data file not found for symbol {symbol}"}), 400
        
        stock_data = pd.read_csv(stock_data_path, index_col=0)
        
        # Prepare data for prediction
        X, _, _ = prepare_data(stock_data)
        
        # Make predictions
        predictions = []
        for _ in range(days_ahead):
            predicted_value = model.predict(X[-1].reshape(1, X.shape[1], 1))
            predictions.append(predicted_value[0][0])
            
            # Update X with the predicted value for the next prediction
            new_X = np.roll(X, -1, axis=1)
            new_X[0, -1, 0] = predicted_value  # Replace last value with prediction
            X = new_X
        
        # Inverse scale the predictions
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        return jsonify({"predictions": predictions.flatten().tolist()})
    
    except Exception as e:
        print(e)
        logging.error(f"Error during prediction: {str(e)}")
        return jsonify({"error": "An error occurred during prediction."}), 500

if __name__ == "__main__":
    # Train models first (uncomment this line if you want to train the models)
    # train_all_models()
    
    # Run the Flask app
    app.run(host="0.0.0.0", port=5000, debug=True)
