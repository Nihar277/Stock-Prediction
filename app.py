from flask import Flask, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import io
import base64

# Initialize Flask App
app = Flask(__name__)

# Load the Pre-trained LSTM Model
model = load_model('stock_price_lstm_model.h5')

# Load and Preprocess Data
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    data = data[['Close']]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    return data, scaled_data, scaler

# Create Sequences for LSTM
def create_sequences(data, time_step=60):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

# Predict Future Prices (1, 2, 5 days ahead)
def predict_future_prices(model, data, time_step, days, scaler):
    predictions = []
    last_sequence = data[-time_step:]
    
    # Predicting for specific days
    for _ in days:
        next_day_prediction = model.predict(last_sequence.reshape(1, time_step, 1))
        predictions.append(next_day_prediction[0][0])
        last_sequence = np.append(last_sequence[1:], next_day_prediction)  # Update for next prediction
    
    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

@app.route('/')
def index():
    return render_template('index.html')

# @app.route('/predict', methods=['GET'])
# def predict():
#     # File path to your stock data
#     file_path = 'Data/SBIN.csv'
    
#     # Load and preprocess data
#     data, scaled_data, scaler = load_and_preprocess_data(file_path)
    
#     # Create sequences
#     time_step = 60
#     X, y = create_sequences(scaled_data, time_step)
    
#     # Reshape data for LSTM
#     X = X.reshape(X.shape[0], X.shape[1], 1)
    
#     # Predict future prices (1, 2, 5 days ahead)
#     days_to_predict = [1, 2, 5]
#     future_predictions = predict_future_prices(model, scaled_data, time_step, days_to_predict, scaler)
    
#     # Prepare predicted prices for JSON response
#     predicted_prices = future_predictions.flatten().tolist()
    
#     # Get dates for the predictions (1, 2, 5 days ahead from the last date)
#     last_date = data.index[-1]
#     predicted_dates = [(last_date + pd.Timedelta(days=day)).strftime('%Y-%m-%d') for day in days_to_predict]
    
#     # Prepare actual data (last few actual prices) for JSON response
#     actual_prices = data['Close'].tail(60).tolist()  # Get last 60 actual prices
    
#     # Prepare the response for JSON output
#     response = {
#         'actual_data': {
#             'dates': data.index[-60:].strftime('%Y-%m-%d').tolist(),
#             'prices': actual_prices,
#         },
#         'predicted_data': {
#             'predicted_dates': predicted_dates,
#             'predicted_prices': predicted_prices,
#         }
#     }
    
#     return jsonify(response)

import matplotlib.pyplot as plt
import io
import base64

@app.route('/predict', methods=['GET'])
def predict():
    # File path to your stock data
    file_path = 'Data/SBIN.csv'
    
    # Load and preprocess data
    data, scaled_data, scaler = load_and_preprocess_data(file_path)
    
    # Create sequences
    time_step = 60
    X, y = create_sequences(scaled_data, time_step)
    
    # Reshape data for LSTM
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    # Predict future prices (1, 2, 5 days ahead)
    days_to_predict = [1, 2, 5]
    future_predictions = predict_future_prices(model, scaled_data, time_step, days_to_predict, scaler)
    
    # Prepare predicted prices for JSON response
    predicted_prices = future_predictions.flatten().tolist()
    
    # Get dates for the predictions (1, 2, 5 days ahead from the last date)
    last_date = data.index[-1]
    predicted_dates = [(last_date + pd.Timedelta(days=day)).strftime('%Y-%m-%d') for day in days_to_predict]
    
    # Prepare actual data (last few actual prices) for JSON response
    actual_prices = data['Close'].tail(60).tolist()
    actual_dates = data.index[-60:].strftime('%Y-%m-%d').tolist()
    
    # Combine actual and predicted data for plotting
    combined_dates = actual_dates + predicted_dates
    combined_prices = actual_prices + predicted_prices

    # Plot actual and predicted prices
    plt.figure(figsize=(10, 5))
    plt.plot(actual_dates, actual_prices, label='Actual Prices', color='blue', marker='o')
    plt.plot(predicted_dates, predicted_prices, label='Predicted Prices', color='red', marker='o', linestyle='dashed')
    plt.axvline(x=actual_dates[-1], color='black', linestyle='--', label="Prediction Start")
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.title("Stock Price Prediction")
    plt.legend()
    plt.xticks(rotation=45)
    
    # Convert the plot to a Base64 string
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')
    
    # Prepare the response for JSON output
    response = {
        'actual_data': {
            'dates': actual_dates,
            'prices': actual_prices,
        },
        'predicted_data': {
            'predicted_dates': predicted_dates,
            'predicted_prices': predicted_prices,
        },
    }
    
    return jsonify(response)


# Run the Flask App
if __name__ == '__main__':
    app.run(debug=True)
