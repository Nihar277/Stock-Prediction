# Import Required Libraries
from flask import Flask, jsonify, request
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

# Predict Future Prices
def predict_future_days(model, data, time_step, days, scaler):
    predictions = []
    last_sequence = data[-time_step:]
    for _ in range(days):
        next_day_prediction = model.predict(last_sequence.reshape(1, time_step, 1))
        predictions.append(next_day_prediction[0][0])
        last_sequence = np.append(last_sequence[1:], next_day_prediction)
    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# Flask Route for Predictions
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
    
    # Make predictions
    predictions = model.predict(X)
    predictions = scaler.inverse_transform(predictions)
    
    # Prepare data for JSON response
    dates = data.index[-len(y):].strftime('%Y-%m-%d').tolist()
    actual_prices = scaler.inverse_transform(y.reshape(-1, 1)).flatten().tolist()
    predicted_prices = predictions.flatten().tolist()
    
    # Create a plot
    plt.figure(figsize=(10, 6))
    plt.plot(dates, actual_prices, label='Actual Prices')
    plt.plot(dates, predicted_prices, label='Predicted Prices')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.title('Actual vs Predicted Stock Prices')
    plt.legend()
    plt.xticks(rotation=45)
    
    # Save plot to a BytesIO object
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    
    # Prepare JSON response
    response = {
        'dates': dates,
        'actual_prices': actual_prices,
        'predicted_prices': predicted_prices,
        'plot_url': plot_url
    }
    
    return jsonify(response)

# Run the Flask App
if __name__ == '__main__':
    app.run(debug=True)