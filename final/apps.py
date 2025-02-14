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
    
    current_sequence = last_sequence.copy()
    
    for _ in range(days):
        # Reshape for LSTM input
        current_input = current_sequence.reshape(1, time_step, 1)
        
        # Predict next day
        next_day = model.predict(current_input, verbose=0)
        predictions.append(next_day[0, 0])
        
        # Update sequence for next prediction
        current_sequence = np.roll(current_sequence, -1)
        current_sequence[-1] = next_day
    
    # Convert predictions to original scale
    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# Generate future dates
def generate_future_dates(last_date, num_days):
    future_dates = []
    current_date = pd.to_datetime(last_date)
    
    for i in range(num_days):
        # Skip weekends
        current_date = current_date + pd.Timedelta(days=1)
        while current_date.weekday() > 4:  # Skip Saturday (5) and Sunday (6)
            current_date = current_date + pd.Timedelta(days=1)
        future_dates.append(current_date.strftime('%Y-%m-%d'))
    
    return future_dates

# Flask Route for Historical and Future Predictions
@app.route('/predict', methods=['GET'])
def predict():
    try:       
        # Load the Pre-trained LSTM Model
        # Get parameters from query
        symbol = request.args.get('symbol', '').upper()
        future_days = int(request.args.get('future_days', 0))

        # Validate symbol
        if not symbol:  # Check if symbol is empty
            return jsonify({'error': 'symbol is required'}), 400

        model = load_model(f'models/{symbol}.h5')

        if future_days < 0:
            return jsonify({'error': 'future_days must be non-negative'}), 400
        
        file_path = f'Data/{symbol}.csv'

        # Load and preprocess data
        data, scaled_data, scaler = load_and_preprocess_data(file_path)
        
        # Create sequences
        time_step = 60
        X, y = create_sequences(scaled_data, time_step)
        
        # Reshape data for LSTM
        X = X.reshape(X.shape[0], X.shape[1], 1)
        
        # Make predictions for historical data
        historical_predictions = model.predict(X, verbose=0)
        historical_predictions = scaler.inverse_transform(historical_predictions)
        
        # Prepare historical data
        historical_dates = data.index[-len(y):].strftime('%Y-%m-%d').tolist()
        historical_actual = scaler.inverse_transform(y.reshape(-1, 1)).flatten().tolist()
        historical_predicted = historical_predictions.flatten().tolist()
        
        response = {
            'historical_data': {
                'dates': historical_dates,
                'actual_prices': historical_actual,
                'predicted_prices': historical_predicted
            }
        }
        
        # If future days requested, add future predictions
        if future_days > 0:
            future_predictions = predict_future_days(
                model, 
                scaled_data, 
                time_step, 
                future_days, 
                scaler
            )
            
            # Generate future dates
            future_dates = generate_future_dates(
                data.index[-1],
                future_days
            )
            
            response['future_predictions'] = {
                'dates': future_dates,
                'predicted_prices': future_predictions.flatten().tolist()
            }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the Flask App
if __name__ == '__main__':
    app.run(debug=True)