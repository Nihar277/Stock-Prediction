# Import Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Step 1: Load and Preprocess Data
def load_and_preprocess_data(file_path):
    # Load data
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    
    # Use 'Close' price for prediction
    data = data[['Close']]
    
    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    return data, scaled_data, scaler

# Step 2: Create Sequences for LSTM
def create_sequences(data, time_step=60):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

# Step 3: Build and Train LSTM Model
def build_and_train_model(X_train, y_train, X_test, y_test):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    history = model.fit(X_train, y_train, epochs=4, batch_size=32, validation_data=(X_test, y_test), verbose=1)
    
    return model

# Step 4: Predict Future Prices
def predict_future_days(model, data, time_step, days, scaler):
    predictions = []
    last_sequence = data[-time_step:]
    for _ in range(days):
        next_day_prediction = model.predict(last_sequence.reshape(1, time_step, 1))
        predictions.append(next_day_prediction[0][0])
        last_sequence = np.append(last_sequence[1:], next_day_prediction)
    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# Main Function
def main():
    # File path to your stock data
    file_path = 'Data/SBIN.csv'  # Replace with your file name
    
    # Step 1: Load and preprocess data
    data, scaled_data, scaler = load_and_preprocess_data(file_path)
    
    # Step 2: Create sequences
    time_step = 60
    X, y = create_sequences(scaled_data, time_step)
    
    # Step 3: Train-test split
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Reshape data for LSTM
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    # Step 4: Build and train the model
    model = build_and_train_model(X_train, y_train, X_test, y_test)
    
    # Step 5: Make predictions
    # Predict on test data
    test_predictions = model.predict(X_test)
    test_predictions = scaler.inverse_transform(test_predictions)
    
    # Plot actual vs predicted prices
    plt.plot(data.index[-len(y_test):], scaler.inverse_transform(y_test.reshape(-1, 1)), label='Actual')
    plt.plot(data.index[-len(y_test):], test_predictions, label='Predicted')
    plt.legend()
    plt.show()
    
    # Step 6: Predict future prices
    # 1-Day Prediction
    last_sequence = scaled_data[-time_step:]
    last_sequence = last_sequence.reshape(1, time_step, 1)
    next_day_prediction = model.predict(last_sequence)
    next_day_prediction = scaler.inverse_transform(next_day_prediction)
    print(f'1-Day Prediction: {next_day_prediction[0][0]}')
    
    # 2-Day Prediction
    two_day_predictions = predict_future_days(model, scaled_data, time_step, 2, scaler)
    print(f'2-Day Predictions: {two_day_predictions}')
    
    # 5-Day Prediction
    five_day_predictions = predict_future_days(model, scaled_data, time_step, 5, scaler)
    print(f'5-Day Predictions: {five_day_predictions}')

# Run the script
if __name__ == '__main__':
    main()