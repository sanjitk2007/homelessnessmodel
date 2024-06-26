import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

file_path = 'homeless use - homeless_data_reordered.csv'
homeless_data = pd.read_csv(file_path)

states = ['CA', 'FL', 'NY', 'TX', 'WA']

def prepare_data_2023(state):
    state_row = homeless_data[homeless_data['State'] == state]
    
    years = [f'Year {year}' for year in range(2007, 2016)] + [f'Year {year}' for year in range(2017, 2023)]
    homeless = state_row[years].values.flatten()
    
    homeless = [int(value.replace(',', '')) for value in homeless]
    
    return years, homeless

def predict_homeless_population_2023(state):
    years, homeless = prepare_data_2023(state)
    
    # Prepare the feature matrix
    X = np.array([int(year.split()[1]) for year in years]).reshape(-1, 1)
    y = np.array(homeless)

    results = {}

    # Linear Regression
    linear_model = LinearRegression()
    linear_model.fit(X, y)
    predicted_2023_linear = linear_model.predict([[2023]])
    actual_2023 = int(homeless_data[homeless_data['State'] == state]['Year 2023'].values[0].replace(',', ''))
    mse_linear = mean_squared_error([actual_2023], predicted_2023_linear)
    mae_linear = mean_absolute_error([actual_2023], predicted_2023_linear)
    results['Linear Regression'] = {
        'Prediction': predicted_2023_linear[0],
        'MSE': mse_linear,
        'MAE': mae_linear
    }
    print(f"{state} - Linear Regression - Prediction: {predicted_2023_linear[0]:.0f}, MSE: {mse_linear:.2f}, MAE: {mae_linear:.2f}")

    # SVM
    svm_model = SVR(kernel='linear')
    svm_model.fit(X, y)
    predicted_2023_svm = svm_model.predict([[2023]])
    mse_svm = mean_squared_error([actual_2023], predicted_2023_svm)
    mae_svm = mean_absolute_error([actual_2023], predicted_2023_svm)
    results['SVM'] = {
        'Prediction': predicted_2023_svm[0],
        'MSE': mse_svm,
        'MAE': mae_svm
    }
    print(f"{state} - SVM - Prediction: {predicted_2023_svm[0]:.0f}, MSE: {mse_svm:.2f}, MAE: {mae_svm:.2f}")

    # Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X, y)
    predicted_2023_rf = rf_model.predict([[2023]])
    mse_rf = mean_squared_error([actual_2023], predicted_2023_rf)
    mae_rf = mean_absolute_error([actual_2023], predicted_2023_rf)
    results['Random Forest'] = {
        'Prediction': predicted_2023_rf[0],
        'MSE': mse_rf,
        'MAE': mae_rf
    }
    print(f"{state} - Random Forest - Prediction: {predicted_2023_rf[0]:.0f}, MSE: {mse_rf:.2f}, MAE: {mae_rf:.2f}")

    X_norm = (X - X.min()) / (X.max() - X.min())
    y_norm = (y - y.min()) / (y.max() - y.min())

    # MLP
    mlp_model = Sequential()
    mlp_model.add(Dense(64, input_dim=1, activation='relu'))
    mlp_model.add(Dense(32, activation='relu'))
    mlp_model.add(Dense(1, activation='linear'))

    mlp_model.compile(optimizer='adam', loss='mean_squared_error')
    mlp_model.fit(X_norm, y_norm, epochs=100, verbose=0)

    year_2023_norm = (2023 - X.min()) / (X.max() - X.min())
    predicted_2023_norm = mlp_model.predict(np.array([[year_2023_norm]]))
    predicted_2023_mlp = predicted_2023_norm[0][0] * (y.max() - y.min()) + y.min()

    mse_mlp = mean_squared_error([actual_2023], [predicted_2023_mlp])
    mae_mlp = mean_absolute_error([actual_2023], [predicted_2023_mlp])
    results['MLP'] = {
        'Prediction': predicted_2023_mlp,
        'MSE': mse_mlp,
        'MAE': mae_mlp
    }
    print(f"{state} - MLP - Prediction: {predicted_2023_mlp:.0f}, MSE: {mse_mlp:.2f}, MAE: {mae_mlp:.2f}")

    # LSTM
    X_lstm = X_norm.reshape((X_norm.shape[0], 1, X_norm.shape[1]))

    lstm_model = Sequential()
    lstm_model.add(LSTM(50, activation='relu', input_shape=(X_lstm.shape[1], X_lstm.shape[2])))
    lstm_model.add(Dense(1))

    lstm_model.compile(optimizer='adam', loss='mean_squared_error')
    lstm_model.fit(X_lstm, y_norm, epochs=200, verbose=0)

    year_2023_norm = np.array([[year_2023_norm]])
    year_2023_norm = year_2023_norm.reshape((year_2023_norm.shape[0], 1, year_2023_norm.shape[1]))

    predicted_2023_norm = lstm_model.predict(year_2023_norm)
    predicted_2023_lstm = predicted_2023_norm[0][0] * (y.max() - y.min()) + y.min()

    mse_lstm = mean_squared_error([actual_2023], [predicted_2023_lstm])
    mae_lstm = mean_absolute_error([actual_2023], [predicted_2023_lstm])
    results['LSTM'] = {
        'Prediction': predicted_2023_lstm,
        'MSE': mse_lstm,
        'MAE': mae_lstm
    }
    print(f"{state} - LSTM - Prediction: {predicted_2023_lstm:.0f}, MSE: {mse_lstm:.2f}, MAE: {mae_lstm:.2f}")

    return results

state_results_2023 = {}
for state in states:
    state_results_2023[state] = predict_homeless_population_2023(state)

for state, results in state_results_2023.items():
    for model, metrics in results.items():
        print(f"{state} - {model} - Prediction: {metrics['Prediction']:.0f}, MSE: {metrics['MSE']:.2f}, MAE: {metrics['MAE']:.2f}")
