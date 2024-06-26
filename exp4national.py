import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

file_path = 'homeless use - homeless_data_reordered.csv'
homeless_data = pd.read_csv(file_path)

total_row = homeless_data[homeless_data['State'] == 'Total']

years = [f'Year {year}' for year in range(2007, 2016)] + [f'Year {year}' for year in range(2017, 2024)]
print(f"Years considered for training: {years}")

total_homeless = total_row[years].values.flatten()

total_homeless = [int(value.replace(',', '')) for value in total_homeless]

X = np.array([int(year.split()[1]) for year in years]).reshape(-1, 1)
y = np.array(total_homeless)

results = {}

# Linear Regression
linear_model = LinearRegression()
linear_model.fit(X, y)
predicted_2024_linear = linear_model.predict([[2024]])
results['Linear Regression'] = {
    'Prediction': predicted_2024_linear[0],
}
print(f"Linear Regression - Prediction: {predicted_2024_linear[0]:.0f}")

# SVM
svm_model = SVR(kernel='linear')
svm_model.fit(X, y)
predicted_2024_svm = svm_model.predict([[2024]])
results['SVM'] = {
    'Prediction': predicted_2024_svm[0],
}
print(f"SVM - Prediction: {predicted_2024_svm[0]:.0f}")

# Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X, y)
predicted_2024_rf = rf_model.predict([[2024]])
results['Random Forest'] = {
    'Prediction': predicted_2024_rf[0],
}
print(f"Random Forest - Prediction: {predicted_2024_rf[0]:.0f}")

# Normalize the data for neural networks
X_norm = (X - X.min()) / (X.max() - X.min())
y_norm = (y - y.min()) / (y.max() - y.min())

# MLP
mlp_model = Sequential()
mlp_model.add(Dense(64, input_dim=1, activation='relu'))
mlp_model.add(Dense(32, activation='relu'))
mlp_model.add(Dense(1, activation='linear'))

mlp_model.compile(optimizer='adam', loss='mean_squared_error')
mlp_model.fit(X_norm, y_norm, epochs=100, verbose=0)

year_2024_norm = (2024 - X.min()) / (X.max() - X.min())
predicted_2024_norm = mlp_model.predict(np.array([[year_2024_norm]]))
predicted_2024_mlp = predicted_2024_norm[0][0] * (y.max() - y.min()) + y.min()

results['MLP'] = {
    'Prediction': predicted_2024_mlp,
}
print(f"MLP - Prediction: {predicted_2024_mlp:.0f}")

# LSTM
X_lstm = X_norm.reshape((X_norm.shape[0], 1, X_norm.shape[1]))

lstm_model = Sequential()
lstm_model.add(LSTM(50, activation='relu', input_shape=(X_lstm.shape[1], X_lstm.shape[2])))
lstm_model.add(Dense(1))

lstm_model.compile(optimizer='adam', loss='mean_squared_error')
lstm_model.fit(X_lstm, y_norm, epochs=200, verbose=0)

year_2024_norm = np.array([[year_2024_norm]])
year_2024_norm = year_2024_norm.reshape((year_2024_norm.shape[0], 1, year_2024_norm.shape[1]))

predicted_2024_norm = lstm_model.predict(year_2024_norm)
predicted_2024_lstm = predicted_2024_norm[0][0] * (y.max() - y.min()) + y.min()

results['LSTM'] = {
    'Prediction': predicted_2024_lstm,
}
print(f"LSTM - Prediction: {predicted_2024_lstm:.0f}")

for model, metrics in results.items():
    print(f"{model} - Prediction: {metrics['Prediction']:.0f}")
