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

total_row = homeless_data[homeless_data['State'] == 'Total']


years = [f'Year {year}' for year in range(2007, 2016)] + [f'Year {year}' for year in range(2017, 2022)]
print(f"Years considered for training: {years}")

total_homeless = total_row[years].values.flatten()

total_homeless = [int(value.replace(',', '')) for value in total_homeless]

X = np.array([int(year.split()[1]) for year in years]).reshape(-1, 1)
y = np.array(total_homeless)

results = {}

# Linear Regression
linear_model = LinearRegression()
linear_model.fit(X, y)
predicted_2022_linear = linear_model.predict([[2022]])
actual_2022 = int(total_row['Year 2022'].values[0].replace(',', ''))
mse_linear = mean_squared_error([actual_2022], predicted_2022_linear.astype(int))
mae_linear = mean_absolute_error([actual_2022], predicted_2022_linear.astype(int))
results['Linear Regression'] = {
    'Prediction': predicted_2022_linear[0],
    'MSE': mse_linear,
    'MAE': mae_linear
}
print(f"Linear Regression - Prediction: {predicted_2022_linear[0]:.0f}, MSE: {mse_linear:.2f}, MAE: {mae_linear:.2f}")

# SVM
svm_model = SVR(kernel='linear')
svm_model.fit(X, y)
predicted_2022_svm = svm_model.predict([[2022]])
mse_svm = mean_squared_error([actual_2022], predicted_2022_svm.astype(int))
mae_svm = mean_absolute_error([actual_2022], predicted_2022_svm.astype(int))
results['SVM'] = {
    'Prediction': predicted_2022_svm[0],
    'MSE': mse_svm,
    'MAE': mae_svm
}
print(f"SVM - Prediction: {predicted_2022_svm[0]:.0f}, MSE: {mse_svm:.2f}, MAE: {mae_svm:.2f}")

# Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X, y)
predicted_2022_rf = rf_model.predict([[2022]])
mse_rf = mean_squared_error([actual_2022], predicted_2022_rf.astype(int))
mae_rf = mean_absolute_error([actual_2022], predicted_2022_rf.astype(int))
results['Random Forest'] = {
    'Prediction': predicted_2022_rf[0],
    'MSE': mse_rf,
    'MAE': mae_rf
}
print(f"Random Forest - Prediction: {predicted_2022_rf[0]:.0f}, MSE: {mse_rf:.2f}, MAE: {mae_rf:.2f}")

X_norm = (X - X.min()) / (X.max() - X.min())
y_norm = (y - y.min()) / (y.max() - y.min())

# MLP
mlp_model = Sequential()
mlp_model.add(Dense(64, input_dim=1, activation='relu'))
mlp_model.add(Dense(32, activation='relu'))
mlp_model.add(Dense(1, activation='linear'))

mlp_model.compile(optimizer='adam', loss='mean_squared_error')
mlp_model.fit(X_norm, y_norm, epochs=100, verbose=0)

year_2022_norm = (2022 - X.min()) / (X.max() - X.min())
predicted_2022_norm = mlp_model.predict(np.array([[year_2022_norm]]))
predicted_2022_mlp = predicted_2022_norm[0][0] * (y.max() - y.min()) + y.min()

mse_mlp = mean_squared_error([actual_2022], [predicted_2022_mlp])
mae_mlp = mean_absolute_error([actual_2022], [predicted_2022_mlp])
results['MLP'] = {
    'Prediction': predicted_2022_mlp,
    'MSE': mse_mlp,
    'MAE': mae_mlp
}
print(f"MLP - Prediction: {predicted_2022_mlp:.0f}, MSE: {mse_mlp:.2f}, MAE: {mae_mlp:.2f}")

# LSTM
X_lstm = X_norm.reshape((X_norm.shape[0], 1, X_norm.shape[1]))


lstm_model = Sequential()
lstm_model.add(LSTM(50, activation='relu', input_shape=(X_lstm.shape[1], X_lstm.shape[2])))
lstm_model.add(Dense(1))

lstm_model.compile(optimizer='adam', loss='mean_squared_error')
lstm_model.fit(X_lstm, y_norm, epochs=200, verbose=0)

year_2022_norm = np.array([[year_2022_norm]])
year_2022_norm = year_2022_norm.reshape((year_2022_norm.shape[0], 1, year_2022_norm.shape[1]))

predicted_2022_norm = lstm_model.predict(year_2022_norm)
predicted_2022_lstm = predicted_2022_norm[0][0] * (y.max() - y.min()) + y.min()

mse_lstm = mean_squared_error([actual_2022], [predicted_2022_lstm])
mae_lstm = mean_absolute_error([actual_2022], [predicted_2022_lstm])
results['LSTM'] = {
    'Prediction': predicted_2022_lstm,
    'MSE': mse_lstm,
    'MAE': mae_lstm
}
print(f"LSTM - Prediction: {predicted_2022_lstm:.0f}, MSE: {mse_lstm:.2f}, MAE: {mae_lstm:.2f}")

for model, metrics in results.items():
    print(f"{model} - Prediction: {metrics['Prediction']:.0f}, MSE: {metrics['MSE']:.2f}, MAE: {metrics['MAE']:.2f}")
