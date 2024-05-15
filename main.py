!pip install pycaret
!pip install scikit-learn

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
import xgboost as xgb
from pycaret.regression import setup, compare_models, evaluate_model, predict_model, plot_model, finalize_model, create_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM
import requests
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.ensemble import RandomForestRegressor


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls"
r = requests.get(url)
with open('Concrete_Data.xls', 'wb') as f:
    f.write(r.content)

concrete_data = pd.read_excel('Concrete_Data.xls')

print(concrete_data.info())
print(concrete_data.head())
print(concrete_data.describe())


correlation_matrix = concrete_data.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True,
            cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()


X = concrete_data.drop(columns=['Concrete compressive strength(MPa, megapascals) '])
y = concrete_data['Concrete compressive strength(MPa, megapascals) ']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Przeskalowanie cech
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Wykresy cech
features = concrete_data.columns.tolist()[:]
n_rows = 3
n_cols = (len(features) + n_rows - 1) // n_rows

plt.figure(figsize=(15, 10))
for i, feature in enumerate(features, 1):
    plt.subplot(n_rows, n_cols, i)
    sns.histplot(data=concrete_data, x=feature, kde=True)
    plt.title(feature)

plt.tight_layout()
plt.show()

# Metoda regresji wielomianowej

poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

poly_reg_model = LinearRegression()
poly_reg_model.fit(X_train_poly, y_train)
poly_reg_predictions = poly_reg_model.predict(X_test_poly)

print("Wyniki dla regresji wielomianowej:")
print(mean_absolute_percentage_error(y_test, poly_reg_predictions))

x_range = np.linspace(X_test.min(), X_test.max(), 100).reshape(-1, 8)
x_range_poly = poly.transform(scaler.transform(x_range))
y_range = poly_reg_model.predict(x_range_poly)
plt.plot(x_range, y_range, color='red', label='Regression Line')

poly_reg_predictions



def calculate_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return rmse, r2, mape

# Obliczanie metryk dla modelu regresji wielomianowej
poly_reg_rmse, poly_reg_r2, poly_reg_mape = calculate_metrics(y_test,
                                                poly_reg_predictions)

# Metoda XGBoost

xgb_model = xgb.XGBRegressor(seed=100,
  n_estimators=100,
  max_depth=10,
  learning_rate=0.1,
  min_child_weight=1,
  subsample=1,
  colsample_bytree=1,
  colsample_bylevel=1,
  gamma=0)
xgb_model.fit(X_train_scaled, y_train)
xgb_predictions = xgb_model.predict(X_test_scaled)
print("Wyniki dla modelu XGBoost:")
print(mean_absolute_percentage_error(y_test, xgb_predictions))

plt.figure(figsize=(10, 6))
plt.scatter(y_test, xgb_predictions, color='blue', label='Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='True values')
plt.title('Actual vs Predicted values (XGBoost)')
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.legend()
plt.show()

# Model RNN
rnn_model = Sequential([
    SimpleRNN(32, input_shape=(X_train_scaled.shape[1], 1)),
    Dense(1)
])
rnn_model.compile(optimizer='adam', loss='mean_squared_error')
rnn_model.fit(X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1)),
              y_train, epochs=100, batch_size=32)
rnn_predictions = rnn_model.predict(X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1)))
print(mean_absolute_percentage_error(y_test, rnn_predictions))

plt.figure(figsize=(10, 6))
plt.scatter(y_test, rnn_predictions, color='blue', label='Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='True values')
plt.title('Actual vs Predicted values (RNN)')
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.legend()
plt.show()

# Model LSTM
lstm_model = Sequential([
    LSTM(32, input_shape=(X_train_scaled.shape[1], 1)),
    Dense(1)
])
lstm_model.compile(optimizer='adam', loss='mean_squared_error')
lstm_model.fit(X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1)),
               y_train, epochs=100, batch_size=32)
print("LSTM: ", lstm_model )

lstm_predictions = lstm_model.predict(X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1)))
print(mean_absolute_percentage_error(y_test, lstm_predictions))

plt.figure(figsize=(10, 6))
plt.scatter(y_test, lstm_predictions, color='blue', label='Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='True values')
plt.title('Actual vs Predicted values (LSTM)')
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.legend()
plt.show()

#Model DNN
dnn_model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])
dnn_model.compile(optimizer='adam', loss='mean_squared_error')
dnn_model.fit(X_train_scaled, y_train, epochs=100, batch_size=32)
dnn_predictions = dnn_model.predict(X_test_scaled)
print("Wyniki dla modelu DNN:")
print(mean_absolute_percentage_error(y_test, dnn_predictions))

plt.figure(figsize=(10, 6))
plt.scatter(y_test, dnn_predictions, color='blue', label='Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='True values')
plt.title('Actual vs Predicted values (DNN)')
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.legend()
plt.show()

print("Predictions: \n",
      "Regression: ", poly_reg_predictions, "\n",
      "XGBoost: ", xgb_predictions, "\n",
      "RNN: ", rnn_predictions, "\n",
      "LSTM: ", lstm_predictions, "\n",
      "DNN: ", dnn_predictions, "\n")

# Calculate MAPE
xgb_mape = mean_absolute_percentage_error(y_test, xgb_predictions)
rnn_mape = mean_absolute_percentage_error(y_test, rnn_predictions)
lstm_mape = mean_absolute_percentage_error(y_test, lstm_predictions)
dnn_mape = mean_absolute_percentage_error(y_test, dnn_predictions)

# Calculate RMSE for XGBoost
xgb_rmse = mean_squared_error(y_test, xgb_predictions, squared=False)
rnn_rmse = mean_squared_error(y_test, rnn_predictions, squared=False)
lstm_rmse = mean_squared_error(y_test, lstm_predictions, squared=False)
dnn_rmse = mean_squared_error(y_test, dnn_predictions, squared=False)


# Calculate R2 for XGBoost
xgb_r2 = r2_score(y_test, xgb_predictions)
rnn_r2 = r2_score(y_test, rnn_predictions)
lstm_r2 = r2_score(y_test, lstm_predictions)
dnn_r2 = r2_score(y_test, dnn_predictions)

data = {
    'Model': ['XGBoost', 'RNN', 'LSTM', 'DNN'],
    'MAPE': [xgb_mape, rnn_mape, lstm_mape, dnn_mape],
    'RMSE': [xgb_rmse, rnn_rmse, lstm_rmse, dnn_rmse],
    'R2': [xgb_r2, rnn_r2, lstm_r2, dnn_r2]
}

metrics_df = pd.DataFrame(data)

# Виведення таблиці на екран
print(metrics_df)

def plot_regression_metrics(metrics_df, metric_name, color):
    plt.figure(figsize=(8, 6))
    plt.plot(metrics_df['Model'], metrics_df[metric_name], marker='o', color=color)
    plt.title(metric_name + ' with Regression')
    plt.xlabel('Model')
    plt.ylabel(metric_name)
    plt.xticks(rotation=45)
    plt.plot(metrics_df['Model'], metrics_df[metric_name], linestyle='dashed',
             color='red' if metric_name == 'MAPE' else 'blue')
    plt.show()

plot_regression_metrics(metrics_df, 'MAPE', 'skyblue')
plot_regression_metrics(metrics_df, 'RMSE', 'salmon')
plot_regression_metrics(metrics_df, 'R2', 'lightgreen')


# Użycie biblioteki PyCaret do automatyzacji procesu modelowania

regression_setup = setup(data=concrete_data, target='Concrete compressive strength(MPa, megapascals) ')
best_model = compare_models()
print(evaluate_model(best_model))
final_model = finalize_model(best_model)

plot_model(best_model, plot='residuals')
plot_model(best_model, plot='error')
plot_model(best_model, plot='feature')
plot_model(final_model, plot = 'residuals')
predictions = predict_model(final_model, data=X_test)

print("Wyniki dla najlepszego modelu wybranego przez PyCaret:")
print(mean_absolute_percentage_error(y_test, predictions['prediction_label']))


rf = RandomForestRegressor()
rf.fit(X_train, y_train)

feature_importance = pd.Series(rf.feature_importances_, index=X_train.columns)
sorted_feature_importance = feature_importance.sort_values(ascending=False)

top_features = sorted_feature_importance.head(5)
print("Top 5 Features:")
print(top_features)

for feature in top_features.index:
    new_feature_name = feature + '_squared'
    X_train[new_feature_name] = X_train[feature] ** 2
    X_test[new_feature_name] = X_test[feature] ** 2

new_features = X_train.columns

print("\nNew Features:")
print(new_features)

# Ponowne przetrenowanie modeli na pełnym zbiorze danych
poly_reg_model.fit(X_train_poly, y_train)
xgb_model.fit(X_train_scaled, y_train)
rnn_model.fit(X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1)),
              y_train, epochs=800, batch_size=32)
lstm_model.fit(X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1)),
               y_train, epochs=800, batch_size=32)
dnn_model.fit(X_train_scaled, y_train, epochs=800, batch_size=32, verbose=0)

# Przewidywanie na zbiorze testowym
poly_reg_predictions = poly_reg_model.predict(X_test_poly)
xgb_predictions = xgb_model.predict(X_test_scaled)
rnn_predictions = rnn_model.predict(X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1)))
lstm_predictions = lstm_model.predict(X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1)))
dnn_predictions = dnn_model.predict(X_test_scaled)

# Obliczenie metryk MAPE, RMSE i R2 dla każdego modelu
poly_reg_mape = mean_absolute_percentage_error(y_test, poly_reg_predictions)
xgb_mape = mean_absolute_percentage_error(y_test, xgb_predictions)
rnn_mape = mean_absolute_percentage_error(y_test, rnn_predictions)
lstm_mape = mean_absolute_percentage_error(y_test, lstm_predictions)
dnn_mape = mean_absolute_percentage_error(y_test, dnn_predictions)

poly_reg_rmse = mean_squared_error(y_test, poly_reg_predictions, squared=False)
xgb_rmse = mean_squared_error(y_test, xgb_predictions, squared=False)
rnn_rmse = mean_squared_error(y_test, rnn_predictions, squared=False)
lstm_rmse = mean_squared_error(y_test, lstm_predictions, squared=False)
dnn_rmse = mean_squared_error(y_test, dnn_predictions, squared=False)

poly_reg_r2 = r2_score(y_test, poly_reg_predictions)
xgb_r2 = r2_score(y_test, xgb_predictions)
rnn_r2 = r2_score(y_test, rnn_predictions)
lstm_r2 = r2_score(y_test, lstm_predictions)
dnn_r2 = r2_score(y_test, dnn_predictions)

# Tworzenie DataFrame z wynikami metryk
results_df = pd.DataFrame({
    'Model': ['Regresja wielomianowa', 'XGBoost', 'RNN', 'LSTM', 'DNN'],
    'MAPE': [poly_reg_mape, xgb_mape, rnn_mape, lstm_mape, dnn_mape],
    'RMSE': [poly_reg_rmse, xgb_rmse, rnn_rmse, lstm_rmse, dnn_rmse],
    'R2': [poly_reg_r2, xgb_r2, rnn_r2, lstm_r2, dnn_r2]
})
print(results_df)

def plot_model_predictions(y_test, predictions, model_name):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, predictions, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(model_name)
    plt.show()

plot_model_predictions(y_test, poly_reg_predictions, 'Polynomial Regression')
plot_model_predictions(y_test, xgb_predictions, 'XGBoost')
plot_model_predictions(y_test, rnn_predictions, 'RNN')
plot_model_predictions(y_test, lstm_predictions, 'LSTM')
plot_model_predictions(y_test, dnn_predictions, 'DNN')

print("The best model: ", best_model)
