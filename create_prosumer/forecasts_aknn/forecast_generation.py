# Copyright (c) 2019 Nikhil Singh
# GNU General Public License v3.0+ (see COPYING or https://www.gnu.org/licenses/gpl-3.0.txt)
# name: PV and Wind power generation forecast via LR, KNN, GBR, MLP, Ridge, Lasso and AKNN
# author: Nikhil Singh (nikkytub@gmail.com)
# data-source: Pecan Street Data from 01/01/2015 to 31/12/2015

import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from lpi_python import lpi_distance, lpi_mean
import numpy as np
import math
import matplotlib.pyplot as plt
import csv

data = pd.read_csv('data.csv', parse_dates=True)

# To predict the load in 60 minutes intervals because optimization team need it in 60 minutes intervals
# 01/01/2015 to 30/12/2015
ini_data = data[:34944:4]

# 30/12/2015
prev_day_data = data[34848:34944:4]

# 31/12/2015
last_day_data = data[34944::4]

y_test_pv_31_12 = last_day_data['PV']
y_test_wind_31_12 = last_day_data['Wind']

generation_pv_30_12 = np.array(prev_day_data['PV'])
generation_wind_30_12 = np.array(prev_day_data['Wind'])

y_train_pv = np.array(ini_data['PV'])
y_train_wind = np.array(ini_data['Wind'])

chunks_pv = [y_train_pv[x:x + 24] for x in range(0, len(y_train_pv), 24)]
chunks_wind = [y_train_wind[x:x + 24] for x in range(0, len(y_train_wind), 24)]

time_last_day = np.array(last_day_data['Unnamed: 0'])

# Using dew-point and wind-speed as a feature vector for PV and Wind
x_pv = ini_data['dew_point']
X_PV = x_pv.values.reshape(-1, 1)

x_wind = ini_data['wind_speed']
X_WIND = x_wind.values.reshape(-1, 1)

# Labels
Y_PV = ini_data['PV']
Y_WIND = ini_data['Wind']

x_test_pv_31_12 = last_day_data['dew_point']
x_test_pv_31_12 = x_test_pv_31_12.values.reshape(-1, 1)

x_test_wind_31_12 = last_day_data['wind_speed']
x_test_wind_31_12 = x_test_wind_31_12.values.reshape(-1, 1)

X_train_pv, X_test_pv, Y_train_pv, Y_test_pv = train_test_split(X_PV, Y_PV, test_size=0.25, shuffle=False)
X_train_wind, X_test_wind, Y_train_wind, Y_test_wind = train_test_split(X_WIND, Y_WIND, test_size=0.25, shuffle=False)


def aknn(load, chunks):
    x_generation = []
    dist = []
    d1_dist = dict()
    for x in chunks:
        x_generation.append(x)
        d = lpi_distance(load, x)
        dist.append(d)
        d1_dist.update({d:x})
    sorted_dict = dict()
    for key in sorted(d1_dist.keys()):
        sorted_dict.update({key: d1_dist[key]})
    d1_generation = []
    for key in sorted_dict.keys():
        d1_generation.append(sorted_dict[key])
    m = lpi_mean(d1_generation[:6])
    return m


def prediction(load, chunks):
    aknn_predicted_load_school = [aknn(load, chunks)]
    plot_values = []
    for pred in aknn_predicted_load_school:
        for l in pred:
            plot_values.append(l)
    return plot_values


def step_graph(prediction, actual, ylabl, xlabl):
    hour = []
    for i in range(24):
        hour.append(i)
    plt.step(hour, prediction, label='Predicted')
    plt.step(hour, actual.values, label='Actual')
    plt.ylabel(ylabl)
    plt.xticks([0, 5, 10, 15, 20],
               ['00:00', '05:00', '10:00', '15:00', '20:00'])
    plt.xlabel(xlabl)
    plt.legend()
    return plt.show()


def cont_graph(prediction, actual, ylabl, xlabl):
    plt.plot(prediction, label='Predicted')
    plt.plot(actual.values, label='Actual')
    plt.ylabel(ylabl)
    plt.xlabel(xlabl)
    plt.legend()
    return plt.show()


def continuous_graph(prediction, actual, ylabl, xlabl):
    plt.plot(prediction, label='Predicted')
    plt.plot(actual.values, label='Actual')
    plt.ylabel(ylabl)
    plt.xticks([0, 5, 10, 15, 20],
               ['00:00', '05:00', '10:00', '15:00', '20:00'])
    plt.xlabel(xlabl)
    plt.legend()
    return plt.show()


# Linear regression PV
linear_model_pv = LinearRegression()
linear_model_pv.fit(X_train_pv, Y_train_pv)
y_predict_pv = linear_model_pv.predict(X_test_pv)
y_predict_31_12_pv = linear_model_pv.predict(x_test_pv_31_12)
continuous_graph(y_predict_31_12_pv, y_test_pv_31_12, 'PV power generation in kW via LR(31.12.2015)', 'Hour')
step_graph(y_predict_31_12_pv, y_test_pv_31_12, 'PV power generation in kW via LR(31.12.2015)', 'Hour')

# Linear regression Wind
linear_model_wind = LinearRegression()
linear_model_wind.fit(X_train_wind, Y_train_wind)
y_predict_wind = linear_model_wind.predict(X_test_wind)
y_predict_31_12_wind = linear_model_wind.predict(x_test_wind_31_12)
continuous_graph(y_predict_31_12_wind, y_test_wind_31_12, 'Wind power generation in kW via LR(31.12.2015)', 'Hour')
step_graph(y_predict_31_12_wind, y_test_wind_31_12, 'Wind power generation in kW via LR(31.12.2015)', 'Hour')

# KNN PV
knn_model_pv = KNeighborsRegressor(n_neighbors=6)
knn_model_pv.fit(X_train_pv, Y_train_pv)
k_predict_pv = knn_model_pv.predict(X_test_pv)
k_predict_31_12_pv = knn_model_pv.predict(x_test_pv_31_12)
continuous_graph(k_predict_31_12_pv, y_test_pv_31_12, 'PV power generation in kW via KNN(31.12.2015)', 'Hour')
step_graph(k_predict_31_12_pv, y_test_pv_31_12, 'PV power generation in kW via KNN(31.12.2015)', 'Hour')

# KNN Wind
knn_model_wind = KNeighborsRegressor(n_neighbors=6)
knn_model_wind.fit(X_train_wind, Y_train_wind)
k_predict_wind = knn_model_wind.predict(X_test_wind)
k_predict_31_12_wind = knn_model_wind.predict(x_test_wind_31_12)
continuous_graph(k_predict_31_12_wind, y_test_wind_31_12, 'Wind power generation in kW via KNN(31.12.2015)', 'Hour')
step_graph(k_predict_31_12_wind, y_test_wind_31_12, 'Wind power generation in kW via KNN(31.12.2015)', 'Hour')

# Gradient boosting regression PV
params = {'n_estimators': 500, 'max_depth': 6, 'min_samples_split': 2, 'learning_rate': 0.01, 'loss': 'ls'}
gbr_model_pv = GradientBoostingRegressor(**params)
gbr_model_pv.fit(X_train_pv, Y_train_pv)
gbr_predict_pv = gbr_model_pv.predict(X_test_pv)
gbr_predict_31_12_pv = gbr_model_pv.predict(x_test_pv_31_12)
continuous_graph(gbr_predict_31_12_pv, y_test_pv_31_12, 'PV power generation in kW via GBR(31.12.2015)', 'Hour')
step_graph(gbr_predict_31_12_pv, y_test_pv_31_12, 'PV power generation in kW via GBR(31.12.2015)', 'Hour')

# Gradient boosting regression Wind
gbr_model_wind = GradientBoostingRegressor(**params)
gbr_model_wind.fit(X_train_wind, Y_train_wind)
gbr_predict_wind = gbr_model_wind.predict(X_test_wind)
gbr_predict_31_12_wind = gbr_model_wind.predict(x_test_wind_31_12)
continuous_graph(gbr_predict_31_12_wind, y_test_wind_31_12, 'Wind power generation in kW via GBR(31.12.2015)', 'Hour')
step_graph(gbr_predict_31_12_wind, y_test_wind_31_12, 'Wind power generation in kW via GBR(31.12.2015)', 'Hour')

# MLP PV
mlp_pv = MLPRegressor()
mlp_pv.fit(X_train_pv, Y_train_pv)
mlp_predict_pv = mlp_pv.predict(X_test_pv)
mlp_predict_31_12_pv = mlp_pv.predict(x_test_pv_31_12)
continuous_graph(mlp_predict_31_12_pv, y_test_pv_31_12, 'PV power generation in kW via MLP(31.12.2015)', 'Hour')
step_graph(mlp_predict_31_12_pv, y_test_pv_31_12, 'PV power generation in kW via MLP(31.12.2015)', 'Hour')

# MLP Wind
mlp_wind = MLPRegressor()
mlp_wind.fit(X_train_wind, Y_train_wind)
mlp_predict_wind = mlp_wind.predict(X_test_wind)
mlp_predict_31_12_wind = mlp_wind.predict(x_test_wind_31_12)
continuous_graph(mlp_predict_31_12_wind, y_test_wind_31_12, 'Wind power generation in kW via MLP(31.12.2015)', 'Hour')
step_graph(mlp_predict_31_12_wind, y_test_wind_31_12, 'Wind power generation in kW via MLP(31.12.2015)', 'Hour')

# Ridge PV
ridge_pv = Ridge()
ridge_pv.fit(X_train_pv, Y_train_pv)
r_predict_pv = ridge_pv.predict(X_test_pv)
r_predict_31_12_pv = ridge_pv.predict(x_test_pv_31_12)
continuous_graph(r_predict_31_12_pv, y_test_pv_31_12, 'PV power generation in kW via Ridge regression(31.12.2015)',
                 'Hour')
step_graph(r_predict_31_12_pv, y_test_pv_31_12, 'PV power generation in kW via Ridge regression(31.12.2015)', 'Hour')

# Ridge Wind
ridge_wind = Ridge()
ridge_wind.fit(X_train_wind, Y_train_wind)
r_predict_wind = ridge_wind.predict(X_test_wind)
r_predict_31_12_wind = ridge_wind.predict(x_test_wind_31_12)
continuous_graph(r_predict_31_12_wind, y_test_wind_31_12, 'Wind power generation in kW via Ridge (31.12.2015)', 'Hour')
step_graph(r_predict_31_12_wind, y_test_wind_31_12, 'Wind power generation in kW via Ridge (31.12.2015)', 'Hour')


# Lasso PV
lasso_pv = Lasso(alpha=0.1)
lasso_pv.fit(X_train_pv, Y_train_pv)
lasso_predict_pv = lasso_pv.predict(X_test_pv)
lasso_predict_31_12_pv = lasso_pv.predict(x_test_pv_31_12)
continuous_graph(lasso_predict_31_12_pv, y_test_pv_31_12, 'PV power generation in kW via Lasso (31.12.2015)', 'Hour')
step_graph(lasso_predict_31_12_pv, y_test_pv_31_12, 'PV power generation in kW via Lasso (31.12.2015)', 'Hour')

# Lasso Wind
lasso_wind = Lasso(alpha=0.1)
lasso_wind.fit(X_train_wind, Y_train_wind)
lasso_predict_wind = lasso_wind.predict(X_test_wind)
lasso_predict_31_12_wind = lasso_wind.predict(x_test_wind_31_12)
continuous_graph(lasso_predict_31_12_wind, y_test_wind_31_12, 'Wind power generation in kW via Lasso (31.12.2015)',
                 'Hour')
step_graph(lasso_predict_31_12_wind, y_test_wind_31_12, 'Wind power generation in kW via Lasso (31.12.2015)', 'Hour')

# AKNN PV
plot_values_pv = prediction(generation_pv_30_12, chunks_pv)
continuous_graph(plot_values_pv, y_test_pv_31_12, 'PV power generation in kW via AKNN(31.12.2015)', 'Hour')
step_graph(plot_values_pv, y_test_pv_31_12, 'PV power generation in kW via AKNN(31.12.2015)', 'Hour')

# AKNN Wind
plot_values_wind = prediction(generation_wind_30_12, chunks_wind)
continuous_graph(plot_values_wind, y_test_wind_31_12, 'Wind power generation in kW via AKNN(31.12.2015)', 'Hour')
step_graph(plot_values_wind, y_test_wind_31_12, 'Wind power generation in kW via AKNN(31.12.2015)', 'Hour')


# Comparison PV
plt.plot(y_predict_31_12_pv, label='Predicted LR')
plt.plot(y_test_pv_31_12.values, label='Actual')
plt.plot(k_predict_31_12_pv, label='Predicted KNN')
plt.plot(gbr_predict_31_12_pv, label='Predicted GBR')
plt.plot(mlp_predict_31_12_pv, label='Predicted MLP')
plt.plot(r_predict_31_12_pv, label='Predicted Ridge')
plt.plot(lasso_predict_31_12_pv, label='Predicted Lasso')
plt.plot(plot_values_pv, label='Predicted AKNN')
plt.ylabel('Comparison PV power generation in kW on 31.12.2015')
plt.xticks([0, 5, 10, 15, 20],
           ['00:00', '05:00', '10:00', '15:00', '20:00'])
plt.xlabel('Hour')
plt.legend()
plt.show()


# Comparison PV using step graph
hour = []
for i in range(24):
    hour.append(i)
plt.step(hour, y_predict_31_12_pv, label='Predicted LR')
plt.step(hour, y_test_pv_31_12.values, label='Actual')
plt.step(hour, k_predict_31_12_pv, label='Predicted KNN')
plt.step(hour, gbr_predict_31_12_pv, label='Predicted GBR')
plt.step(hour, mlp_predict_31_12_pv, label='Predicted MLP')
plt.step(hour, r_predict_31_12_pv, label='Predicted Ridge')
plt.step(hour, lasso_predict_31_12_pv, label='Predicted Lasso')
plt.step(hour, plot_values_pv, label='Predicted AKNN')
plt.ylabel('Power (kW)')
plt.xticks([0, 5, 10, 15, 20],
           ['00:00', '05:00', '10:00', '15:00', '20:00'])
plt.xlabel('Hour')
plt.legend()
plt.show()


# Comparison Wind
plt.plot(y_predict_31_12_wind, label='Predicted LR')
plt.plot(y_test_wind_31_12.values, label='Actual')
plt.plot(k_predict_31_12_wind, label='Predicted KNN')
plt.plot(gbr_predict_31_12_wind, label='Predicted GBR')
plt.plot(mlp_predict_31_12_wind, label='Predicted MLP')
plt.plot(r_predict_31_12_wind, label='Predicted Ridge')
plt.plot(lasso_predict_31_12_wind, label='Predicted Lasso')
plt.plot(plot_values_wind, label='Predicted AKNN')
plt.ylabel('Comparison Wind power generation in kW on 31.12.2015')
plt.xticks([0, 5, 10, 15, 20],
           ['00:00', '05:00', '10:00', '15:00', '20:00'])
plt.xlabel('Hour')
plt.legend()
plt.show()

# Comparison Wind using step graph
plt.step(hour, y_predict_31_12_wind, label='Predicted LR')
plt.step(hour, y_test_wind_31_12.values, label='Actual')
plt.step(hour, k_predict_31_12_wind, label='Predicted KNN')
plt.step(hour, gbr_predict_31_12_wind, label='Predicted GBR')
plt.step(hour, mlp_predict_31_12_wind, label='Predicted MLP')
plt.step(hour, r_predict_31_12_wind, label='Predicted Ridge')
plt.step(hour, lasso_predict_31_12_wind, label='Predicted Lasso')
plt.step(hour, plot_values_wind, label='Predicted AKNN')
plt.ylabel('Power (kW)')
plt.xticks([0, 5, 10, 15, 20],
           ['00:00', '05:00', '10:00', '15:00', '20:00'])
plt.xlabel('Hour')
plt.legend()
plt.show()

mean_pv = np.mean(y_test_pv_31_12)
mse_pv_lr = mean_squared_error(y_test_pv_31_12, y_predict_31_12_pv)
rmse_pv_lr = math.sqrt(mse_pv_lr)
nrmse_pv_lr = rmse_pv_lr / mean_pv
print('Mean for PV Power generation is {}kW.'.format(mean_pv))
print('MSE for PV Power generation via Linear regression is {}'.format(mse_pv_lr))
print('RMSE for PV Power generation via Linear regression is --> {}'.format(rmse_pv_lr))
print('NRMSE for PV Power generation via Linear regression is --> {}'.format(nrmse_pv_lr), "\n")

mean_wind = np.mean(y_test_wind_31_12)
mse_wind_lr = mean_squared_error(y_test_wind_31_12, y_predict_31_12_wind)
rmse_wind_lr = math.sqrt(mse_wind_lr)
nrmse_wind_lr = rmse_wind_lr / mean_wind
print('Mean for Wind Power generation is {}kW.'.format(mean_wind))
print('MSE for Wind Power generation via Linear regression is {}'.format(mse_wind_lr))
print('RMSE for Wind Power generation via Linear regression is --> {}'.format(rmse_wind_lr))
print('NRMSE for Wind Power generation via Linear regression is --> {}'.format(nrmse_wind_lr), "\n")

mse_pv_knn = mean_squared_error(y_test_pv_31_12, k_predict_31_12_pv)
rmse_pv_knn = math.sqrt(mse_pv_knn)
nrmse_pv_knn = rmse_pv_knn / mean_pv
print('MSE for PV Power generation via KNN is {}'.format(mse_pv_knn))
print('RMSE for PV Power generation via KNN is --> {}'.format(rmse_pv_knn))
print('NRMSE for PV Power generation via KNN is --> {}'.format(nrmse_pv_knn), "\n")

mse_wind_knn = mean_squared_error(y_test_wind_31_12, k_predict_31_12_wind)
rmse_wind_knn = math.sqrt(mse_wind_knn)
nrmse_wind_knn = rmse_wind_knn / mean_wind
print('MSE for Wind Power generation via KNN is {}'.format(mse_wind_knn))
print('RMSE for Wind Power generation via KNN is --> {}'.format(rmse_wind_knn))
print('NRMSE for Wind Power generation via KNN is --> {}'.format(nrmse_wind_knn), "\n")

mse_pv_gbr = mean_squared_error(y_test_pv_31_12, gbr_predict_31_12_pv)
rmse_pv_gbr = math.sqrt(mse_pv_gbr)
nrmse_pv_gbr = rmse_pv_gbr / mean_pv
print('MSE for PV Power generation via GBR is {}'.format(mse_pv_gbr))
print('RMSE for PV Power generation via GBR is --> {}'.format(rmse_pv_gbr))
print('NRMSE for PV Power generation via GBR is --> {}'.format(nrmse_pv_gbr), "\n")

mse_wind_gbr = mean_squared_error(y_test_wind_31_12, gbr_predict_31_12_wind)
rmse_wind_gbr = math.sqrt(mse_wind_gbr)
nrmse_wind_gbr = rmse_wind_gbr / mean_wind
print('MSE for Wind Power generation via GBR is {}'.format(mse_wind_gbr))
print('RMSE for Wind Power generation via GBR is --> {}'.format(rmse_wind_gbr))
print('NRMSE for Wind Power generation via GBR is --> {}'.format(nrmse_wind_gbr), "\n")

mse_pv_mlp = mean_squared_error(y_test_pv_31_12, mlp_predict_31_12_pv)
rmse_pv_mlp = math.sqrt(mse_pv_mlp)
nrmse_pv_mlp = rmse_pv_mlp / mean_pv
print('MSE for PV Power generation via MLP is {}'.format(mse_pv_mlp))
print('RMSE for PV Power generation via MLP is --> {}'.format(rmse_pv_mlp))
print('NRMSE for PV Power generation via MLP is --> {}'.format(nrmse_pv_mlp), "\n")

mse_wind_mlp = mean_squared_error(y_test_wind_31_12, mlp_predict_31_12_wind)
rmse_wind_mlp = math.sqrt(mse_wind_mlp)
nrmse_wind_mlp = rmse_wind_mlp / mean_wind
print('MSE for Wind Power generation via MLP is {}'.format(mse_wind_mlp))
print('RMSE for Wind Power generation via MLP is --> {}'.format(rmse_wind_mlp))
print('NRMSE for Wind Power generation via MLP is --> {}'.format(nrmse_wind_mlp), "\n")

mse_pv_ridge = mean_squared_error(y_test_pv_31_12, r_predict_31_12_pv)
rmse_pv_ridge = math.sqrt(mse_pv_ridge)
nrmse_pv_ridge = rmse_pv_ridge / mean_pv
print('MSE for PV Power generation via ridge regression is {}'.format(mse_pv_ridge))
print('RMSE for PV Power generation via ridge regression is --> {}'.format(rmse_pv_ridge))
print('NRMSE for PV Power generation via ridge regression is --> {}'.format(nrmse_pv_ridge), "\n")

mse_wind_ridge = mean_squared_error(y_test_wind_31_12, r_predict_31_12_wind)
rmse_wind_ridge = math.sqrt(mse_wind_ridge)
nrmse_wind_ridge = rmse_wind_ridge / mean_wind
print('MSE for Wind Power generation via ridge regression is {}'.format(mse_wind_ridge))
print('RMSE for Wind Power generation via ridge regression is --> {}'.format(rmse_wind_ridge))
print('NRMSE for Wind Power generation via ridge regression is --> {}'.format(nrmse_wind_ridge), "\n")

mse_pv_lasso = mean_squared_error(y_test_pv_31_12, lasso_predict_31_12_pv)
rmse_pv_lasso = math.sqrt(mse_pv_lasso)
nrmse_pv_lasso = rmse_pv_lasso / mean_pv
print('MSE for PV Power generation via lasso regression is {}'.format(mse_pv_lasso))
print('RMSE for PV Power generation via lasso regression is --> {}'.format(rmse_pv_lasso))
print('NRMSE for PV Power generation via lasso regression is --> {}'.format(nrmse_pv_lasso), "\n")

mse_wind_lasso = mean_squared_error(y_test_wind_31_12, lasso_predict_31_12_wind)
rmse_wind_lasso = math.sqrt(mse_wind_lasso)
nrmse_wind_lasso = rmse_wind_lasso / mean_wind
print('MSE for Wind Power generation via lasso regression is {}'.format(mse_wind_lasso))
print('RMSE for Wind Power generation via lasso regression is --> {}'.format(rmse_wind_lasso))
print('NRMSE for Wind Power generation via lasso regression is --> {}'.format(nrmse_wind_lasso), "\n")

mse_pv_aknn = mean_squared_error(y_test_pv_31_12, plot_values_pv)
rmse_pv_aknn = math.sqrt(mse_pv_aknn)
nrmse_pv_aknn = rmse_pv_aknn / mean_pv
print('MSE for PV Power generation via AKNN is {}'.format(mse_pv_aknn))
print('RMSE for PV Power generation via AKNN is --> {}'.format(rmse_pv_aknn))
print('NRMSE for PV Power generation via AKNN is --> {}'.format(nrmse_pv_aknn), "\n")

mse_wind_aknn = mean_squared_error(y_test_wind_31_12, plot_values_wind)
rmse_wind_aknn = math.sqrt(mse_wind_aknn)
nrmse_wind_aknn = rmse_wind_aknn / mean_wind
print('MSE for Wind Power generation via AKNN is {}'.format(mse_wind_aknn))
print('RMSE for Wind Power generation via AKNN is --> {}'.format(rmse_wind_aknn))
print('NRMSE for Wind Power generation via AKNN is --> {}'.format(nrmse_wind_aknn), "\n")


# To create a csv file of forecast which can be used for optimization
date_time = ["Day"]
for a in time_last_day:
    date_time.append(a)

prediction_pv = ["PV power generation in kW"]
for a in plot_values_pv:
    prediction_pv.append(a)

# For wind Lasso gave the best prediction based on lowest NRMSE and graph comparison. Hence we are using it.
prediction_wind = ["Wind power generation in kW"]
for a in lasso_predict_31_12_wind:
    prediction_wind.append(a)

zip_time_and_forecast = zip(date_time, prediction_pv, prediction_wind)
x = tuple(zip_time_and_forecast)
with open('result_gen_pv_wind.csv', 'w') as csvFile:
    for a in x:
        writer = csv.writer(csvFile)
        writer.writerow(a)