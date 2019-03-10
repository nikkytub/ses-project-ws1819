# Copyright (c) 2019 Nikhil Singh
# GNU General Public License v3.0+ (see COPYING or https://www.gnu.org/licenses/gpl-3.0.txt)
# name: Prosumers (House, School, Zoo, Gym and Event hall) power consumption forecast
# author: Nikhil Singh (nikkytub@gmail.com)
# data-source: Karlsruhe Institute of Technology ("https://im.iism.kit.edu/sciber.php") and ISIS homework-3

import pandas as pd
import time
import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from lpi_python import lpi_distance, lpi_mean
import numpy as np
import math
import matplotlib.pyplot as plt
import csv

data = pd.read_table('SCiBER.txt')
data_house = pd.read_csv('Excercise3-data.csv', parse_dates=True)

# To predict the load in 60 minutes intervals because optimization team need it in 60 minutes intervals
# 01.01.2016 to 30.12.2016
ini_data_house = data_house[17520:35040:2]

# 30.12.2016
prev_day_data_house = data_house[34992:35040:2]

# 31.12.2016
last_day_data_house = data_house[35040::2]

test_y_house = last_day_data_house['Building 1']
load_house = np.array(prev_day_data_house['Building 1'])
y_train_house = np.array(ini_data_house['Building 1'])
chunks_house = [y_train_house[x:x + 24] for x in range(0, len(y_train_house), 24)]

times = []
for d in data['Date']:
    t = time.mktime(datetime.datetime.strptime(d, "%d.%m.%Y %H:%M").timetuple())
    times.append(str(t))
data.insert(1, 'Timestamps', times)
d = data['Timestamps'].astype(float)
data.insert(1, 'Timestamps-float', d)

# To predict the load in 60 minutes intervals because optimization team need it in 60 minutes intervals
# 01.01.2016 to 30.12.2016
ini_data = data[105119:140063:4]
# 30.12.2016
prev_day_data = data[139967:140063:4]
# 31.12.2016
last_day_data = data[140063:140159:4]
y_test_school = last_day_data['School']
y_test_zoo = last_day_data['Zoo']
y_test_gym = last_day_data['Gym']
y_test_event_hall = last_day_data['Event hall']
y_test_garden = last_day_data['Garden']

load_school = np.array(prev_day_data['School'])
load_zoo = np.array(prev_day_data['Zoo'])
load_gym = np.array(prev_day_data['Gym'])
load_event_hall = np.array(prev_day_data['Event hall'])
load_garden = np.array(prev_day_data['Garden'])

y_train_school = np.array(ini_data['School'])
y_train_zoo = np.array(ini_data['Zoo'])
y_train_gym = np.array(ini_data['Gym'])
y_train_event_hall = np.array(ini_data['Event hall'])
y_train_garden = np.array(ini_data['Garden'])

chunks_school = [y_train_school[x:x + 24] for x in range(0, len(y_train_school), 24)]
chunks_zoo = [y_train_zoo[x:x + 24] for x in range(0, len(y_train_zoo), 24)]
chunks_gym = [y_train_gym[x:x + 24] for x in range(0, len(y_train_gym), 24)]
chunks_event_hall = [y_train_event_hall[x:x + 24] for x in range(0, len(y_train_event_hall), 24)]
chunks_garden = [y_train_garden[x:x + 24] for x in range(0, len(y_train_garden), 24)]

time_last_day = np.array(last_day_data['Date'])

# Using timestamp/office load as a feature vector
x = ini_data['Office building']
X = x.values.reshape(-1, 1)

# Labels
Y_school = ini_data['School']
Y_zoo = ini_data['Zoo']
Y_gym = ini_data['Gym']
Y_hall = ini_data['Event hall']
Y_garden = ini_data['Garden']

test_x = last_day_data['Office building']
test_x = test_x.values.reshape(-1, 1)
test_y_school = last_day_data['School']
test_y_zoo = last_day_data['Zoo']
test_y_gym = last_day_data['Gym']
test_y_hall = last_day_data['Event hall']
test_y_garden = last_day_data['Garden']

X_train_house, X_test_house, Y_train_house, Y_test_house = train_test_split(X, Y_school, test_size=0.25, shuffle=False)
X_train_school, X_test_school, Y_train_school, Y_test_school = train_test_split(X, Y_school, test_size=0.25, shuffle=False)
X_train_zoo, X_test_zoo, Y_train_zoo, Y_test_zoo = train_test_split(X, Y_zoo, test_size=0.25, shuffle=False)
X_train_gym, X_test_gym, Y_train_gym, Y_test_gym = train_test_split(X, Y_gym, test_size=0.25, shuffle=False)
X_train_hall, X_test_hall, Y_train_hall, Y_test_hall = train_test_split(X, Y_hall, test_size=0.25, shuffle=False)


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


# Linear regression on Prosumer-1(House)
linear_model_house = LinearRegression()
linear_model_house.fit(X_train_house, Y_train_house)
y_predict_house = linear_model_house.predict(X_test_house)
y_predict_31_12_house = linear_model_house.predict(test_x)

# Linear regression on Prosumer-2(School)
linear_model_school = LinearRegression()
linear_model_school.fit(X_train_school, Y_train_school)
y_predict_school = linear_model_school.predict(X_test_school)
y_predict_31_12_school = linear_model_school.predict(test_x)

# Linear regression on Prosumer-3(Zoo)
linear_model_zoo = LinearRegression()
linear_model_zoo.fit(X_train_zoo, Y_train_zoo)
y_predict_zoo = linear_model_zoo.predict(X_test_zoo)
y_predict_31_12_zoo = linear_model_zoo.predict(test_x)

# Linear regression on Prosumer-4(Gym)
linear_model_gym = LinearRegression()
linear_model_gym.fit(X_train_gym, Y_train_gym)
y_predict_gym = linear_model_gym.predict(X_test_gym)
y_predict_31_12_gym = linear_model_gym.predict(test_x)

# Linear regression on Prosumer-5(Event hall)
linear_model_hall = LinearRegression()
linear_model_hall.fit(X_train_hall, Y_train_hall)
y_predict_hall = linear_model_hall.predict(X_test_hall)
y_predict_31_12_hall = linear_model_hall.predict(test_x)

# KNN on Prosumer-1(House)
knn_model_house = KNeighborsRegressor(n_neighbors=6)
knn_model_house.fit(X_train_house, Y_train_house)
k_predict_house = knn_model_house.predict(X_test_house)
k_predict_31_12_house = knn_model_house.predict(test_x)

# KNN on Prosumer-2(School)
knn_model_school = KNeighborsRegressor(n_neighbors=6)
knn_model_school.fit(X_train_school, Y_train_school)
k_predict_school = knn_model_school.predict(X_test_school)
k_predict_31_12_school = knn_model_school.predict(test_x)

# KNN on Prosumer-3(Zoo)
knn_model_zoo = KNeighborsRegressor(n_neighbors=6)
knn_model_zoo.fit(X_train_zoo, Y_train_zoo)
k_predict_zoo = knn_model_zoo.predict(X_test_zoo)
k_predict_31_12_zoo = knn_model_zoo.predict(test_x)

# KNN on Prosumer-4(Gym)
knn_model_gym = KNeighborsRegressor(n_neighbors=6)
knn_model_gym.fit(X_train_gym, Y_train_gym)
k_predict_gym = knn_model_gym.predict(X_test_gym)
k_predict_31_12_gym = knn_model_gym.predict(test_x)

# KNN on Prosumer-5(Event Hall)
knn_model_hall = KNeighborsRegressor(n_neighbors=6)
knn_model_hall.fit(X_train_hall, Y_train_hall)
k_predict_hall = knn_model_hall.predict(X_test_hall)
k_predict_31_12_hall = knn_model_hall.predict(test_x)

# Gradient boosting regression on Prosumer-1(House)
params = {'n_estimators': 500, 'max_depth': 6, 'min_samples_split': 2, 'learning_rate': 0.01, 'loss': 'ls'}
gbr_model_house = GradientBoostingRegressor(**params)
gbr_model_house.fit(X_train_house, Y_train_house)
gbr_predict_house = gbr_model_house.predict(X_test_house)
gbr_predict_31_12_house = gbr_model_house.predict(test_x)

# Gradient boosting regression on Prosumer-2(School)
gbr_model_school = GradientBoostingRegressor(**params)
gbr_model_school.fit(X_train_school, Y_train_school)
gbr_predict_school = gbr_model_school.predict(X_test_school)
gbr_predict_31_12_school = gbr_model_school.predict(test_x)

# Gradient boosting regression on Prosumer-3(Zoo)
gbr_model_zoo = GradientBoostingRegressor(**params)
gbr_model_zoo.fit(X_train_zoo, Y_train_zoo)
gbr_predict_zoo = gbr_model_zoo.predict(X_test_zoo)
gbr_predict_31_12_zoo = gbr_model_zoo.predict(test_x)

# Gradient boosting regression on Prosumer-4(Gym)
gbr_model_gym = GradientBoostingRegressor(**params)
gbr_model_gym.fit(X_train_gym, Y_train_gym)
gbr_predict_gym = gbr_model_gym.predict(X_test_gym)
gbr_predict_31_12_gym = gbr_model_gym.predict(test_x)

# Gradient boosting regression on Prosumer-5(Event hall)
gbr_model_hall = GradientBoostingRegressor(**params)
gbr_model_hall.fit(X_train_hall, Y_train_hall)
gbr_predict_hall = gbr_model_hall.predict(X_test_hall)
gbr_predict_31_12_hall = gbr_model_hall.predict(test_x)

# ANN on Prosumer-1(House)
mlp_house = MLPRegressor()
mlp_house.fit(X_train_house, Y_train_house)
mlp_predict_house = mlp_house.predict(X_test_house)
mlp_predict_31_12_house = mlp_house.predict(test_x)

# ANN on Prosumer-2(School)
mlp_school = MLPRegressor()
mlp_school.fit(X_train_school, Y_train_school)
mlp_predict_school = mlp_school.predict(X_test_school)
mlp_predict_31_12_school = mlp_school.predict(test_x)

# ANN on Prosumer-3(Zoo)
mlp_zoo = MLPRegressor()
mlp_zoo.fit(X_train_zoo, Y_train_zoo)
mlp_predict_zoo = mlp_zoo.predict(X_test_zoo)
mlp_predict_31_12_zoo = mlp_zoo.predict(test_x)

# ANN on Prosumer-4(Gym)
mlp_gym = MLPRegressor()
mlp_gym.fit(X_train_gym, Y_train_gym)
mlp_predict_gym = mlp_gym.predict(X_test_gym)
mlp_predict_31_12_gym = mlp_gym.predict(test_x)

# ANN on Prosumer-5(Event Hall)
mlp_hall = MLPRegressor()
mlp_hall.fit(X_train_hall, Y_train_hall)
mlp_predict_hall = mlp_hall.predict(X_test_hall)
mlp_predict_31_12_hall = mlp_hall.predict(test_x)

# Ridge regression on Prosumer-1(House)
ridge_house = Ridge()
ridge_house.fit(X_train_house, Y_train_house)
r_predict_house = ridge_house.predict(X_test_house)
r_predict_31_12_house = ridge_house.predict(test_x)

# Ridge regression on Prosumer-2(School)
ridge_school = Ridge()
ridge_school.fit(X_train_school, Y_train_school)
r_predict_school = ridge_school.predict(X_test_school)
r_predict_31_12_school = ridge_school.predict(test_x)

# Ridge regression on Prosumer-3(Zoo)
ridge_zoo = Ridge()
ridge_zoo.fit(X_train_zoo, Y_train_zoo)
r_predict_zoo = ridge_zoo.predict(X_test_zoo)
r_predict_31_12_zoo = ridge_zoo.predict(test_x)

# Ridge regression on Prosumer-4(Gym)
ridge_gym = Ridge()
ridge_gym.fit(X_train_gym, Y_train_gym)
r_predict_gym = ridge_gym.predict(X_test_gym)
r_predict_31_12_gym = ridge_gym.predict(test_x)

# Ridge regression on Prosumer-5(Event hall)
ridge_hall = Ridge()
ridge_hall.fit(X_train_hall, Y_train_hall)
r_predict_hall = ridge_hall.predict(X_test_hall)
r_predict_31_12_hall = ridge_hall.predict(test_x)

# Lasso regression on Prosumer-1(House)
lasso_house = Lasso(alpha=0.1)
lasso_house.fit(X_train_house, Y_train_house)
lasso_predict_house = lasso_house.predict(X_test_house)
lasso_predict_31_12_house = lasso_house.predict(test_x)

# Lasso regression on Prosumer-2(School)
lasso_school = Lasso(alpha=0.1)
lasso_school.fit(X_train_school, Y_train_school)
lasso_predict_school = lasso_school.predict(X_test_school)
lasso_predict_31_12_school = lasso_school.predict(test_x)

# Lasso regression on Prosumer-3(Zoo)
lasso_zoo = Lasso(alpha=0.1)
lasso_zoo.fit(X_train_zoo, Y_train_zoo)
lasso_predict_zoo = lasso_zoo.predict(X_test_zoo)
lasso_predict_31_12_zoo = lasso_zoo.predict(test_x)

# Lasso regression on Prosumer-4(Gym)
lasso_gym = Lasso(alpha=0.1)
lasso_gym.fit(X_train_gym, Y_train_gym)
lasso_predict_gym = lasso_gym.predict(X_test_gym)
lasso_predict_31_12_gym = lasso_gym.predict(test_x)

# Lasso regression on Prosumer-5(Event hall)
lasso_hall = Lasso(alpha=0.1)
lasso_hall.fit(X_train_hall, Y_train_hall)
lasso_predict_hall = lasso_hall.predict(X_test_hall)
lasso_predict_31_12_hall = lasso_hall.predict(test_x)

# AKNN on Prosumer-1(House)
plot_values_house = prediction(load_house, chunks_house)
continuous_graph(plot_values_house, test_y_house, 'Prosumer-1 power consumption in kW via AKNN(31.12.2016)', 'Time')
step_graph(plot_values_house, test_y_house, 'Prosumer-1 power consumption in kW via AKNN(31.12.2016)', 'Time')

# AKNN on Prosumer-2(School)
plot_values_school = prediction(load_school, chunks_school)
continuous_graph(plot_values_school, test_y_school, 'Prosumer-2 power consumption in kW via AKNN(31.12.2016)', 'Time')
step_graph(plot_values_school, test_y_school, 'Prosumer-2 power consumption in kW via AKNN(31.12.2016)', 'Time')

# AKNN on Prosumer-3(Zoo)
plot_values_zoo = prediction(load_zoo, chunks_zoo)
continuous_graph(plot_values_zoo, test_y_zoo, 'Prosumer-3 power consumption in kW via AKNN(31.12.2016)', 'Time')
step_graph(plot_values_zoo, test_y_zoo, 'Prosumer-3 power consumption in kW via AKNN(31.12.2016)', 'Time')

# AKNN on Prosumer-4(Gym)
plot_values_gym = prediction(load_gym, chunks_gym)
continuous_graph(plot_values_gym, test_y_gym, 'Prosumer-4 power consumption in kW via AKNN(31.12.2016)', 'Time')
step_graph(plot_values_gym, test_y_gym, 'Prosumer-4 power consumption in kW via AKNN(31.12.2016)', 'Time')

# AKNN on Prosumer-5(Event hall)
plot_values_hall = prediction(load_event_hall, chunks_event_hall)
continuous_graph(plot_values_hall, test_y_hall, 'Prosumer-5 power consumption in kW via AKNN(31.12.2016)', 'Time')
step_graph(plot_values_hall, test_y_hall, 'Prosumer-5 power consumption in kW via AKNN(31.12.2016)', 'Time')


# Comparison Prosumers using step graph via AKNN.
hour = []
for i in range(24):
    hour.append(i)
plt.step(hour, plot_values_house, label='P. P-1')
plt.step(hour, test_y_house.values, label='A. P-1')
plt.step(hour, plot_values_school, label='P. P-2')
plt.step(hour, test_y_school.values, label='A. P-2')
plt.step(hour, plot_values_zoo, label='P. P-3')
plt.step(hour, test_y_zoo.values, label='A. P-3')
plt.step(hour, plot_values_gym, label='P. P-4')
plt.step(hour, test_y_gym.values, label='A. P-4')
plt.step(hour, plot_values_hall, label='P. P-5')
plt.step(hour, test_y_hall.values, label='A. P-5')
plt.ylabel('Power (kW)')
plt.xticks([0, 5, 10, 15, 20],
           ['00:00', '05:00', '10:00', '15:00', '20:00'])
plt.xlabel('Hour')
plt.legend()
plt.show()

# Comparison between different algorithms on Prosumer-1(House) using step graph
plt.step(hour, y_predict_31_12_house, label='Predicted LR')
plt.step(hour, test_y_house.values, label='Actual')
plt.step(hour, k_predict_31_12_house, label='Predicted KNN')
plt.step(hour, gbr_predict_31_12_house, label='Predicted GBR')
plt.step(hour, mlp_predict_31_12_house, label='Predicted ANN')
plt.step(hour, r_predict_31_12_house, label='Predicted Ridge')
plt.step(hour, lasso_predict_31_12_house, label='Predicted Lasso')
plt.step(hour, plot_values_house, label='Predicted AKNN')
plt.ylabel('Comparison Prosumer-1 load in kW on 31.12.2016')
plt.xticks([0, 5, 10, 15, 20],
           ['00:00', '05:00', '10:00', '15:00', '20:00'])
plt.xlabel('Time')
plt.legend()
plt.show()

# Comparison between different algorithms on Prosumer-2(School) using step graph
plt.step(hour, y_predict_31_12_school, label='Predicted LR')
plt.step(hour, test_y_school.values, label='Actual')
plt.step(hour, k_predict_31_12_school, label='Predicted KNN')
plt.step(hour, gbr_predict_31_12_school, label='Predicted GBR')
plt.step(hour, mlp_predict_31_12_school, label='Predicted ANN')
plt.step(hour, r_predict_31_12_school, label='Predicted Ridge')
plt.step(hour, lasso_predict_31_12_school, label='Predicted Lasso')
plt.step(hour, plot_values_school, label='Predicted AKNN')
plt.ylabel('Comparison Prosumer-2 load in kW on 31.12.2016')
plt.xticks([0, 5, 10, 15, 20],
           ['00:00', '05:00', '10:00', '15:00', '20:00'])
plt.xlabel('Time')
plt.legend()
plt.show()

# Comparison between different algorithms on Prosumer-3(Zoo) using step graph
plt.step(hour, y_predict_31_12_zoo, label='Predicted LR')
plt.step(hour, test_y_zoo.values, label='Actual')
plt.step(hour, k_predict_31_12_zoo, label='Predicted KNN')
plt.step(hour, gbr_predict_31_12_zoo, label='Predicted GBR')
plt.step(hour, mlp_predict_31_12_zoo, label='Predicted ANN')
plt.step(hour, r_predict_31_12_zoo, label='Predicted Ridge')
plt.step(hour, lasso_predict_31_12_zoo, label='Predicted Lasso')
plt.step(hour, plot_values_zoo, label='Predicted AKNN')
plt.ylabel('Comparison Prosumer-3 load in kW on 31.12.2016')
plt.xticks([0, 5, 10, 15, 20],
           ['00:00', '05:00', '10:00', '15:00', '20:00'])
plt.xlabel('Time')
plt.legend()
plt.show()

# Comparison between different algorithms on Prosumer-4(Gym) using step graph
plt.step(hour, y_predict_31_12_gym, label='Predicted LR')
plt.step(hour, test_y_gym.values, label='Actual')
plt.step(hour, k_predict_31_12_gym, label='Predicted KNN')
plt.step(hour, gbr_predict_31_12_gym, label='Predicted GBR')
plt.step(hour, mlp_predict_31_12_gym, label='Predicted ANN')
plt.step(hour, r_predict_31_12_gym, label='Predicted Ridge')
plt.step(hour, lasso_predict_31_12_gym, label='Predicted Lasso')
plt.step(hour, plot_values_gym, label='Predicted AKNN')
plt.ylabel('Comparison Prosumer-4 load in kW on 31.12.2016')
plt.xticks([0, 5, 10, 15, 20],
           ['00:00', '05:00', '10:00', '15:00', '20:00'])
plt.xlabel('Time')
plt.legend()
plt.show()

# Comparison between different algorithms on Prosumer-5(Event hall) using step graph
plt.step(hour, y_predict_31_12_hall, label='Predicted LR')
plt.step(hour, test_y_hall.values, label='Actual')
plt.step(hour, k_predict_31_12_hall, label='Predicted KNN')
plt.step(hour, gbr_predict_31_12_hall, label='Predicted GBR')
plt.step(hour, mlp_predict_31_12_hall, label='Predicted ANN')
plt.step(hour, r_predict_31_12_hall, label='Predicted Ridge')
plt.step(hour, lasso_predict_31_12_hall, label='Predicted Lasso')
plt.step(hour, plot_values_hall, label='Predicted AKNN')
plt.ylabel('Comparison Prosumer-5 load in kW on 31.12.2016')
plt.xticks([0, 5, 10, 15, 20],
           ['00:00', '05:00', '10:00', '15:00', '20:00'])
plt.xlabel('Time')
plt.legend()
plt.show()

# AKNN Pr-1
mean_house = np.mean(test_y_house)
mse_house_aknn = mean_squared_error(test_y_house, plot_values_house)
rmse_house_aknn = math.sqrt(mse_house_aknn)
nrmse_house_aknn = rmse_house_aknn / mean_house
print('Mean for Prosumer-1 power consumption is {}kW'.format(mean_house))
print('MSE for Prosumer-1 power consumption is {}'.format(mse_house_aknn))
print('RMSE for Prosumer-1 power consumption is --> {}'.format(rmse_house_aknn))
print('NRMSE for Prosumer-1 power consumption via AKNN is --> {}'.format(nrmse_house_aknn))

# Linear regression Pr-1
mse_house_lr = mean_squared_error(test_y_house, y_predict_31_12_house)
rmse_house_lr = math.sqrt(mse_house_lr)
nrmse_house_lr = rmse_house_lr / mean_house
print('MSE for Prosumer-1 power consumption is {}'.format(mse_house_lr))
print('RMSE for Prosumer-1 power consumption is --> {}'.format(rmse_house_lr))
print('NRMSE for Prosumer-1 power consumption via LR is --> {}'.format(nrmse_house_lr))

# KNN Pr-1
mse_house_knn = mean_squared_error(test_y_house, k_predict_31_12_house)
rmse_house_knn = math.sqrt(mse_house_knn)
nrmse_house_knn = rmse_house_knn / mean_house
print('MSE for Prosumer-1 power consumption is {}'.format(mse_house_knn))
print('RMSE for Prosumer-1 power consumption is --> {}'.format(rmse_house_knn))
print('NRMSE for Prosumer-1 power consumption via KNN is --> {}'.format(nrmse_house_knn))

# GBR Pr-1
mse_house_gbr = mean_squared_error(test_y_house, gbr_predict_31_12_house)
rmse_house_gbr = math.sqrt(mse_house_gbr)
nrmse_house_gbr = rmse_house_gbr / mean_house
print('MSE for Prosumer-1 power consumption is {}'.format(mse_house_gbr))
print('RMSE for Prosumer-1 power consumption is --> {}'.format(rmse_house_gbr))
print('NRMSE for Prosumer-1 power consumption via GBR is --> {}'.format(nrmse_house_gbr))

# ANN Pr-1
mse_house_ann = mean_squared_error(test_y_house, mlp_predict_31_12_house)
rmse_house_ann = math.sqrt(mse_house_ann)
nrmse_house_ann = rmse_house_ann / mean_house
print('MSE for Prosumer-1 power consumption is {}'.format(mse_house_ann))
print('RMSE for Prosumer-1 power consumption is --> {}'.format(rmse_house_ann))
print('NRMSE for Prosumer-1 power consumption via ANN is --> {}'.format(nrmse_house_ann))

# Ridge Pr-1
mse_house_r = mean_squared_error(test_y_house, r_predict_31_12_house)
rmse_house_r = math.sqrt(mse_house_r)
nrmse_house_r = rmse_house_r / mean_house
print('MSE for Prosumer-1 power consumption is {}'.format(mse_house_r))
print('RMSE for Prosumer-1 power consumption is --> {}'.format(rmse_house_r))
print('NRMSE for Prosumer-1 power consumption via Ridge is --> {}'.format(nrmse_house_r))

# Lasso Pr-1
mse_house_lasso = mean_squared_error(test_y_house, lasso_predict_31_12_house)
rmse_house_lasso = math.sqrt(mse_house_lasso)
nrmse_house_lasso = rmse_house_lasso / mean_house
print('MSE for Prosumer-1 power consumption is {}'.format(mse_house_lasso))
print('RMSE for Prosumer-1 power consumption is --> {}'.format(rmse_house_lasso))
print('NRMSE for Prosumer-1 power consumption via Lasso is --> {}'.format(nrmse_house_lasso))


# AKNN Pr-2
mean_school = np.mean(test_y_school)
mse_school_aknn = mean_squared_error(test_y_school, plot_values_school)
rmse_school_aknn = math.sqrt(mse_school_aknn)
nrmse_school_aknn = rmse_school_aknn / mean_school
print('Mean for Prosumer-2 power consumption is {}kW'.format(mean_school))
print('MSE for Prosumer-2 power consumption is {}'.format(mse_school_aknn))
print('RMSE for Prosumer-2 power consumption is --> {}'.format(rmse_school_aknn))
print('NRMSE for Prosumer-2 power consumption via AKNN is --> {}'.format(nrmse_school_aknn))

# LR Pr-2
mse_school_lr = mean_squared_error(test_y_school, y_predict_31_12_school)
rmse_school_lr = math.sqrt(mse_school_lr)
nrmse_school_lr = rmse_school_lr / mean_school
print('MSE for Prosumer-2 power consumption is {}'.format(mse_school_lr))
print('RMSE for Prosumer-2 power consumption is --> {}'.format(rmse_school_lr))
print('NRMSE for Prosumer-2 power consumption via LR is --> {}'.format(nrmse_school_lr))

# KNN Pr-2
mse_school_knn = mean_squared_error(test_y_school, k_predict_31_12_school)
rmse_school_knn = math.sqrt(mse_school_knn)
nrmse_school_knn = rmse_school_knn / mean_school
print('MSE for Prosumer-2 power consumption is {}'.format(mse_school_knn))
print('RMSE for Prosumer-2 power consumption is --> {}'.format(rmse_school_knn))
print('NRMSE for Prosumer-2 power consumption via KNN is --> {}'.format(nrmse_school_knn))

# GBR Pr-2
mse_school_gbr = mean_squared_error(test_y_school, gbr_predict_31_12_school)
rmse_school_gbr = math.sqrt(mse_school_gbr)
nrmse_school_gbr = rmse_school_gbr / mean_school
print('MSE for Prosumer-2 power consumption is {}'.format(mse_school_gbr))
print('RMSE for Prosumer-2 power consumption is --> {}'.format(rmse_school_gbr))
print('NRMSE for Prosumer-2 power consumption via GBR is --> {}'.format(nrmse_school_gbr))

# ANN Pr-2
mse_school_ann = mean_squared_error(test_y_school, mlp_predict_31_12_school)
rmse_school_ann = math.sqrt(mse_school_ann)
nrmse_school_ann = rmse_school_ann / mean_school
print('MSE for Prosumer-2 power consumption is {}'.format(mse_school_ann))
print('RMSE for Prosumer-2 power consumption is --> {}'.format(rmse_school_ann))
print('NRMSE for Prosumer-2 power consumption via ANN is --> {}'.format(nrmse_school_ann))

# Ridge Pr-2
mse_school_r = mean_squared_error(test_y_school, r_predict_31_12_school)
rmse_school_r = math.sqrt(mse_school_r)
nrmse_school_r = rmse_school_r / mean_school
print('MSE for Prosumer-2 power consumption is {}'.format(mse_school_r))
print('RMSE for Prosumer-2 power consumption is --> {}'.format(rmse_school_r))
print('NRMSE for Prosumer-2 power consumption via Ridge is --> {}'.format(nrmse_school_r))

# Lasso Pr-2
mse_school_lasso = mean_squared_error(test_y_school, lasso_predict_31_12_school)
rmse_school_lasso = math.sqrt(mse_school_lasso)
nrmse_school_lasso = rmse_school_lasso / mean_school
print('MSE for Prosumer-2 power consumption is {}'.format(mse_school_lasso))
print('RMSE for Prosumer-2 power consumption is --> {}'.format(rmse_school_lasso))
print('NRMSE for Prosumer-2 power consumption via Lasso is --> {}'.format(nrmse_school_lasso))


# AKNN Pr-3
mean_zoo = np.mean(test_y_zoo)
mse_zoo_aknn = mean_squared_error(test_y_zoo, plot_values_zoo)
rmse_zoo_aknn = math.sqrt(mse_zoo_aknn)
nrmse_zoo_aknn = rmse_zoo_aknn / mean_zoo
print('Mean for Prosumer-3 power consumption is {}kW'.format(mean_zoo))
print('MSE for Prosumer-3 power consumption is {}'.format(mse_zoo_aknn))
print('RMSE for Prosumer-3 power consumption is --> {}'.format(rmse_zoo_aknn))
print('NRMSE for Prosumer-3 power consumption via AKNN is --> {}'.format(nrmse_zoo_aknn))

# LR Pr-3
mse_zoo_lr = mean_squared_error(test_y_zoo, y_predict_31_12_zoo)
rmse_zoo_lr = math.sqrt(mse_zoo_lr)
nrmse_zoo_lr = rmse_zoo_lr / mean_zoo
print('MSE for Prosumer-3 power consumption is {}'.format(mse_zoo_lr))
print('RMSE for Prosumer-3 power consumption is --> {}'.format(rmse_zoo_lr))
print('NRMSE for Prosumer-3 power consumption via LR is --> {}'.format(nrmse_zoo_lr))

# KNN Pr-3
mse_zoo_knn = mean_squared_error(test_y_zoo, k_predict_31_12_zoo)
rmse_zoo_knn = math.sqrt(mse_zoo_knn)
nrmse_zoo_knn = rmse_zoo_knn / mean_zoo
print('MSE for Prosumer-3 power consumption is {}'.format(mse_zoo_knn))
print('RMSE for Prosumer-3 power consumption is --> {}'.format(rmse_zoo_knn))
print('NRMSE for Prosumer-3 power consumption via KNN is --> {}'.format(nrmse_zoo_knn))

# GBR Pr-3
mse_zoo_gbr = mean_squared_error(test_y_zoo, gbr_predict_31_12_zoo)
rmse_zoo_gbr = math.sqrt(mse_zoo_gbr)
nrmse_zoo_gbr = rmse_zoo_gbr / mean_zoo
print('MSE for Prosumer-3 power consumption is {}'.format(mse_zoo_gbr))
print('RMSE for Prosumer-3 power consumption is --> {}'.format(rmse_zoo_gbr))
print('NRMSE for Prosumer-3 power consumption via GBR is --> {}'.format(nrmse_zoo_gbr))

# ANN Pr-3
mse_zoo_ann = mean_squared_error(test_y_zoo, mlp_predict_31_12_zoo)
rmse_zoo_ann = math.sqrt(mse_zoo_ann)
nrmse_zoo_ann = rmse_zoo_ann / mean_zoo
print('MSE for Prosumer-3 power consumption is {}'.format(mse_zoo_ann))
print('RMSE for Prosumer-3 power consumption is --> {}'.format(rmse_zoo_ann))
print('NRMSE for Prosumer-3 power consumption via ANN is --> {}'.format(nrmse_zoo_ann))

# Ridge Pr-3
mse_zoo_r = mean_squared_error(test_y_zoo, r_predict_31_12_zoo)
rmse_zoo_r = math.sqrt(mse_zoo_r)
nrmse_zoo_r = rmse_zoo_r / mean_zoo
print('MSE for Prosumer-3 power consumption is {}'.format(mse_zoo_r))
print('RMSE for Prosumer-3 power consumption is --> {}'.format(rmse_zoo_r))
print('NRMSE for Prosumer-3 power consumption via Ridge is --> {}'.format(nrmse_zoo_r))

# Lasso Pr-3
mse_zoo_lasso = mean_squared_error(test_y_zoo, lasso_predict_31_12_zoo)
rmse_zoo_lasso = math.sqrt(mse_zoo_lasso)
nrmse_zoo_lasso = rmse_zoo_lasso / mean_zoo
print('MSE for Prosumer-3 power consumption is {}'.format(mse_zoo_lasso))
print('RMSE for Prosumer-3 power consumption is --> {}'.format(rmse_zoo_lasso))
print('NRMSE for Prosumer-3 power consumption via Lasso is --> {}'.format(nrmse_zoo_lasso))


# AKNN Pr-4
mean_gym = np.mean(test_y_gym)
mse_gym_aknn = mean_squared_error(test_y_gym, plot_values_gym)
rmse_gym_aknn = math.sqrt(mse_gym_aknn)
nrmse_gym_aknn = rmse_gym_aknn / mean_gym
print('Mean for Prosumer-4 power consumption is {}kW'.format(mean_gym))
print('MSE for Prosumer-4 power consumption is {}'.format(mse_gym_aknn))
print('RMSE for Prosumer-4 power consumption is --> {}'.format(rmse_gym_aknn))
print('NRMSE for Prosumer-4 power consumption via AKNN is --> {}'.format(nrmse_gym_aknn))

# LR Pr-4
mse_gym_lr = mean_squared_error(test_y_gym, y_predict_31_12_gym)
rmse_gym_lr = math.sqrt(mse_gym_lr)
nrmse_gym_lr = rmse_gym_lr / mean_gym
print('MSE for Prosumer-4 power consumption is {}'.format(mse_gym_lr))
print('RMSE for Prosumer-4 power consumption is --> {}'.format(rmse_gym_lr))
print('NRMSE for Prosumer-4 power consumption via LR is --> {}'.format(nrmse_gym_lr))

# KNN Pr-4
mse_gym_knn = mean_squared_error(test_y_gym, k_predict_31_12_gym)
rmse_gym_knn = math.sqrt(mse_gym_knn)
nrmse_gym_knn = rmse_gym_knn / mean_gym
print('MSE for Prosumer-4 power consumption is {}'.format(mse_gym_knn))
print('RMSE for Prosumer-4 power consumption is --> {}'.format(rmse_gym_knn))
print('NRMSE for Prosumer-4 power consumption via KNN is --> {}'.format(nrmse_gym_knn))

# GBR Pr-4
mse_gym_gbr = mean_squared_error(test_y_gym, gbr_predict_31_12_gym)
rmse_gym_gbr = math.sqrt(mse_gym_gbr)
nrmse_gym_gbr = rmse_gym_gbr / mean_gym
print('MSE for Prosumer-4 power consumption is {}'.format(mse_gym_gbr))
print('RMSE for Prosumer-4 power consumption is --> {}'.format(rmse_gym_gbr))
print('NRMSE for Prosumer-4 power consumption via GBR is --> {}'.format(nrmse_gym_gbr))

# ANN Pr-4
mse_gym_ann = mean_squared_error(test_y_gym, mlp_predict_31_12_gym)
rmse_gym_ann = math.sqrt(mse_gym_ann)
nrmse_gym_ann = rmse_gym_ann / mean_gym
print('MSE for Prosumer-4 power consumption is {}'.format(mse_gym_ann))
print('RMSE for Prosumer-4 power consumption is --> {}'.format(rmse_gym_ann))
print('NRMSE for Prosumer-4 power consumption via ANN is --> {}'.format(nrmse_gym_ann))

# Ridge Pr-4
mse_gym_r = mean_squared_error(test_y_gym, r_predict_31_12_gym)
rmse_gym_r = math.sqrt(mse_gym_r)
nrmse_gym_r = rmse_gym_r / mean_gym
print('MSE for Prosumer-4 power consumption is {}'.format(mse_gym_r))
print('RMSE for Prosumer-4 power consumption is --> {}'.format(rmse_gym_r))
print('NRMSE for Prosumer-4 power consumption via Ridge is --> {}'.format(nrmse_gym_r))

# Lasso Pr-4
mse_gym_lasso = mean_squared_error(test_y_gym, lasso_predict_31_12_gym)
rmse_gym_lasso = math.sqrt(mse_gym_lasso)
nrmse_gym_lasso = rmse_gym_lasso / mean_gym
print('MSE for Prosumer-4 power consumption is {}'.format(mse_gym_lasso))
print('RMSE for Prosumer-4 power consumption is --> {}'.format(rmse_gym_lasso))
print('NRMSE for Prosumer-4 power consumption via Lasso is --> {}'.format(nrmse_gym_lasso))


# AKNN Pr-5
mean_event_hall = np.mean(test_y_hall)
mse_event_hall_aknn = mean_squared_error(test_y_hall, plot_values_hall)
rmse_event_hall_aknn = math.sqrt(mse_event_hall_aknn)
nrmse_event_hall_aknn = rmse_event_hall_aknn / mean_event_hall
print('Mean for Prosumer-5 power consumption is {}kW'.format(mean_event_hall))
print('MSE for Prosumer-5 power consumption is {}'.format(mse_event_hall_aknn))
print('RMSE for Prosumer-5 power consumption is --> {}'.format(rmse_event_hall_aknn))
print('NRMSE for Prosumer-5 power consumption via AKNN is --> {}'.format(nrmse_event_hall_aknn))

# LR Pr-5
mse_event_hall_lr = mean_squared_error(test_y_hall, y_predict_31_12_hall)
rmse_event_hall_lr = math.sqrt(mse_event_hall_lr)
nrmse_event_hall_lr = rmse_event_hall_lr / mean_event_hall
print('MSE for Prosumer-5 power consumption is {}'.format(mse_event_hall_lr))
print('RMSE for Prosumer-5 power consumption is --> {}'.format(rmse_event_hall_lr))
print('NRMSE for Prosumer-5 power consumption via LR is --> {}'.format(nrmse_event_hall_lr))

# KNN Pr-5
mse_event_hall_knn = mean_squared_error(test_y_hall, k_predict_31_12_hall)
rmse_event_hall_knn = math.sqrt(mse_event_hall_knn)
nrmse_event_hall_knn = rmse_event_hall_knn / mean_event_hall
print('MSE for Prosumer-5 power consumption is {}'.format(mse_event_hall_knn))
print('RMSE for Prosumer-5 power consumption is --> {}'.format(rmse_event_hall_knn))
print('NRMSE for Prosumer-5 power consumption via KNN is --> {}'.format(nrmse_event_hall_knn))

# GBR Pr-5
mse_event_hall_gbr = mean_squared_error(test_y_hall, gbr_predict_31_12_hall)
rmse_event_hall_gbr = math.sqrt(mse_event_hall_gbr)
nrmse_event_hall_gbr = rmse_event_hall_gbr / mean_event_hall
print('MSE for Prosumer-5 power consumption is {}'.format(mse_event_hall_gbr))
print('RMSE for Prosumer-5 power consumption is --> {}'.format(rmse_event_hall_gbr))
print('NRMSE for Prosumer-5 power consumption via GBR is --> {}'.format(nrmse_event_hall_gbr))

# ANN Pr-5
mse_event_hall_ann = mean_squared_error(test_y_hall, mlp_predict_31_12_hall)
rmse_event_hall_ann = math.sqrt(mse_event_hall_ann)
nrmse_event_hall_ann = rmse_event_hall_ann / mean_event_hall
print('MSE for Prosumer-5 power consumption is {}'.format(mse_event_hall_ann))
print('RMSE for Prosumer-5 power consumption is --> {}'.format(rmse_event_hall_ann))
print('NRMSE for Prosumer-5 power consumption via ANN is --> {}'.format(nrmse_event_hall_ann))

# Ridge Pr-5
mse_event_hall_r = mean_squared_error(test_y_hall, r_predict_31_12_hall)
rmse_event_hall_r = math.sqrt(mse_event_hall_r)
nrmse_event_hall_r = rmse_event_hall_r / mean_event_hall
print('MSE for Prosumer-5 power consumption is {}'.format(mse_event_hall_r))
print('RMSE for Prosumer-5 power consumption is --> {}'.format(rmse_event_hall_r))
print('NRMSE for Prosumer-5 power consumption via Ridge is --> {}'.format(nrmse_event_hall_r))

# Lasso Pr-5
mse_event_hall_lasso = mean_squared_error(test_y_hall, lasso_predict_31_12_hall)
rmse_event_hall_lasso = math.sqrt(mse_event_hall_lasso)
nrmse_event_hall_lasso = rmse_event_hall_lasso / mean_event_hall
print('MSE for Prosumer-5 power consumption is {}'.format(mse_event_hall_lasso))
print('RMSE for Prosumer-5 power consumption is --> {}'.format(rmse_event_hall_lasso))
print('NRMSE for Prosumer-5 power consumption via Lasso is --> {}'.format(nrmse_event_hall_lasso))


# To create a csv file of forecast using AKNN which can be used for optimization
time_list = ["Date"]
for a in time_last_day:
    time_list.append(a)

prediction_house = ["Prosumer-1 power consumption in kW"]
for a in plot_values_house:
    prediction_house.append(a)

prediction_school = ["Prosumer-2 power consumption in kW"]
for a in plot_values_school:
    prediction_school.append(a)

prediction_zoo = ["Prosumer-3 power consumption in kW"]
for a in plot_values_zoo:
    prediction_zoo.append(a)

prediction_gym = ["Prosumer-4 power consumption in kW"]
for a in plot_values_gym:
    prediction_gym.append(a)

prediction_event_hall = ["Prosumer-5 power consumption in kW"]
for a in plot_values_hall:
    prediction_event_hall.append(a)

zip_time_and_forecast = zip(time_list, prediction_house, prediction_school, prediction_zoo, prediction_gym,
                            prediction_event_hall)
x = tuple(zip_time_and_forecast)
with open('result_prosumers.csv', 'w') as csvFile:
    for a in x:
        writer = csv.writer(csvFile)
        writer.writerow(a)
