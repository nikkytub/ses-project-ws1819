# Copyright (c) 2019 Nikhil Singh
# GNU General Public License v3.0+ (see COPYING or https://www.gnu.org/licenses/gpl-3.0.txt)
# name: House, School, Zoo, Gym, Event hall and Garden power consumption forecast via Adjusted K nearest neighbor
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

# 01.01.2016 to 30.12.2016
ini_data_house = data_house[17520:35040:2]

# 30.12.2016
prev_day_data_house = data_house[34992:35040:2]

# 31.12.2016
last_day_data_house = data_house[35040::2]

test_y_house = last_day_data_house['Building 1']
load_house = prev_day_data_house['Building 1'].tolist()
y_train_house = ini_data_house['Building 1'].tolist()
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

load_school = prev_day_data['School'].tolist()
load_zoo = prev_day_data['Zoo'].tolist()
load_gym = prev_day_data['Gym'].tolist()
load_event_hall = prev_day_data['Event hall'].tolist()
load_garden = prev_day_data['Garden'].tolist()

y_train_school = ini_data['School'].tolist()
y_train_zoo = ini_data['Zoo'].tolist()
y_train_gym = ini_data['Gym'].tolist()
y_train_event_hall = ini_data['Event hall'].tolist()
y_train_garden = ini_data['Garden'].tolist()

chunks_school = [y_train_school[x:x + 24] for x in range(0, len(y_train_school), 24)]
chunks_zoo = [y_train_zoo[x:x + 24] for x in range(0, len(y_train_zoo), 24)]
chunks_gym = [y_train_gym[x:x + 24] for x in range(0, len(y_train_gym), 24)]
chunks_event_hall = [y_train_event_hall[x:x + 24] for x in range(0, len(y_train_event_hall), 24)]
chunks_garden = [y_train_garden[x:x + 24] for x in range(0, len(y_train_garden), 24)]

time_last_day = last_day_data['Date'].tolist()

# Using timestamp as a feature vector
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


X_train_school, X_test_school, Y_train_school, Y_test_school = train_test_split(X, Y_school, test_size=0.25, shuffle=False)
X_train_zoo, X_test_zoo, Y_train_zoo, Y_test_zoo = train_test_split(X, Y_zoo, test_size=0.25, shuffle=False)
X_train_gym, X_test_gym, Y_train_gym, Y_test_gym = train_test_split(X, Y_gym, test_size=0.25, shuffle=False)
X_train_hall, X_test_hall, Y_train_hall, Y_test_hall = train_test_split(X, Y_hall, test_size=0.25, shuffle=False)
X_train_garden, X_test_garden, Y_train_garden, Y_test_garden = train_test_split(X, Y_garden, test_size=0.25, shuffle=False)


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


# AKNN House
plot_values_house = prediction(load_house, chunks_house)
continuous_graph(plot_values_house, test_y_house, 'House power generation in kW via AKNN(31.12.2016)', 'Time')
step_graph(plot_values_house, test_y_house, 'House power generation in kW via AKNN(31.12.2016)', 'Time')


# Linear regression School
linear_model_school = LinearRegression()
linear_model_school.fit(X_train_school, Y_train_school)
y_predict_school = linear_model_school.predict(X_test_school)
cont_graph(y_predict_school, Y_test_school, 'School power consumption in kW via LR(Test Data)', 'Time-steps')

y_predict_31_12_school = linear_model_school.predict(test_x)
continuous_graph(y_predict_31_12_school, test_y_school, 'School power consumption in kW via LR(31.12.2016)', 'Time')
step_graph(y_predict_31_12_school, test_y_school, 'School power consumption in kW via LR(31.12.2016)', 'Time')

# Linear regression Zoo
linear_model_zoo = LinearRegression()
linear_model_zoo.fit(X_train_zoo, Y_train_zoo)
y_predict_zoo = linear_model_zoo.predict(X_test_zoo)
cont_graph(y_predict_zoo, Y_test_zoo, 'Zoo power consumption in kW via LR(Test Data)', 'Time-steps')

y_predict_31_12_zoo = linear_model_zoo.predict(test_x)
continuous_graph(y_predict_31_12_zoo, test_y_zoo, 'Zoo power consumption in kW via LR(31.12.2016)', 'Time')
step_graph(y_predict_31_12_zoo, test_y_zoo, 'Zoo power consumption in kW via LR(31.12.2016)', 'Time')

# Linear regression Gym
linear_model_gym = LinearRegression()
linear_model_gym.fit(X_train_gym, Y_train_gym)
y_predict_gym = linear_model_gym.predict(X_test_gym)
cont_graph(y_predict_gym, Y_test_gym, 'Gym power consumption in kW via LR(Test Data)', 'Time-steps')

y_predict_31_12_gym = linear_model_gym.predict(test_x)
continuous_graph(y_predict_31_12_gym, test_y_gym, 'Gym power consumption in kW via LR(31.12.2016)', 'Time')
step_graph(y_predict_31_12_gym, test_y_gym, 'Gym power consumption in kW via LR(31.12.2016)', 'Time')

# Linear regression Event hall
linear_model_hall = LinearRegression()
linear_model_hall.fit(X_train_hall, Y_train_hall)
y_predict_hall = linear_model_hall.predict(X_test_hall)
cont_graph(y_predict_hall, Y_test_hall, 'Event Hall power consumption in kW via LR(Test Data)', 'Time-steps')

y_predict_31_12_hall = linear_model_hall.predict(test_x)
continuous_graph(y_predict_31_12_hall, test_y_hall, 'Event Hall power consumption in kW via LR(31.12.2016)', 'Time')
step_graph(y_predict_31_12_hall, test_y_hall, 'Event Hall power consumption in kW via LR(31.12.2016)', 'Time')

# Linear regression Garden
linear_model_garden = LinearRegression()
linear_model_garden.fit(X_train_garden, Y_train_garden)
y_predict_garden = linear_model_garden.predict(X_test_garden)
cont_graph(y_predict_garden, Y_test_garden, 'Garden power consumption in kW via LR(Test Data)', 'Time-steps')

y_predict_31_12_garden = linear_model_garden.predict(test_x)
continuous_graph(y_predict_31_12_garden, test_y_garden, 'Garden power consumption in kW via LR(31.12.2016)', 'Time')
step_graph(y_predict_31_12_garden, test_y_garden, 'Garden power consumption in kW via LR(31.12.2016)', 'Time')


# KNN School
knn_model_school = KNeighborsRegressor(n_neighbors=6)
knn_model_school.fit(X_train_school, Y_train_school)
k_predict_school = knn_model_school.predict(X_test_school)
cont_graph(k_predict_school, Y_test_school, 'School power consumption in kW via KNN(Test Data)', 'Time-steps')

k_predict_31_12_school = knn_model_school.predict(test_x)
continuous_graph(k_predict_31_12_school, test_y_school, 'School power consumption in kW via KNN(31.12.2016)', 'Time')
step_graph(k_predict_31_12_school, test_y_school, 'School power consumption in kW via KNN(31.12.2016)', 'Time')

# KNN Zoo
knn_model_zoo = KNeighborsRegressor(n_neighbors=6)
knn_model_zoo.fit(X_train_zoo, Y_train_zoo)
k_predict_zoo = knn_model_zoo.predict(X_test_zoo)
cont_graph(k_predict_zoo, Y_test_zoo, 'Zoo power consumption in kW via KNN(Test Data)', 'Time-steps')

k_predict_31_12_zoo = knn_model_zoo.predict(test_x)
continuous_graph(k_predict_31_12_zoo, test_y_zoo, 'Zoo power consumption in kW via KNN(31.12.2016)', 'Time')
step_graph(k_predict_31_12_zoo, test_y_zoo, 'Zoo power consumption in kW via KNN(31.12.2016)', 'Time')

# KNN Gym
knn_model_gym = KNeighborsRegressor(n_neighbors=6)
knn_model_gym.fit(X_train_gym, Y_train_gym)
k_predict_gym = knn_model_gym.predict(X_test_gym)
cont_graph(k_predict_gym, Y_test_gym, 'Gym power consumption in kW via KNN(Test Data)', 'Time-steps')

k_predict_31_12_gym = knn_model_gym.predict(test_x)
continuous_graph(k_predict_31_12_gym, test_y_gym, 'Gym power consumption in kW via KNN(31.12.2016)', 'Time')
step_graph(k_predict_31_12_gym, test_y_gym, 'Gym power consumption in kW via KNN(31.12.2016)', 'Time')

# KNN Event Hall
knn_model_hall = KNeighborsRegressor(n_neighbors=6)
knn_model_hall.fit(X_train_hall, Y_train_hall)
k_predict_hall = knn_model_hall.predict(X_test_hall)
cont_graph(k_predict_hall, Y_test_hall, 'Event Hall power consumption in kW via KNN(Test Data)', 'Time-steps')

k_predict_31_12_hall = knn_model_hall.predict(test_x)
continuous_graph(k_predict_31_12_hall, test_y_hall, 'Event Hall power consumption in kW via KNN(31.12.2016)', 'Time')
step_graph(k_predict_31_12_hall, test_y_hall, 'Event Hall power consumption in kW via KNN(31.12.2016)', 'Time')

# KNN Garden
knn_model_garden = KNeighborsRegressor(n_neighbors=6)
knn_model_garden.fit(X_train_garden, Y_train_garden)
k_predict_garden = knn_model_garden.predict(X_test_garden)
cont_graph(k_predict_garden, Y_test_garden, 'Garden power consumption in kW via KNN(Test Data)', 'Time-steps')

k_predict_31_12_garden = knn_model_garden.predict(test_x)
continuous_graph(k_predict_31_12_garden, test_y_garden, 'Garden power consumption in kW via KNN(31.12.2016)', 'Time')
step_graph(k_predict_31_12_garden, test_y_garden, 'Garden power consumption in kW via KNN(31.12.2016)', 'Time')

# Gradient boosting regression School
params = {'n_estimators': 500, 'max_depth': 6, 'min_samples_split': 2, 'learning_rate': 0.01, 'loss': 'ls'}
gbr_model_school = GradientBoostingRegressor(**params)
gbr_model_school.fit(X_train_school, Y_train_school)
gbr_predict_school = gbr_model_school.predict(X_test_school)
cont_graph(gbr_predict_school, Y_test_school, 'School power consumption in kW via GBR(Test Data)', 'Time-steps')

gbr_predict_31_12_school = gbr_model_school.predict(test_x)
continuous_graph(gbr_predict_31_12_school, test_y_school, 'School power consumption in kW via GBR(31.12.2016)', 'Time')
step_graph(gbr_predict_31_12_school, test_y_school, 'School power consumption in kW via GBR(31.12.2016)', 'Time')

# Gradient boosting regression Zoo
gbr_model_zoo = GradientBoostingRegressor(**params)
gbr_model_zoo.fit(X_train_zoo, Y_train_zoo)
gbr_predict_zoo = gbr_model_zoo.predict(X_test_zoo)
cont_graph(gbr_predict_zoo, Y_test_zoo, 'Zoo power consumption in kW via GBR(Test Data)', 'Time-steps')

gbr_predict_31_12_zoo = gbr_model_zoo.predict(test_x)
continuous_graph(gbr_predict_31_12_zoo, test_y_zoo, 'Zoo power consumption in kW via GBR(31.12.2016)', 'Time')
step_graph(gbr_predict_31_12_zoo, test_y_zoo, 'Zoo power consumption in kW via GBR(31.12.2016)', 'Time')

# Gradient boosting regression Gym
gbr_model_gym = GradientBoostingRegressor(**params)
gbr_model_gym.fit(X_train_gym, Y_train_gym)
gbr_predict_gym = gbr_model_gym.predict(X_test_gym)
cont_graph(gbr_predict_gym, Y_test_gym, 'Gym power consumption in kW via GBR(Test Data)', 'Time-steps')

gbr_predict_31_12_gym = gbr_model_gym.predict(test_x)
continuous_graph(gbr_predict_31_12_gym, test_y_gym, 'Gym power consumption in kW via GBR(31.12.2016)', 'Time')
step_graph(gbr_predict_31_12_gym, test_y_gym, 'Gym power consumption in kW via GBR(31.12.2016)', 'Time')

# Gradient boosting regression Event hall
gbr_model_hall = GradientBoostingRegressor(**params)
gbr_model_hall.fit(X_train_hall, Y_train_hall)
gbr_predict_hall = gbr_model_hall.predict(X_test_hall)
cont_graph(gbr_predict_hall, Y_test_hall, 'Event hall power consumption in kW via GBR(Test Data)', 'Time-steps')

gbr_predict_31_12_hall = gbr_model_hall.predict(test_x)
continuous_graph(gbr_predict_31_12_hall, test_y_hall, 'Event hall power consumption in kW via GBR(31.12.2016)', 'Time')
step_graph(gbr_predict_31_12_hall, test_y_hall, 'Event hall power consumption in kW via GBR(31.12.2016)', 'Time')

# Gradient boosting regression Garden
gbr_model_garden = GradientBoostingRegressor(**params)
gbr_model_garden.fit(X_train_garden, Y_train_garden)
gbr_predict_garden = gbr_model_garden.predict(X_test_garden)
cont_graph(gbr_predict_garden, Y_test_garden, 'Garden power consumption in kW via GBR(Test Data)', 'Time-steps')

gbr_predict_31_12_garden = gbr_model_garden.predict(test_x)
continuous_graph(gbr_predict_31_12_garden, test_y_garden, 'Garden power consumption in kW via GBR(31.12.2016)', 'Time')
step_graph(gbr_predict_31_12_garden, test_y_garden, 'Garden power consumption in kW via GBR(31.12.2016)', 'Time')

# ANN School
mlp_school = MLPRegressor()
mlp_school.fit(X_train_school, Y_train_school)
mlp_predict_school = mlp_school.predict(X_test_school)
cont_graph(mlp_predict_school, Y_test_school, 'School power consumption in kW via ANN(Test Data)', 'Time-steps')

mlp_predict_31_12_school = mlp_school.predict(test_x)
continuous_graph(mlp_predict_31_12_school, test_y_school, 'School power consumption in kW via ANN(31.12.2016)', 'Time')
step_graph(mlp_predict_31_12_school, test_y_school, 'School power consumption in kW via ANN(31.12.2016)', 'Time')

# ANN Zoo
mlp_zoo = MLPRegressor()
mlp_zoo.fit(X_train_zoo, Y_train_zoo)
mlp_predict_zoo = mlp_zoo.predict(X_test_zoo)
cont_graph(mlp_predict_zoo, Y_test_zoo, 'Zoo power consumption in kW via ANN(Test Data)', 'Time-steps')

mlp_predict_31_12_zoo = mlp_zoo.predict(test_x)
continuous_graph(mlp_predict_31_12_zoo, test_y_zoo, 'Zoo power consumption in kW via ANN(31.12.2016)', 'Time')
step_graph(mlp_predict_31_12_zoo, test_y_zoo, 'Zoo power consumption in kW via ANN(31.12.2016)', 'Time')

# ANN Gym
mlp_gym = MLPRegressor()
mlp_gym.fit(X_train_gym, Y_train_gym)
mlp_predict_gym = mlp_gym.predict(X_test_gym)
cont_graph(mlp_predict_gym, Y_test_gym, 'Gym power consumption in kW via ANN(Test Data)', 'Time-steps')

mlp_predict_31_12_gym = mlp_gym.predict(test_x)
continuous_graph(mlp_predict_31_12_gym, test_y_gym, 'Gym power consumption in kW via ANN(31.12.2016)', 'Time')
step_graph(mlp_predict_31_12_gym, test_y_gym, 'Gym power consumption in kW via ANN(31.12.2016)', 'Time')

# ANN Event Hall
mlp_hall = MLPRegressor()
mlp_hall.fit(X_train_hall, Y_train_hall)
mlp_predict_hall = mlp_hall.predict(X_test_hall)
cont_graph(mlp_predict_hall, Y_test_hall, 'Event hall power consumption in kW via ANN(Test Data)', 'Time-steps')

mlp_predict_31_12_hall = mlp_hall.predict(test_x)
continuous_graph(mlp_predict_31_12_hall, test_y_hall, 'Event hall power consumption in kW via ANN(31.12.2016)', 'Time')
step_graph(mlp_predict_31_12_hall, test_y_hall, 'Event hall power consumption in kW via ANN(31.12.2016)', 'Time')

# ANN Garden
mlp_garden = MLPRegressor()
mlp_garden.fit(X_train_garden, Y_train_garden)
mlp_predict_garden = mlp_garden.predict(X_test_garden)
cont_graph(mlp_predict_garden, Y_test_garden, 'Garden power consumption in kW via ANN(Test Data)', 'Time-steps')

mlp_predict_31_12_garden = mlp_garden.predict(test_x)
continuous_graph(mlp_predict_31_12_garden, test_y_garden, 'Garden power consumption in kW via ANN(31.12.2016)', 'Time')
step_graph(mlp_predict_31_12_garden, test_y_garden, 'Garden power consumption in kW via ANN(31.12.2016)', 'Time')

# Ridge School
ridge_school = Ridge()
ridge_school.fit(X_train_school, Y_train_school)
r_predict_school = ridge_school.predict(X_test_school)
cont_graph(r_predict_school, Y_test_school, 'School power consumption in kW via Ridge regression(Test Data)', 'Time-steps')

r_predict_31_12_school = ridge_school.predict(test_x)
continuous_graph(r_predict_31_12_school, test_y_school, 'School power consumption in kW via Ridge regression(31.12.2016)', 'Time')
step_graph(r_predict_31_12_school, test_y_school, 'School power consumption in kW via Ridge regression(31.12.2016)', 'Time')

# Ridge Zoo
ridge_zoo = Ridge()
ridge_zoo.fit(X_train_zoo, Y_train_zoo)
r_predict_zoo = ridge_zoo.predict(X_test_zoo)
cont_graph(r_predict_zoo, Y_test_zoo, 'Zoo power consumption in kW via Ridge regression(Test Data)', 'Time-steps')

r_predict_31_12_zoo = ridge_zoo.predict(test_x)
continuous_graph(r_predict_31_12_zoo, test_y_zoo, 'Zoo power consumption in kW via Ridge regression(31.12.2016)', 'Time')
step_graph(r_predict_31_12_zoo, test_y_zoo, 'Zoo power consumption in kW via Ridge regression(31.12.2016)', 'Time')

# Ridge Gym
ridge_gym = Ridge()
ridge_gym.fit(X_train_gym, Y_train_gym)
r_predict_gym = ridge_gym.predict(X_test_gym)
cont_graph(r_predict_gym, Y_test_gym, 'Gym power consumption in kW via Ridge regression(Test Data)', 'Time-steps')

r_predict_31_12_gym = ridge_gym.predict(test_x)
continuous_graph(r_predict_31_12_gym, test_y_gym, 'Gym power consumption in kW via Ridge regression(31.12.2016)', 'Time')
step_graph(r_predict_31_12_gym, test_y_gym, 'Gym power consumption in kW via Ridge regression(31.12.2016)', 'Time')

# Ridge Event hall
ridge_hall = Ridge()
ridge_hall.fit(X_train_hall, Y_train_hall)
r_predict_hall = ridge_hall.predict(X_test_hall)
cont_graph(r_predict_hall, Y_test_hall, 'Event hall power consumption in kW via Ridge regression(Test Data)', 'Time-steps')

r_predict_31_12_hall = ridge_hall.predict(test_x)
continuous_graph(r_predict_31_12_hall, test_y_hall, 'Event hall power consumption in kW via Ridge regression(31.12.2016)', 'Time')
step_graph(r_predict_31_12_hall, test_y_hall, 'Event hall power consumption in kW via Ridge regression(31.12.2016)', 'Time')

# Ridge Garden
ridge_garden = Ridge()
ridge_garden.fit(X_train_garden, Y_train_garden)
r_predict_garden = ridge_garden.predict(X_test_garden)
cont_graph(r_predict_garden, Y_test_garden, 'Garden power consumption in kW via Ridge regression(Test Data)', 'Time-steps')

r_predict_31_12_garden = ridge_garden.predict(test_x)
continuous_graph(r_predict_31_12_garden, test_y_garden, 'Garden power consumption in kW via Ridge regression(31.12.2016)', 'Time')
step_graph(r_predict_31_12_garden, test_y_garden, 'Garden power consumption in kW via Ridge regression(31.12.2016)', 'Time')

# Lasso School
lasso_school = Lasso(alpha=0.1)
lasso_school.fit(X_train_school, Y_train_school)
lasso_predict_school = lasso_school.predict(X_test_school)
cont_graph(lasso_predict_school, Y_test_school, 'School power consumption in kW via Lasso regression(Test Data)', 'Time-steps')

lasso_predict_31_12_school = lasso_school.predict(test_x)
continuous_graph(lasso_predict_31_12_school, test_y_school, 'School power consumption in kW via Lasso regression(31.12.2016)', 'Time')
step_graph(lasso_predict_31_12_school, test_y_school, 'School power consumption in kW via Lasso regression(31.12.2016)', 'Time')

# Lasso Zoo
lasso_zoo = Lasso(alpha=0.1)
lasso_zoo.fit(X_train_zoo, Y_train_zoo)
lasso_predict_zoo = lasso_zoo.predict(X_test_zoo)
cont_graph(lasso_predict_zoo, Y_test_zoo, 'Zoo power consumption in kW via Lasso regression(Test Data)', 'Time-steps')

lasso_predict_31_12_zoo = lasso_zoo.predict(test_x)
continuous_graph(lasso_predict_31_12_zoo, test_y_zoo, 'Zoo power consumption in kW via Lasso regression(31.12.2016)', 'Time')
step_graph(lasso_predict_31_12_zoo, test_y_zoo, 'Zoo power consumption in kW via Lasso regression(31.12.2016)', 'Time')

# Lasso Gym
lasso_gym = Lasso(alpha=0.1)
lasso_gym.fit(X_train_gym, Y_train_gym)
lasso_predict_gym = lasso_gym.predict(X_test_gym)
cont_graph(lasso_predict_gym, Y_test_gym, 'Gym power consumption in kW via Lasso regression(Test Data)', 'Time-steps')

lasso_predict_31_12_gym = lasso_gym.predict(test_x)
continuous_graph(lasso_predict_31_12_gym, test_y_gym, 'Gym power consumption in kW via Lasso regression(31.12.2016)', 'Time')
step_graph(lasso_predict_31_12_gym, test_y_gym, 'Gym power consumption in kW via Lasso regression(31.12.2016)', 'Time')

# Lasso Event hall
lasso_hall = Lasso(alpha=0.1)
lasso_hall.fit(X_train_hall, Y_train_hall)
lasso_predict_hall = lasso_hall.predict(X_test_hall)
cont_graph(lasso_predict_hall, Y_test_hall, 'Event hall power consumption in kW via Lasso regression(Test Data)', 'Time-steps')

lasso_predict_31_12_hall = lasso_hall.predict(test_x)
continuous_graph(lasso_predict_31_12_hall, test_y_hall, 'Event hall power consumption in kW via Lasso regression(31.12.2016)', 'Time')
step_graph(lasso_predict_31_12_hall, test_y_hall, 'Event hall power consumption in kW via Lasso regression(31.12.2016)', 'Time')

# Lasso Garden
lasso_garden = Lasso(alpha=0.1)
lasso_garden.fit(X_train_garden, Y_train_garden)
lasso_predict_garden = lasso_garden.predict(X_test_garden)
cont_graph(lasso_predict_garden, Y_test_garden, 'Garden power consumption in kW via Lasso regression(Test Data)', 'Time-steps')

lasso_predict_31_12_garden = lasso_garden.predict(test_x)
continuous_graph(lasso_predict_31_12_garden, test_y_garden, 'Garden power consumption in kW via Lasso regression(31.12.2016)', 'Time')
step_graph(lasso_predict_31_12_garden, test_y_garden, 'Garden power consumption in kW via Lasso regression(31.12.2016)', 'Time')


# AKNN School
plot_values_school = prediction(load_school, chunks_school)
continuous_graph(plot_values_school, test_y_school, 'School power consumption in kW via AKNN(31.12.2016)', 'Time')
step_graph(plot_values_school, test_y_school, 'Prosumer-2 power consumption in kW via AKNN(31.12.2016)', 'Time')

# AKNN Zoo
plot_values_zoo = prediction(load_zoo, chunks_zoo)
continuous_graph(plot_values_zoo, test_y_zoo, 'Zoo power consumption in kW via AKNN(31.12.2016)', 'Time')
step_graph(plot_values_zoo, test_y_zoo, 'Zoo power consumption in kW via AKNN(31.12.2016)', 'Time')

# AKNN Gym
plot_values_gym = prediction(load_gym, chunks_gym)
continuous_graph(plot_values_gym, test_y_gym, 'Gym power consumption in kW via AKNN(31.12.2016)', 'Time')
step_graph(plot_values_gym, test_y_gym, 'Gym power consumption in kW via AKNN(31.12.2016)', 'Time')

# AKNN Event hall
plot_values_hall = prediction(load_event_hall, chunks_event_hall)
continuous_graph(plot_values_hall, test_y_hall, 'Event hall power consumption in kW via AKNN(31.12.2016)', 'Time')
step_graph(plot_values_hall, test_y_hall, 'Event hall power consumption in kW via AKNN(31.12.2016)', 'Time')

# AKNN Garden
plot_values_garden = prediction(load_garden, chunks_garden)
continuous_graph(plot_values_garden, test_y_garden, 'Garden power consumption in kW via AKNN(31.12.2016)', 'Time')
step_graph(plot_values_garden, test_y_garden, 'Garden power consumption in kW via AKNN(31.12.2016)', 'Time')


# Comparison Prosumers using step graph via AKNN. Excluding garden since we need only 5 in research paper.
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
plt.ylabel('Comparison prosumers power consumption in kW using AKNN on 31.12.2016')
plt.xticks([0, 5, 10, 15, 20],
           ['00:00', '05:00', '10:00', '15:00', '20:00'])
plt.xlabel('Time')
plt.legend()
plt.show()
