# Copyright (c) 2019 Nikhil Singh
# GNU General Public License v3.0+ (see COPYING or https://www.gnu.org/licenses/gpl-3.0.txt)
# name: School, Zoo, Gym, Event hall and Garden power consumption forecast via Adjusted K nearest neighbor
# author: Nikhil Singh (nikkytub@gmail.com)
# data-source: Karlsruhe Institute of Technology ("https://im.iism.kit.edu/sciber.php")

import pandas as pd
from sklearn.metrics import mean_squared_error
from lpi_python import lpi_distance, lpi_mean
import numpy as np
import math
import matplotlib.pyplot as plt
import csv

data = pd.read_table('SCiBER.txt')

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


def continuous_graph(prediction, actual, ylabl, xlabl):
    plt.plot(prediction, label='Predicted')
    plt.plot(actual.values, label='Actual')
    plt.ylabel(ylabl)
    plt.xticks([0, 5, 10, 15, 20],
               ['00:00', '05:00', '10:00', '15:00', '20:00'])
    plt.xlabel(xlabl)
    plt.legend()
    return plt.show()


# Consumption prediction for School for 31/12/2016
def prediction(load, chunks):
    aknn_predicted_load_school = [aknn(load, chunks)]
    plot_values = []
    for pred in aknn_predicted_load_school:
        for l in pred:
            plot_values.append(l)
    return plot_values


plot_values_school = prediction(load_school, chunks_school)
plot_values_zoo = prediction(load_zoo, chunks_zoo)
plot_values_gym = prediction(load_gym, chunks_gym)
plot_values_event_hall = prediction(load_event_hall, chunks_event_hall)
plot_values_garden = prediction(load_garden, chunks_garden)

step_graph(plot_values_school, y_test_school, 'Prosumer-1 power consumption in kW (AKNN)', 'Time')
continuous_graph(plot_values_school, y_test_school, 'Prosumer-1 power consumption in kW (AKNN)', 'Time')

step_graph(plot_values_zoo, y_test_zoo, 'Prosumer-2 power consumption in kW (AKNN)', 'Time')
continuous_graph(plot_values_zoo, y_test_zoo, 'Prosumer-2 power consumption in kW (AKNN)', 'Time')

step_graph(plot_values_gym, y_test_gym, 'Prosumer-3 power consumption in kW (AKNN)', 'Time')
continuous_graph(plot_values_gym, y_test_gym, 'Prosumer-3 power consumption in kW (AKNN)', 'Time')

step_graph(plot_values_event_hall, y_test_event_hall, 'Prosumer-4 power consumption in kW (AKNN)', 'Time')
continuous_graph(plot_values_event_hall, y_test_event_hall, 'Prosumer-4 power consumption in kW (AKNN)', 'Time')

step_graph(plot_values_garden, y_test_garden, 'Prosumer-5 power consumption in kW (AKNN)', 'Time')
continuous_graph(plot_values_garden, y_test_garden, 'Prosumer-5 power consumption in kW (AKNN)', 'Time')


mean_school = np.mean(y_test_school)
mse_school = mean_squared_error(y_test_school, plot_values_school)
rmse_school = math.sqrt(mse_school)
nrmse_school = rmse_school / mean_school
print('Mean for School power consumption is {}'.format(mean_school))
print('MSE for School power consumption is {}'.format(mse_school))
print('RMSE for School power consumption is --> {}'.format(rmse_school))
print('NRMSE for School power consumption via AKNN is --> {}'.format(nrmse_school))


mean_zoo = np.mean(y_test_zoo)
mse_zoo = mean_squared_error(y_test_zoo, plot_values_zoo)
rmse_zoo = math.sqrt(mse_zoo)
nrmse_zoo = rmse_zoo / mean_zoo
print('Mean for Zoo power consumption is {}'.format(mean_zoo))
print('MSE for Zoo power consumption is {}'.format(mse_zoo))
print('RMSE for Zoo power consumption is --> {}'.format(rmse_zoo))
print('NRMSE for Zoo power consumption via AKNN is --> {}'.format(nrmse_zoo))


mean_gym = np.mean(y_test_gym)
mse_gym = mean_squared_error(y_test_gym, plot_values_gym)
rmse_gym = math.sqrt(mse_gym)
nrmse_gym = rmse_gym / mean_gym
print('Mean for Gym power consumption is {}'.format(mean_gym))
print('MSE for Gym power consumption is {}'.format(mse_gym))
print('RMSE for Gym power consumption is --> {}'.format(rmse_gym))
print('NRMSE for Gym power consumption via AKNN is --> {}'.format(nrmse_gym))


mean_event_hall = np.mean(y_test_event_hall)
mse_event_hall = mean_squared_error(y_test_event_hall, plot_values_event_hall)
rmse_event_hall = math.sqrt(mse_event_hall)
nrmse_event_hall = rmse_event_hall / mean_event_hall
print('Mean for Event Hall power consumption is {}'.format(mean_event_hall))
print('MSE for Event Hall power consumption is {}'.format(mse_event_hall))
print('RMSE for Event Hall power consumption is --> {}'.format(rmse_event_hall))
print('NRMSE for Event Hall power consumption via AKNN is --> {}'.format(nrmse_event_hall))


mean_garden = np.mean(y_test_garden)
mse_garden = mean_squared_error(y_test_garden, plot_values_garden)
rmse_garden = math.sqrt(mse_garden)
nrmse_garden = rmse_garden / mean_garden
print('Mean for Garden power consumption is {}'.format(mean_garden))
print('MSE for Garden power consumption is {}'.format(mse_garden))
print('RMSE for Garden power consumption is --> {}'.format(rmse_garden))
print('NRMSE for Garden power consumption via AKNN is --> {}'.format(nrmse_garden))


# To create a csv file of forecast which can be used for optimization
time_list = ["Date"]
for a in time_last_day:
    time_list.append(a)

prediction_school = ["School power consumption in kW"]
for a in plot_values_school:
    prediction_school.append(a)

prediction_zoo = ["Zoo power consumption in kW"]
for a in plot_values_zoo:
    prediction_zoo.append(a)

prediction_gym = ["Gym power consumption in kW"]
for a in plot_values_gym:
    prediction_gym.append(a)

prediction_event_hall = ["Event Hall power consumption in kW"]
for a in plot_values_event_hall:
    prediction_event_hall.append(a)

prediction_garden = ["Garden power consumption in kW"]
for a in plot_values_garden:
    prediction_garden.append(a)

zip_time_and_forecast = zip(time_list, prediction_school, prediction_zoo, prediction_gym, prediction_event_hall,
                            prediction_garden)
x = tuple(zip_time_and_forecast)
with open('result_load_school_zoo_gym_hall_garden.csv', 'w') as csvFile:
    for a in x:
        writer = csv.writer(csvFile)
        writer.writerow(a)
