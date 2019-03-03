# Copyright (c) 2019 Nikhil Singh
# GNU General Public License v3.0+ (see COPYING or https://www.gnu.org/licenses/gpl-3.0.txt)
# name: School, Zoo and Garden power consumption forecast via Adjusted K nearest neighbor
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

# 01.01.2016 to 30.12.2016
ini_data = data[105119:140063]

# 30.12.2016
prev_day_data = data[139967:140063]

# 31.12.2016
last_day_data = data[140063:140159]

y_test_garden = last_day_data['Garden']
y_test_school = last_day_data['School']
y_test_zoo = last_day_data['Zoo']

generation_garden = prev_day_data['Garden'].tolist()
generation_school = prev_day_data['School'].tolist()
generation_zoo = prev_day_data['Zoo'].tolist()

y_train_garden = ini_data['Garden'].tolist()
y_train_school = ini_data['School'].tolist()
y_train_zoo = ini_data['Zoo'].tolist()

chunks_garden = [y_train_garden[x:x + 96] for x in range(0, len(y_train_garden), 96)]
chunks_school = [y_train_school[x:x + 96] for x in range(0, len(y_train_school), 96)]
chunks_zoo = [y_train_zoo[x:x + 96] for x in range(0, len(y_train_zoo), 96)]

time_last_day = last_day_data['Date'].tolist()


def aknn(generation, chunks):
    x_generation = []
    dist = []
    d1_dist = dict()
    for x in chunks:
        x_generation.append(x)
        d = lpi_distance(generation, x)
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


# Consumption prediction for Garden for 31/12/2016
aknn_predicted_generation_garden = [aknn(generation_garden, chunks_garden)]
plot_values_garden = []
for pred in aknn_predicted_generation_garden:
    for l in pred:
        plot_values_garden.append(l)
plt.plot(plot_values_garden, label='Predicted')
plt.plot(y_test_garden.values, label='Actual')
plt.ylabel('Garden power consumption in kW (AKNN)')
plt.xlabel('Time-steps')
plt.legend()
plt.show()

# Consumption prediction for School for 31/12/2016
aknn_predicted_generation_school = [aknn(generation_school, chunks_school)]
plot_values_school = []
for pred in aknn_predicted_generation_school:
    for l in pred:
        plot_values_school.append(l)
plt.plot(plot_values_school, label='Predicted')
plt.plot(y_test_school.values, label='Actual')
plt.ylabel('School power consumption in kW (AKNN)')
plt.xlabel('Time-steps')
plt.legend()
plt.show()

# Consumption prediction for Zoo for 31/12/2016
aknn_predicted_generation_zoo = [aknn(generation_zoo, chunks_zoo)]
plot_values_zoo = []
for pred in aknn_predicted_generation_zoo:
    for l in pred:
        plot_values_zoo.append(l)
plt.plot(plot_values_zoo, label='Predicted')
plt.plot(y_test_zoo.values, label='Actual')
plt.ylabel('Zoo power consumption in kW (AKNN)')
plt.xlabel('Time-steps')
plt.legend()
plt.show()

mean_garden = np.mean(y_test_garden)
mse_garden = mean_squared_error(y_test_garden, plot_values_garden)
rmse_garden = math.sqrt(mse_garden)
nrmse_garden = rmse_garden / mean_garden
print('Mean for Garden power consumption is {}'.format(mean_garden))
print('MSE for Garden power consumption is {}'.format(mse_garden))
print('RMSE for Garden power consumption is --> {}'.format(rmse_garden))
print('NRMSE for Garden power consumption via AKNN is --> {}'.format(nrmse_garden))


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


# To create a csv file of forecast which can be used for optimization
time_list = ["Date"]
for a in time_last_day:
    time_list.append(a)

prediction_garden = ["Garden power consumption in kW"]
for a in plot_values_garden:
    prediction_garden.append(a)

prediction_school = ["School power consumption in kW"]
for a in plot_values_school:
    prediction_school.append(a)

prediction_zoo = ["Zoo power consumption in kW"]
for a in plot_values_zoo:
    prediction_zoo.append(a)

zip_time_and_forecast = zip(time_list, prediction_garden, prediction_school, prediction_zoo)
x = tuple(zip_time_and_forecast)
with open('result_load_school_zoo_garden.csv', 'w') as csvFile:
    for a in x:
        writer = csv.writer(csvFile)
        writer.writerow(a)
