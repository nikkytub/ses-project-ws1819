# Copyright (c) 2019 Nikhil Singh
# GNU General Public License v3.0+ (see COPYING or https://www.gnu.org/licenses/gpl-3.0.txt)
# name: PV and Wind power generation forecast via Adjusted K nearest neighbor
# author: Nikhil Singh (nikkytub@gmail.com)
# data-source: Pecan Street Data from 01/01/2015 to 31/12/2015

import pandas as pd
from sklearn.metrics import mean_squared_error
from lpi_python import lpi_distance, lpi_mean
import numpy as np
import math
import matplotlib.pyplot as plt
import csv

data = pd.read_csv('pec.csv', parse_dates=True)

# 01/01/2015 to 30/12/2015
ini_data = data[:8736]

# 30/12/2015
prev_day_data = data[8712:8736]

# 31/12/2015
last_day_data = data[8736:]

y_test_pv = last_day_data['PV']
y_test_wind = last_day_data['Wind']

generation_pv = prev_day_data['PV'].tolist()
generation_wind = prev_day_data['Wind'].tolist()

y_train_pv = ini_data['PV'].tolist()
y_train_wind = ini_data['Wind'].tolist()

chunks_pv = [y_train_pv[x:x + 24] for x in range(0, len(y_train_pv), 24)]
chunks_wind = [y_train_wind[x:x + 24] for x in range(0, len(y_train_wind), 24)]

time_last_day = last_day_data['Unnamed: 0'].tolist()


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


# Generation prediction for PV for 31/12/2015
aknn_predicted_generation_pv = [aknn(generation_pv, chunks_pv)]
plot_values_pv = []
for pred in aknn_predicted_generation_pv:
    for l in pred:
        plot_values_pv.append(l)
plt.plot(plot_values_pv, label='Predicted')
plt.plot(y_test_pv.values, label='Actual')
plt.ylabel('PV power generation in kW (AKNN)')
plt.xlabel('Hours')
plt.legend()
plt.show()

# Generation prediction for Wind for 31/12/2015
aknn_predicted_generation_wind = [aknn(generation_wind, chunks_wind)]
plot_values_wind = []
for pred in aknn_predicted_generation_wind:
    for l in pred:
        plot_values_wind.append(l)
plt.plot(plot_values_wind, label='Predicted')
plt.plot(y_test_wind.values, label='Actual')
plt.ylabel('Wind power generation in kW (AKNN)')
plt.xlabel('Hours')
plt.legend()
plt.show()

mean_pv = np.mean(y_test_pv)
mse_pv = mean_squared_error(y_test_pv, plot_values_pv)
rmse_pv = math.sqrt(mse_pv)
nrmse_pv = rmse_pv / mean_pv
print('Mean for PV power generation is {}'.format(mean_pv))
print('MSE for PV power generation is {}'.format(mse_pv))
print('RMSE for PV power generation is --> {}'.format(rmse_pv))
print('NRMSE for PV power generation via AKNN is --> {}'.format(nrmse_pv))

mean_wind = np.mean(y_test_wind)
mse_wind = mean_squared_error(y_test_wind, plot_values_wind)
rmse_wind = math.sqrt(mse_wind)
nrmse_wind = rmse_wind / mean_wind
print('Mean for Wind power generation is {}'.format(mean_wind))
print('MSE for Wind power generation is {}'.format(mse_wind))
print('RMSE for Wind power generation is --> {}'.format(rmse_wind))
print('NRMSE for Wind power generation via AKNN is --> {}'.format(nrmse_wind))

# To create a csv file of forecast which can be used for optimization
date_time = ["Time"]
for a in time_last_day:
    date_time.append(a)

prediction_pv = ["PV power generation in kW"]
for a in plot_values_pv:
    prediction_pv.append(a)

prediction_wind = ["Wind power generation in kW"]
for a in plot_values_wind:
    prediction_wind.append(a)

zip_time_and_forecast = zip(date_time, prediction_pv, prediction_wind)
x = tuple(zip_time_and_forecast)
with open('result_gen_60min_pv_wind.csv', 'w') as csvFile:
    for a in x:
        writer = csv.writer(csvFile)
        writer.writerow(a)
