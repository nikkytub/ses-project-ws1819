# Copyright (c) 2019 Nikhil Singh
# GNU General Public License v3.0+ (see COPYING or https://www.gnu.org/licenses/gpl-3.0.txt)
# name: Supermarket load prediction via Adjusted K nearest neighbor
# author: Nikhil Singh (nikkytub@gmail.com)
# data-source: Randomly created supermarket load in the range (42 to 52kW) from 01/01/2017 to 31/12/2017

import pandas as pd
from lpi_python import lpi_distance, lpi_mean
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import math
import numpy as np
import csv
import time

data = pd.read_csv('supermarkt.csv', encoding="ISO-8859-1")

ini_data = data[:8736]
last_day_data = data[8736:]
y_test_supermarket = last_day_data['Load(kW)']
load_supermarket = data[8736:]['Load(kW)'].tolist()
y_train_supermarket = ini_data['Load(kW)'].tolist()
chunks_supermarket = [y_train_supermarket[x:x + 24] for x in range(0, len(y_train_supermarket), 24)]
time_last_day = last_day_data['Timestamps'].tolist()


def aknn(load, chunks):
    x_load = []
    dist = []
    d1_dist = dict()
    for x in chunks:
        x_load.append(x)
        d = lpi_distance(load, x)
        dist.append(d)
        d1_dist.update({d:x})
    sorted_dict = dict()
    for key in sorted(d1_dist.keys()):
        sorted_dict.update({key: d1_dist[key]})
    d1_load = []
    for key in sorted_dict.keys():
        d1_load.append(sorted_dict[key])
    m = lpi_mean(d1_load[:6])
    return m


# Load Prediction for Supermarket on 31/12/2017
aknn(load_supermarket, chunks_supermarket)
aknn_predicted_cost_sup = [aknn(load_supermarket, chunks_supermarket)]
plot_values_sup = []
for pred in aknn_predicted_cost_sup:
    for l in pred:
        plot_values_sup.append(l)
plt.plot(plot_values_sup, label='Predicted')
plt.plot(y_test_supermarket.values, label='Actual')
plt.ylabel('Supermarket Load in kW (AKNN)')
plt.xlabel('Hours')
plt.legend()
plt.show()

# Mean, MSE, RMSE and NRMSE of Supermarket using AKNN
y_mean = np.mean(y_test_supermarket)
mse = mean_squared_error(y_test_supermarket, plot_values_sup)
rmse = math.sqrt(mse)
nrmse = rmse / y_mean
print('Mean for supermarket {}'.format(y_mean))
print('MSE for supermarket via AKNN is {}'.format(mse))
print('RMSE for supermarket via AKNN is --> {}'.format(rmse))
print('NRMSE for supermarket via AKNN is --> {}'.format(nrmse))

# To create a csv file of forecast which can be used for optimization
date_time = ["Time"]
for a in time_last_day:
    forecast_time = time.ctime(float(a))
    date_time.append(forecast_time)

time_list = ["Timestamp"]
for a in time_last_day:
    time_list.append(a)

prediction_sup = ["Load in kW (Supermarket)"]
for a in plot_values_sup:
    prediction_sup.append(a)

zip_time_and_forecast = zip(date_time, time_list, prediction_sup)
x = tuple(zip_time_and_forecast)
with open('result_load_supermarket.csv', 'w') as csvFile:
    for a in x:
        writer = csv.writer(csvFile)
        writer.writerow(a)
