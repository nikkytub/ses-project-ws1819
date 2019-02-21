# Copyright (c) 2019 Nikhil Singh
# GNU General Public License v3.0+ (see COPYING or https://www.gnu.org/licenses/gpl-3.0.txt)
# name: Single Household and office load prediction via Adjusted K nearest neighbor
# author: Nikhil Singh (nikkytub@gmail.com)
# data-source: Smart energy systems (homework3-forecasting) on ISIS from 01/01/2015 to 31/12/2016

import pandas as pd
import time
import datetime
from sklearn.metrics import mean_squared_error
from lpi_python import lpi_distance, lpi_mean
import numpy as np
import math
import matplotlib.pyplot as plt

data = pd.read_csv('Excercise3-data.csv', parse_dates=True)
times = []
for d in data['Unnamed: 0']:
    t = time.mktime(datetime.datetime.strptime(d, "%Y-%m-%d %H:%M:%S%z").timetuple())
    times.append(str(t))
data.insert(1, 'Timestamps', times)
data = data.drop('Unnamed: 0', axis=1)
ini_data = data[:27742]
week_data = data[27742:28078]

# Training data for Single household
x_train = ini_data.drop('Building 1', axis=1)
y_train = ini_data['Building 1']

# Testing data for Single household
x_test = week_data.drop('Building 1', axis=1)
y_test = week_data['Building 1']

# Training data for Office
a_train = ini_data.drop('Building 2', axis=1)
b_train = ini_data['Building 2']

# Testing data for Office
a_test = week_data.drop('Building 2', axis=1)
b_test = week_data['Building 2']

load_actual_b1 = data[27742:27790]['Building 1']
load_actual_b2 = data[27742:27790]['Building 2']

# Because it should be a multiple of 48(each day values)
training_data_aknn = data[:27744]

print("Please be patient while we are executing the file. It may take around two minutes...")

# Actual load from 31/07 to 06/08 for Single household
load_day1_b1 = data[27694:27742]['Building 1'].tolist()
load_day2_b1 = data[27742:27790]['Building 1'].tolist()
load_day3_b1 = data[27790:27838]['Building 1'].tolist()
load_day4_b1 = data[27838:27886]['Building 1'].tolist()
load_day5_b1 = data[27886:27934]['Building 1'].tolist()
load_day6_b1 = data[27934:27982]['Building 1'].tolist()
load_day7_b1 = data[27982:28030]['Building 1'].tolist()

y_train_b1 = training_data_aknn['Building 1'].tolist()
chunks_b1 = [y_train_b1[x:x+48] for x in range(0, len(y_train_b1), 48)]

# Actual load from 31/07 to 06/08 for Office
load_day1_b2 = data[27694:27742]['Building 2'].tolist()
load_day2_b2 = data[27742:27790]['Building 2'].tolist()
load_day3_b2 = data[27790:27838]['Building 2'].tolist()
load_day4_b2 = data[27838:27886]['Building 2'].tolist()
load_day5_b2 = data[27886:27934]['Building 2'].tolist()
load_day6_b2 = data[27934:27982]['Building 2'].tolist()
load_day7_b2 = data[27982:28030]['Building 2'].tolist()

y_train_b2 = training_data_aknn['Building 2'].tolist()
chunks_b2 = [y_train_b2[x:x+48] for x in range(0, len(y_train_b2), 48)]


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


# Prediction for single household: 01/08/2016
aknn(load_day1_b1, chunks_b1)

# Prediction for single household: 02/08/2016
aknn(load_day2_b1, chunks_b1)

# Prediction for single household: 03/08/2016
aknn(load_day3_b1, chunks_b1)

# Prediction for single household: 04/08/2016
aknn(load_day4_b1, chunks_b1)

# Prediction for single household: 05/08/2016
aknn(load_day5_b1, chunks_b1)

# Prediction for single household: 06/08/2016
aknn(load_day6_b1, chunks_b1)

# Prediction for single household: 07/08/2016
aknn(load_day7_b1, chunks_b1)

# Prediction for single household: 01/08/2016 to 07/08/2016
aknn_predicted_b1 = [aknn(load_day1_b1, chunks_b1), aknn(load_day2_b1, chunks_b1), aknn(load_day3_b1, chunks_b1),
                     aknn(load_day4_b1, chunks_b1), aknn(load_day5_b1, chunks_b1), aknn(load_day6_b1, chunks_b1),
                     aknn(load_day7_b1, chunks_b1)]

# Prediction for single household on 01/08/2016
aknn_predicted_b1_24hours = [aknn(load_day1_b1, chunks_b1)]

plot_values_b1 = []
for pred in aknn_predicted_b1:
    for l in pred:
        plot_values_b1.append(l)
plt.plot(plot_values_b1, label='Predicted')
plt.plot(y_test.values, label='Actual')
plt.ylabel('Single Household Load in kW (AKNN)')
plt.xlabel('Time-steps')
plt.legend()
plt.show()

plot_values_b1_day = []
for pred in aknn_predicted_b1_24hours:
    for l in pred:
        plot_values_b1_day.append(l)
plt.plot(plot_values_b1_day, label='Predicted')
plt.plot(load_actual_b1.values, label='Actual')
plt.ylabel('Single Household Load in kW (AKNN)')
plt.xlabel('Time-steps')
plt.legend()
plt.show()

# Prediction for Office: 01/08/2016
aknn(load_day1_b2, chunks_b2)

# Prediction for Office: 02/08/2016
aknn(load_day2_b2, chunks_b2)

# Prediction for Office: 03/08/2016
aknn(load_day3_b2, chunks_b2)

# Prediction for Office: 04/08/2016
aknn(load_day4_b2, chunks_b2)

# Prediction for Office: 05/08/2016
aknn(load_day5_b2, chunks_b2)

# Prediction for Office: 06/08/2016
aknn(load_day6_b2, chunks_b2)

# Prediction for Office: 07/08/2016
aknn(load_day7_b2, chunks_b2)

# Prediction for office: 01/08/2016 to 07/08/2016
aknn_predicted_b2 = [aknn(load_day1_b2, chunks_b2), aknn(load_day2_b2, chunks_b2), aknn(load_day3_b2, chunks_b2),
                     aknn(load_day4_b2, chunks_b2), aknn(load_day5_b2, chunks_b2), aknn(load_day6_b2, chunks_b2),
                     aknn(load_day7_b2, chunks_b2)]

# Prediction for office on 01/08/2016
aknn_predicted_b2_24hours = [aknn(load_day1_b2, chunks_b2)]

plot_values_b2 = []
for pred in aknn_predicted_b2:
    for l in pred:
        plot_values_b2.append(l)
plt.plot(plot_values_b2, label='Predicted')
plt.plot(b_test.values, label='Actual')
plt.ylabel('Office Load in kW (AKNN)')
plt.xlabel('Time-steps')
plt.legend()
plt.show()

plot_values_b2_day = []
for pred in aknn_predicted_b2_24hours:
    for l in pred:
        plot_values_b2_day.append(l)
plt.plot(plot_values_b2_day, label='Predicted')
plt.plot(load_actual_b2.values, label='Actual')
plt.ylabel('Office Load in kW (AKNN)')
plt.xlabel('Time-steps')
plt.legend()
plt.show()

y_mean = np.mean(y_test)
mse_b1 = mean_squared_error(y_test, plot_values_b1)
rmse_b1 = math.sqrt(mse_b1)
nrmse_b1 = rmse_b1 / y_mean
print('NRMSE for Single household via AKNN is --> {}'.format(nrmse_b1))

b_mean = np.mean(b_test)
mse_b2 = mean_squared_error(b_test, plot_values_b2)
rmse_b2 = math.sqrt(mse_b2)
nrmse_b2 = rmse_b2 / b_mean
print('NRMSE for Office via AKNN is --> {}'.format(nrmse_b2))
