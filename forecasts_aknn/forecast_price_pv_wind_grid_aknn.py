# Copyright (c) 2019 Nikhil Singh
# GNU General Public License v3.0+ (see COPYING or https://www.gnu.org/licenses/gpl-3.0.txt)
# name: PV, Wind and Grid price prediction via Adjusted K nearest neighbor
# author: Nikhil Singh (nikkytub@gmail.com)

import pandas as pd
import time
import datetime
from lpi_python import lpi_distance, lpi_mean
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import math

data = pd.read_csv('task3_energycost.csv', encoding="ISO-8859-1")
times = []
for d in data['time']:
    t = time.mktime(datetime.datetime.strptime(d, "%m/%d/%Y %H:%M").timetuple())
    times.append(str(t))
data.insert(1, 'Timestamps', times)
data = data.drop('time', axis=1)
ini_data = data[:8736]
last_day_data = data[8736:]
y_test_pv = last_day_data['PV energy cost (/kWh)']
y_test_wind = last_day_data['wind energy cost (/kWh)']
y_test_grid = last_day_data['external grid energy price (/kWh)']
cost_pv = last_day_data['PV energy cost (/kWh)'].tolist()
cost_wind = last_day_data['wind energy cost (/kWh)'].tolist()
cost_grid = last_day_data['external grid energy price (/kWh)'].tolist()
y_train_pv = ini_data['PV energy cost (/kWh)'].tolist()
y_train_wind = ini_data['wind energy cost (/kWh)'].tolist()
y_train_grid = ini_data['external grid energy price (/kWh)'].tolist()
chunks_pv = [y_train_pv[x:x + 24] for x in range(0, len(y_train_pv), 24)]
chunks_wind = [y_train_wind[x:x + 24] for x in range(0, len(y_train_wind), 24)]
chunks_grid = [y_train_grid[x:x + 24] for x in range(0, len(y_train_grid), 24)]


def aknn(cost, chunks):
    x_load = []
    dist = []
    d1_dist = dict()
    for x in chunks:
        x_load.append(x)
        d = lpi_distance(cost, x)
        dist.append(d)
        d1_dist.update({d:x})
    sorted_dict = dict()
    for key in sorted(d1_dist.keys()):
        sorted_dict.update({key: d1_dist[key]})
    d1_cost = []
    for key in sorted_dict.keys():
        d1_cost.append(sorted_dict[key])
    m = lpi_mean(d1_cost[:6])
    return m


# Price prediction for PV for 31/12/2017
aknn(cost_pv, chunks_pv)
aknn_predicted_cost_pv = [aknn(cost_pv, chunks_pv)]
plot_values_pv = []
for pred in aknn_predicted_cost_pv:
    for l in pred:
        plot_values_pv.append(l)
plt.plot(plot_values_pv, label='Predicted')
plt.plot(y_test_pv.values, label='Actual')
plt.ylabel('PV energy cost(Euro/kWh) via AKNN')
plt.xlabel('Hour')
plt.legend()
plt.show()

# Price prediction for Wind for 31/12/2017
aknn(cost_wind, chunks_wind)
aknn_predicted_cost_wind = [aknn(cost_wind, chunks_wind)]
plot_values_wind = []
for pred in aknn_predicted_cost_wind:
    for l in pred:
        plot_values_wind.append(l)
plt.plot(plot_values_wind, label='Predicted')
plt.plot(y_test_wind.values, label='Actual')
plt.ylabel('Wind energy cost(Euro/kWh) via AKNN')
plt.xlabel('Hour')
plt.legend()
plt.show()

# Price prediction for Grid for 31/12/2017
aknn(cost_grid, chunks_grid)
aknn_predicted_cost_grid = [aknn(cost_grid, chunks_grid)]
plot_values_grid = []
for pred in aknn_predicted_cost_grid:
    for l in pred:
        plot_values_grid.append(l)
plt.plot(plot_values_grid, label='Predicted')
plt.plot(y_test_grid.values, label='Actual')
plt.ylabel('External Grid energy price(Euro/kWh) via AKNN')
plt.xlabel('Hour')
plt.legend()
plt.show()

# Mean, MSE, RMSE and NRMSE of PV
y_mean_pv = np.mean(y_test_pv)
mse_pv = mean_squared_error(y_test_pv, plot_values_pv)
rmse_pv = math.sqrt(mse_pv)
nrmse_pv = rmse_pv / y_mean_pv
print('Mean Price PV(Euro/kWh) {}'.format(y_mean_pv))
print('MSE PV {}'.format(mse_pv))
print('RMSE for PV via AKNN is --> {}'.format(rmse_pv))
print('NRMSE for PV via AKNN is --> {}'.format(nrmse_pv))

# Mean, MSE, RMSE and NRMSE of Wind
y_mean_wind = np.mean(y_test_wind)
mse_wind = mean_squared_error(y_test_wind, plot_values_wind)
rmse_wind = math.sqrt(mse_wind)
nrmse_wind = rmse_wind / y_mean_wind
print('Mean Price Wind(Euro/kWh) {}'.format(y_mean_wind))
print('MSE Wind {}'.format(mse_wind))
print('RMSE for Wind via AKNN is --> {}'.format(rmse_wind))
print('NRMSE for Wind via AKNN is --> {}'.format(nrmse_wind))

# Mean, MSE, RMSE and NRMSE of Grid
y_mean_grid = np.mean(y_test_grid)
mse_grid = mean_squared_error(y_test_grid, plot_values_grid)
rmse_grid = math.sqrt(mse_grid)
nrmse_grid = rmse_grid / y_mean_grid
print('Mean Price Grid(Euro/kWh) {}'.format(y_mean_grid))
print('MSE Grid {}'.format(mse_grid))
print('RMSE for Wind via AKNN is --> {}'.format(rmse_grid))
print('NRMSE for Wind via AKNN is --> {}'.format(nrmse_grid))
