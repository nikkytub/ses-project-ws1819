import pandas as pd
import time
import datetime
from lpi_python import lpi_distance, lpi_mean
import matplotlib.pyplot as plt

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

training_data_aknn = data[:8736]

cost_pv = data[8736:]['PV energy cost (/kWh)'].tolist()
cost_wind = data[8736:]['wind energy cost (/kWh)'].tolist()
cost_grid = data[8736:]['external grid energy price (/kWh)'].tolist()

y_train_pv = training_data_aknn['PV energy cost (/kWh)'].tolist()
y_train_wind = training_data_aknn['wind energy cost (/kWh)'].tolist()
y_train_grid = training_data_aknn['external grid energy price (/kWh)'].tolist()

chunks_pv = [y_train_pv[x:x + 24] for x in range(0, len(y_train_pv), 24)]
chunks_wind = [y_train_wind[x:x + 24] for x in range(0, len(y_train_wind), 24)]
chunks_grid = [y_train_grid[x:x + 24] for x in range(0, len(y_train_grid), 24)]

# Prediction 31/12/2017 for PV
x_pv = []
dist_pv = []
d1_dist_pv = dict()
for x in chunks_pv:
    x_pv.append(x)
    d = lpi_distance(cost_pv, x)
    dist_pv.append(d)
    d1_dist_pv.update({d:x})

sorted_dict_pv = dict()
for key in sorted(d1_dist_pv.keys()):
    sorted_dict_pv.update({key: d1_dist_pv[key]})

d1_pv = []
for key in sorted_dict_pv.keys():
    d1_pv.append(sorted_dict_pv[key])

m_pv = lpi_mean(d1_pv[:6])
aknn_predicted_cost_pv = [m_pv]
plot_values_pv = []
for pred in aknn_predicted_cost_pv:
    for l in pred:
        plot_values_pv.append(l)
plt.plot(plot_values_pv, label='Predicted')
plt.plot(y_test_pv.values, label='Actual')
plt.ylabel('PV energy cost via AKNN')
plt.legend()
plt.show()

# Prediction 31/12/2017 for Wind
x_wind = []
dist_wind = []
d1_dist_wind = dict()
for x in chunks_wind:
    x_wind.append(x)
    d = lpi_distance(cost_wind, x)
    dist_wind.append(d)
    d1_dist_wind.update({d:x})

sorted_dict_wind = dict()
for key in sorted(d1_dist_wind.keys()):
    sorted_dict_wind.update({key: d1_dist_wind[key]})

d1_wind = []
for key in sorted_dict_wind.keys():
    d1_wind.append(sorted_dict_wind[key])

m_wind = lpi_mean(d1_wind[:6])
aknn_predicted_cost_wind = [m_wind]
plot_values_wind = []
for pred in aknn_predicted_cost_wind:
    for l in pred:
        plot_values_wind.append(l)
plt.plot(plot_values_wind, label='Predicted')
plt.plot(y_test_wind.values, label='Actual')
plt.ylabel('Wind energy cost via AKNN')
plt.legend()
plt.show()

# Prediction 31/12/2017 for Grid
x_grid = []
dist_grid = []
d1_dist_grid = dict()
for x in chunks_grid:
    x_grid.append(x)
    d = lpi_distance(cost_grid, x)
    dist_grid.append(d)
    d1_dist_grid.update({d:x})

sorted_dict_grid = dict()
for key in sorted(d1_dist_grid.keys()):
    sorted_dict_grid.update({key: d1_dist_grid[key]})

d1_grid = []
for key in sorted_dict_grid.keys():
    d1_grid.append(sorted_dict_grid[key])

m_grid = lpi_mean(d1_grid[:6])
aknn_predicted_cost_grid = [m_grid]
plot_values_grid = []
for pred in aknn_predicted_cost_grid:
    for l in pred:
        plot_values_grid.append(l)
plt.plot(plot_values_grid, label='Predicted')
plt.plot(y_test_grid.values, label='Actual')
plt.ylabel('External Grid energy price via AKNN')
plt.legend()
plt.show()

print("PV energy cost predicted values via AKNN for 31/12/2017 --> ", plot_values_pv)
print("Wind energy cost predicted values via AKNN for 31/12/2017 --> ", plot_values_wind)
print("External Grid energy price predicted values via AKNN for 31/12/2017 --> ", plot_values_grid)
