# name: Prosumers (House, School, Zoo, Gym and Event hall) power consumption forecast
#       via LR, KNN, GBR, MLP, Ridge, Lasso and AKNN.
# author: Nikhil Singh (nikhil.singh@campus.tu-berlin.de)
# data-source: Karlsruhe Institute of Technology ("https://im.iism.kit.edu/sciber.php") and ISIS homework-3
# Reference: Some ideas and code taken from ISIS full tutorial

import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit, KFold, GridSearchCV
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

#pd.set_option('display.max_rows', None)


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


def mae(y_true, y_pred):
    """Mean Absolute Error"""
    return np.average(np.abs(y_pred - y_true))


def rmse(y_true, y_pred):
    """Root Mean Square Error"""
    return np.sqrt(np.average((y_true - y_pred) ** 2))


# 24 hours data from 31.05.2016 01:00 to 01.06.2016 00:00 is missing in the sciber.txt data-set
data_prosumers = pd.read_table('SCiBER.txt')
data_house = pd.read_csv('Excercise3-data.csv', parse_dates=True)

# 01.01.2015 to 01.12.2016
dp = data_prosumers[70079:137279:4]
dp.index = range(16800)

dh = data_house[:33648]
# To align the data with sciber.txt
dh = dh.drop(dh.index[24768:24816])
dh = dh[::2]
dh = dh.drop('Building 2', axis=1)
dh.index = range(16800)

x = pd.DataFrame(index=range(16800))
x.loc[:, "Date"] = dh.loc[:, 'Unnamed: 0']
x.loc[:, "Pr-1(House) load in kW"] = dh.loc[:, 'Building 1']
x.loc[:, "Pr-2(School) load in kW"] = dp.loc[:, 'School']
x.loc[:, "Pr-3(Zoo) load in kW"] = dp.loc[:, 'Zoo']
x.loc[:, "Pr-4(Gym) load in kW"] = dp.loc[:, 'Gym.1']
x.loc[:, "Pr-5(Event hall) load in kW"] = dp.loc[:, 'Event hall']
x.loc[:, "Temperature"] = dh.loc[:, 'Temperature']

# AKNN
# 01/01/2015 to 30/11/2016
ini_data = x[:16776]

# 30/11/2016
prev_day_data = x[16752:16776]

# 01/12/2016
last_day_data = x[16776:]

y_test_pr1_01_12 = last_day_data['Pr-1(House) load in kW']
y_test_pr2_01_12 = last_day_data['Pr-2(School) load in kW']
y_test_pr3_01_12 = last_day_data['Pr-3(Zoo) load in kW']
y_test_pr4_01_12 = last_day_data['Pr-4(Gym) load in kW']
y_test_pr5_01_12 = last_day_data['Pr-5(Event hall) load in kW']

load_pr1_30_11 = np.array(prev_day_data['Pr-1(House) load in kW'])
load_pr2_30_11 = np.array(prev_day_data['Pr-2(School) load in kW'])
load_pr3_30_11 = np.array(prev_day_data['Pr-3(Zoo) load in kW'])
load_pr4_30_11 = np.array(prev_day_data['Pr-4(Gym) load in kW'])
load_pr5_30_11 = np.array(prev_day_data['Pr-5(Event hall) load in kW'])

y_train_pr1 = np.array(ini_data['Pr-1(House) load in kW'])
y_train_pr2 = np.array(ini_data['Pr-2(School) load in kW'])
y_train_pr3 = np.array(ini_data['Pr-3(Zoo) load in kW'])
y_train_pr4 = np.array(ini_data['Pr-4(Gym) load in kW'])
y_train_pr5 = np.array(ini_data['Pr-5(Event hall) load in kW'])


chunks_pr1 = [y_train_pr1[x:x + 24] for x in range(0, len(y_train_pr1), 24)]
chunks_pr2 = [y_train_pr2[x:x + 24] for x in range(0, len(y_train_pr2), 24)]
chunks_pr3 = [y_train_pr3[x:x + 24] for x in range(0, len(y_train_pr3), 24)]
chunks_pr4 = [y_train_pr4[x:x + 24] for x in range(0, len(y_train_pr4), 24)]
chunks_pr5 = [y_train_pr5[x:x + 24] for x in range(0, len(y_train_pr5), 24)]

time_last_day = np.array(last_day_data['Date'])


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
    aknn_predicted_load = [aknn(load, chunks)]
    plot_values = []
    for pred in aknn_predicted_load:
        for l in pred:
            plot_values.append(l)
    return plot_values


# AKNN Pr-1
plot_values_pr1 = prediction(load_pr1_30_11, chunks_pr1)
continuous_graph(plot_values_pr1, y_test_pr1_01_12, 'Pr-1 power consumption in kW (AKNN)', 'hour')
step_graph(plot_values_pr1, y_test_pr1_01_12, 'Pr-1 power consumption in kW (AKNN)', 'hour')

mse_pr1_aknn = mean_squared_error(y_test_pr1_01_12, plot_values_pr1)
rmse_pr1_aknn = math.sqrt(mse_pr1_aknn)
mean_pr1 = np.mean(y_test_pr1_01_12)
nrmse_pr1_aknn = rmse_pr1_aknn / mean_pr1
print("MAE Pr-1 AKNN: {:.2f}" .format(mae(y_test_pr1_01_12, plot_values_pr1)))
print('NRMSE Pr-1 AKNN: {:.2f}'.format(nrmse_pr1_aknn), "\n")

# AKNN Pr-2
plot_values_pr2 = prediction(load_pr2_30_11, chunks_pr2)
continuous_graph(plot_values_pr2, y_test_pr2_01_12, 'Pr-2 power consumption in kW (AKNN)', 'hour')
step_graph(plot_values_pr2, y_test_pr2_01_12, 'Pr-2 power consumption in kW (AKNN)', 'hour')

mse_pr2_aknn = mean_squared_error(y_test_pr2_01_12, plot_values_pr2)
rmse_pr2_aknn = math.sqrt(mse_pr2_aknn)
mean_pr2 = np.mean(y_test_pr2_01_12)
nrmse_pr2_aknn = rmse_pr2_aknn / mean_pr2
print("MAE Pr-2 AKNN: {:.2f}" .format(mae(y_test_pr2_01_12, plot_values_pr2)))
print('NRMSE Pr-2 AKNN: {:.2f}'.format(nrmse_pr2_aknn), "\n")

# AKNN Pr-3
plot_values_pr3 = prediction(load_pr3_30_11, chunks_pr3)
continuous_graph(plot_values_pr3, y_test_pr3_01_12, 'Pr-3 power consumption in kW (AKNN)', 'hour')
step_graph(plot_values_pr3, y_test_pr3_01_12, 'Pr-3 power consumption in kW (AKNN)', 'hour')

mse_pr3_aknn = mean_squared_error(y_test_pr3_01_12, plot_values_pr3)
rmse_pr3_aknn = math.sqrt(mse_pr3_aknn)
mean_pr3 = np.mean(y_test_pr3_01_12)
nrmse_pr3_aknn = rmse_pr3_aknn / mean_pr3
print("MAE Pr-3 AKNN: {:.2f}" .format(mae(y_test_pr3_01_12, plot_values_pr3)))
print('NRMSE Pr-3 AKNN: {:.2f}'.format(nrmse_pr3_aknn), "\n")


# AKNN Pr-4
plot_values_pr4 = prediction(load_pr4_30_11, chunks_pr4)
continuous_graph(plot_values_pr4, y_test_pr4_01_12, 'Pr-4 power consumption in kW (AKNN)', 'hour')
step_graph(plot_values_pr4, y_test_pr4_01_12, 'Pr-4 power consumption in kW (AKNN)', 'hour')

mse_pr4_aknn = mean_squared_error(y_test_pr4_01_12, plot_values_pr4)
rmse_pr4_aknn = math.sqrt(mse_pr4_aknn)
mean_pr4 = np.mean(y_test_pr4_01_12)
nrmse_pr4_aknn = rmse_pr4_aknn / mean_pr4
print("MAE Pr-4 AKNN: {:.2f}" .format(mae(y_test_pr4_01_12, plot_values_pr4)))
print('NRMSE Pr-4 AKNN: {:.2f}'.format(nrmse_pr4_aknn), "\n")


# AKNN Pr-5
plot_values_pr5 = prediction(load_pr5_30_11, chunks_pr5)
continuous_graph(plot_values_pr5, y_test_pr5_01_12, 'Pr-5 power consumption in kW (AKNN)', 'hour')
step_graph(plot_values_pr5, y_test_pr5_01_12, 'Pr-5 power consumption in kW (AKNN)', 'hour')
mse_pr5_aknn = mean_squared_error(y_test_pr5_01_12, plot_values_pr5)
rmse_pr5_aknn = math.sqrt(mse_pr5_aknn)
mean_pr5 = np.mean(y_test_pr5_01_12)
nrmse_pr5_aknn = rmse_pr5_aknn / mean_pr5
print("MAE Pr-5 AKNN: {:.2f}" .format(mae(y_test_pr5_01_12, plot_values_pr5)))
print('NRMSE Pr-5 AKNN: {:.2f}'.format(nrmse_pr5_aknn), "\n")

x['Date'] = pd.to_datetime(x['Date'])
x["month_of_year"] = x['Date'].dt.month
x["hour_of_day"] = x['Date'].dt.hour
x["day_of_week"] = x['Date'].dt.dayofweek

X_pr1 = pd.DataFrame(index=x.index)
X_pr2 = pd.DataFrame(index=x.index)
X_pr3 = pd.DataFrame(index=x.index)
X_pr4 = pd.DataFrame(index=x.index)
X_pr5 = pd.DataFrame(index=x.index)

lags = [1, 2, 3, 4, 5, 6, 24, 48, 168]
for lag in lags:
    X_pr1.loc[:, "lag_"+str(lag)] = x["Pr-1(House) load in kW"].shift(lag)
for lag in lags:
    X_pr2.loc[:, "lag_"+str(lag)] = x["Pr-2(School) load in kW"].shift(lag)
for lag in lags:
    X_pr3.loc[:, "lag_"+str(lag)] = x["Pr-3(Zoo) load in kW"].shift(lag)
for lag in lags:
    X_pr4.loc[:, "lag_"+str(lag)] = x["Pr-4(Gym) load in kW"].shift(lag)
for lag in lags:
    X_pr5.loc[:, "lag_"+str(lag)] = x["Pr-5(Event hall) load in kW"].shift(lag)

X_pr1.loc[:, "HoD"] = x["hour_of_day"]
X_pr1.loc[:, "DoW"] = x["day_of_week"]
X_pr1.loc[:, "MoY"] = x["month_of_year"]
X_pr1.loc[:, "Temperature"] = x["Temperature"]

X_pr2.loc[:, "HoD"] = x["hour_of_day"]
X_pr2.loc[:, "DoW"] = x["day_of_week"]
X_pr2.loc[:, "MoY"] = x["month_of_year"]
X_pr2.loc[:, "Temperature"] = x["Temperature"]

X_pr3.loc[:, "HoD"] = x["hour_of_day"]
X_pr3.loc[:, "DoW"] = x["day_of_week"]
X_pr3.loc[:, "MoY"] = x["month_of_year"]
X_pr3.loc[:, "Temperature"] = x["Temperature"]

X_pr4.loc[:, "HoD"] = x["hour_of_day"]
X_pr4.loc[:, "DoW"] = x["day_of_week"]
X_pr4.loc[:, "MoY"] = x["month_of_year"]
X_pr4.loc[:, "Temperature"] = x["Temperature"]

X_pr5.loc[:, "HoD"] = x["hour_of_day"]
X_pr5.loc[:, "DoW"] = x["day_of_week"]
X_pr5.loc[:, "MoY"] = x["month_of_year"]
X_pr5.loc[:, "Temperature"] = x["Temperature"]


# One-hot encoding
X_pr1 = pd.get_dummies(X_pr1, columns=["HoD", "DoW", "MoY"])
X_pr2 = pd.get_dummies(X_pr2, columns=["HoD", "DoW", "MoY"])
X_pr3 = pd.get_dummies(X_pr3, columns=["HoD", "DoW", "MoY"])
X_pr4 = pd.get_dummies(X_pr4, columns=["HoD", "DoW", "MoY"])
X_pr5 = pd.get_dummies(X_pr5, columns=["HoD", "DoW", "MoY"])

Y_pr1 = x["Pr-1(House) load in kW"]
Y_pr2 = x["Pr-2(School) load in kW"]
Y_pr3 = x["Pr-3(Zoo) load in kW"]
Y_pr4 = x["Pr-4(Gym) load in kW"]
Y_pr5 = x["Pr-5(Event hall) load in kW"]


# Train/Test Split
X_train_pr1, X_test_pr1, y_train_pr1, y_test_pr1 = train_test_split(X_pr1.iloc[168:, :], Y_pr1.iloc[168:],
                                                                    test_size=0.0014, shuffle=False)
X_train_pr2, X_test_pr2, y_train_pr2, y_test_pr2 = train_test_split(X_pr2.iloc[168:, :], Y_pr2.iloc[168:],
                                                                    test_size=0.0014, shuffle=False)
X_train_pr3, X_test_pr3, y_train_pr3, y_test_pr3 = train_test_split(X_pr3.iloc[168:, :], Y_pr3.iloc[168:],
                                                                    test_size=0.0014, shuffle=False)
X_train_pr4, X_test_pr4, y_train_pr4, y_test_pr4 = train_test_split(X_pr4.iloc[168:, :], Y_pr4.iloc[168:],
                                                                    test_size=0.0014, shuffle=False)
X_train_pr5, X_test_pr5, y_train_pr5, y_test_pr5 = train_test_split(X_pr5.iloc[168:, :], Y_pr5.iloc[168:],
                                                                    test_size=0.0014, shuffle=False)

# KNN Pr1
knn_pr1 = KNeighborsRegressor()
knn_pr1.fit(X_train_pr1, y_train_pr1)

# Cross Validation Block
N = 4
# Tried with 3, 6, 9, 12, 15, 18, 24. Got best results with 6
knn_parameters = {"n_neighbors": [6]}

gs_cv_block_pr1 = GridSearchCV(knn_pr1, knn_parameters,
                              scoring="neg_mean_squared_error", cv=KFold(n_splits=N, shuffle=False),
                              n_jobs=-1, return_train_score=True)
gs_cv_block_pr1.fit(X_train_pr1, y_train_pr1)
print(gs_cv_block_pr1.best_params_)
y_hat_pr1 = gs_cv_block_pr1.predict(X_test_pr1)
print("Test RMSE Block: %.2f" % rmse(y_test_pr1, y_hat_pr1), "\n")


# Cross Validation Shuffle
gs_cv_shuffle_pr1 = GridSearchCV(knn_pr1, knn_parameters,
                                scoring="neg_mean_squared_error", cv=KFold(n_splits=N, shuffle=True),
                                n_jobs=-1, return_train_score=True)
gs_cv_shuffle_pr1.fit(X_train_pr1, y_train_pr1)
print(gs_cv_shuffle_pr1.best_params_)
y_hat_shuffle_pr1 = gs_cv_shuffle_pr1.predict(X_test_pr1)
print("Test RMSE Shuffle: %.2f" % rmse(y_test_pr1, y_hat_shuffle_pr1), "\n")


# Cross Validation Time series
gs_cv_ts_pr1 = GridSearchCV(knn_pr1, knn_parameters,
                           scoring="neg_mean_squared_error", cv=TimeSeriesSplit(n_splits=N),
                           n_jobs=-1, return_train_score=True)
gs_cv_ts_pr1.fit(X_train_pr1, y_train_pr1)
print(gs_cv_ts_pr1.best_params_)
y_hat_ts_pr1 = gs_cv_ts_pr1.predict(X_test_pr1)
print("Test RMSE Time Series: %.2f" % rmse(y_test_pr1, y_hat_ts_pr1), "\n")

knn_pr1 = KNeighborsRegressor(n_neighbors=6)
knn_pr1.fit(X_train_pr1, y_train_pr1)
y_pr1_hat = knn_pr1.predict(X_test_pr1)
continuous_graph(y_pr1_hat, y_test_pr1, 'Pr1 power consumption in kW (KNN)', 'hour')
step_graph(y_pr1_hat, y_test_pr1, 'Pr1 power consumption in kW (KNN)', 'hour')
print("MAE Pr1 KNN: {:.2f}" .format(mae(y_test_pr1, y_pr1_hat)))
mean_pr1 = np.mean(y_test_pr1)
nrmse_pr1_knn = rmse(y_test_pr1, y_pr1_hat) / mean_pr1
print("NRMSE Pr1 KNN: {:.2f}" .format(nrmse_pr1_knn), "\n")


# KNN Pr2
knn_pr2 = KNeighborsRegressor()
knn_pr2.fit(X_train_pr2, y_train_pr2)

# Cross Validation Block
gs_cv_block_pr2 = GridSearchCV(knn_pr2, knn_parameters,
                              scoring="neg_mean_squared_error", cv=KFold(n_splits=N, shuffle=False),
                              n_jobs=-1, return_train_score=True)
gs_cv_block_pr2.fit(X_train_pr2, y_train_pr2)
print(gs_cv_block_pr2.best_params_)
y_hat_pr2 = gs_cv_block_pr2.predict(X_test_pr2)
print("Test RMSE Block: %.2f" % rmse(y_test_pr2, y_hat_pr2), "\n")


# Cross Validation Shuffle
gs_cv_shuffle_pr2 = GridSearchCV(knn_pr2, knn_parameters,
                                scoring="neg_mean_squared_error", cv=KFold(n_splits=N, shuffle=True),
                                n_jobs=-1, return_train_score=True)
gs_cv_shuffle_pr2.fit(X_train_pr2, y_train_pr2)
print(gs_cv_shuffle_pr2.best_params_)
y_hat_shuffle_pr2 = gs_cv_shuffle_pr2.predict(X_test_pr2)
print("Test RMSE Shuffle: %.2f" % rmse(y_test_pr2, y_hat_shuffle_pr2), "\n")


# Cross Validation Time series
gs_cv_ts_pr2 = GridSearchCV(knn_pr2, knn_parameters,
                           scoring="neg_mean_squared_error", cv=TimeSeriesSplit(n_splits=N),
                           n_jobs=-1, return_train_score=True)
gs_cv_ts_pr2.fit(X_train_pr2, y_train_pr2)
print(gs_cv_ts_pr2.best_params_)
y_hat_ts_pr2 = gs_cv_ts_pr2.predict(X_test_pr2)
print("Test RMSE Time Series: %.2f" % rmse(y_test_pr2, y_hat_ts_pr2), "\n")

knn_pr2 = KNeighborsRegressor(n_neighbors=6)
knn_pr2.fit(X_train_pr2, y_train_pr2)
y_pr2_hat = knn_pr2.predict(X_test_pr2)
continuous_graph(y_pr2_hat, y_test_pr2, 'Pr2 power consumption in kW (KNN)', 'hour')
step_graph(y_pr2_hat, y_test_pr2, 'Pr2 power consumption in kW (KNN)', 'hour')
print("MAE Pr2 KNN: {:.2f}" .format(mae(y_test_pr2, y_pr2_hat)))
mean_pr2 = np.mean(y_test_pr2)
nrmse_pr2_knn = rmse(y_test_pr2, y_pr2_hat) / mean_pr2
print("NRMSE Pr2 KNN: {:.2f}" .format(nrmse_pr2_knn), "\n")


# KNN Pr3
knn_pr3 = KNeighborsRegressor()
knn_pr3.fit(X_train_pr3, y_train_pr3)

# Cross Validation Block
gs_cv_block_pr3 = GridSearchCV(knn_pr3, knn_parameters,
                              scoring="neg_mean_squared_error", cv=KFold(n_splits=N, shuffle=False),
                              n_jobs=-1, return_train_score=True)
gs_cv_block_pr3.fit(X_train_pr3, y_train_pr3)
print(gs_cv_block_pr3.best_params_)
y_hat_pr3 = gs_cv_block_pr3.predict(X_test_pr3)
print("Test RMSE Block: %.2f" % rmse(y_test_pr3, y_hat_pr3), "\n")


# Cross Validation Shuffle
gs_cv_shuffle_pr3 = GridSearchCV(knn_pr3, knn_parameters,
                                scoring="neg_mean_squared_error", cv=KFold(n_splits=N, shuffle=True),
                                n_jobs=-1, return_train_score=True)
gs_cv_shuffle_pr3.fit(X_train_pr3, y_train_pr3)
print(gs_cv_shuffle_pr3.best_params_)
y_hat_shuffle_pr3 = gs_cv_shuffle_pr3.predict(X_test_pr3)
print("Test RMSE Shuffle: %.2f" % rmse(y_test_pr3, y_hat_shuffle_pr3), "\n")


# Cross Validation Time series
gs_cv_ts_pr3 = GridSearchCV(knn_pr3, knn_parameters,
                           scoring="neg_mean_squared_error", cv=TimeSeriesSplit(n_splits=N),
                           n_jobs=-1, return_train_score=True)
gs_cv_ts_pr3.fit(X_train_pr3, y_train_pr3)
print(gs_cv_ts_pr3.best_params_)
y_hat_ts_pr3 = gs_cv_ts_pr3.predict(X_test_pr3)
print("Test RMSE Time Series: %.2f" % rmse(y_test_pr3, y_hat_ts_pr3), "\n")

knn_pr3 = KNeighborsRegressor(n_neighbors=6)
knn_pr3.fit(X_train_pr3, y_train_pr3)
y_pr3_hat = knn_pr3.predict(X_test_pr3)
continuous_graph(y_pr3_hat, y_test_pr3, 'Pr3 power consumption in kW (KNN)', 'hour')
step_graph(y_pr3_hat, y_test_pr3, 'Pr3 power consumption in kW (KNN)', 'hour')
print("MAE Pr3 KNN: {:.2f}" .format(mae(y_test_pr3, y_pr3_hat)))
mean_pr3 = np.mean(y_test_pr3)
nrmse_pr3_knn = rmse(y_test_pr3, y_pr3_hat) / mean_pr3
print("NRMSE Pr3 KNN: {:.2f}" .format(nrmse_pr3_knn), "\n")


# KNN Pr4
knn_pr4 = KNeighborsRegressor()
knn_pr4.fit(X_train_pr4, y_train_pr4)

# Cross Validation Block
gs_cv_block_pr4 = GridSearchCV(knn_pr4, knn_parameters,
                              scoring="neg_mean_squared_error", cv=KFold(n_splits=N, shuffle=False),
                              n_jobs=-1, return_train_score=True)
gs_cv_block_pr4.fit(X_train_pr4, y_train_pr4)
print(gs_cv_block_pr4.best_params_)
y_hat_pr4 = gs_cv_block_pr4.predict(X_test_pr4)
print("Test RMSE Block: %.2f" % rmse(y_test_pr4, y_hat_pr4), "\n")


# Cross Validation Shuffle
gs_cv_shuffle_pr4 = GridSearchCV(knn_pr4, knn_parameters,
                                scoring="neg_mean_squared_error", cv=KFold(n_splits=N, shuffle=True),
                                n_jobs=-1, return_train_score=True)
gs_cv_shuffle_pr4.fit(X_train_pr4, y_train_pr4)
print(gs_cv_shuffle_pr4.best_params_)
y_hat_shuffle_pr4 = gs_cv_shuffle_pr4.predict(X_test_pr4)
print("Test RMSE Shuffle: %.2f" % rmse(y_test_pr4, y_hat_shuffle_pr4), "\n")


# Cross Validation Time series
gs_cv_ts_pr4 = GridSearchCV(knn_pr4, knn_parameters,
                           scoring="neg_mean_squared_error", cv=TimeSeriesSplit(n_splits=N),
                           n_jobs=-1, return_train_score=True)
gs_cv_ts_pr4.fit(X_train_pr4, y_train_pr4)
print(gs_cv_ts_pr4.best_params_)
y_hat_ts_pr4 = gs_cv_ts_pr4.predict(X_test_pr4)
print("Test RMSE Time Series: %.2f" % rmse(y_test_pr4, y_hat_ts_pr4), "\n")

knn_pr4 = KNeighborsRegressor(n_neighbors=6)
knn_pr4.fit(X_train_pr4, y_train_pr4)
y_pr4_hat = knn_pr4.predict(X_test_pr4)
continuous_graph(y_pr4_hat, y_test_pr4, 'Pr4 power consumption in kW (KNN)', 'hour')
step_graph(y_pr4_hat, y_test_pr4, 'Pr4 power consumption in kW (KNN)', 'hour')
print("MAE Pr4 KNN: {:.2f}" .format(mae(y_test_pr4, y_pr4_hat)))
mean_pr4 = np.mean(y_test_pr4)
nrmse_pr4_knn = rmse(y_test_pr4, y_pr4_hat) / mean_pr4
print("NRMSE Pr4 KNN: {:.2f}" .format(nrmse_pr4_knn), "\n")


# KNN Pr5
knn_pr5 = KNeighborsRegressor()
knn_pr5.fit(X_train_pr5, y_train_pr5)

# Cross Validation Block
gs_cv_block_pr5 = GridSearchCV(knn_pr5, knn_parameters,
                              scoring="neg_mean_squared_error", cv=KFold(n_splits=N, shuffle=False),
                              n_jobs=-1, return_train_score=True)
gs_cv_block_pr5.fit(X_train_pr5, y_train_pr5)
print(gs_cv_block_pr5.best_params_)
y_hat_pr5 = gs_cv_block_pr5.predict(X_test_pr5)
print("Test RMSE Block: %.2f" % rmse(y_test_pr5, y_hat_pr5), "\n")


# Cross Validation Shuffle
gs_cv_shuffle_pr5 = GridSearchCV(knn_pr5, knn_parameters,
                                scoring="neg_mean_squared_error", cv=KFold(n_splits=N, shuffle=True),
                                n_jobs=-1, return_train_score=True)
gs_cv_shuffle_pr5.fit(X_train_pr5, y_train_pr5)
print(gs_cv_shuffle_pr5.best_params_)
y_hat_shuffle_pr5 = gs_cv_shuffle_pr5.predict(X_test_pr5)
print("Test RMSE Shuffle: %.2f" % rmse(y_test_pr5, y_hat_shuffle_pr5), "\n")


# Cross Validation Time series
gs_cv_ts_pr5 = GridSearchCV(knn_pr5, knn_parameters,
                           scoring="neg_mean_squared_error", cv=TimeSeriesSplit(n_splits=N),
                           n_jobs=-1, return_train_score=True)
gs_cv_ts_pr5.fit(X_train_pr5, y_train_pr5)
print(gs_cv_ts_pr5.best_params_)
y_hat_ts_pr5 = gs_cv_ts_pr5.predict(X_test_pr5)
print("Test RMSE Time Series: %.2f" % rmse(y_test_pr5, y_hat_ts_pr5), "\n")

knn_pr5 = KNeighborsRegressor(n_neighbors=6)
knn_pr5.fit(X_train_pr5, y_train_pr5)
y_pr5_hat = knn_pr5.predict(X_test_pr5)
continuous_graph(y_pr5_hat, y_test_pr5, 'Pr5 power consumption in kW (KNN)', 'hour')
step_graph(y_pr5_hat, y_test_pr5, 'Pr5 power consumption in kW (KNN)', 'hour')
print("MAE Pr5 KNN: {:.2f}" .format(mae(y_test_pr5, y_pr5_hat)))
mean_pr5 = np.mean(y_test_pr5)
nrmse_pr5_knn = rmse(y_test_pr5, y_pr5_hat) / mean_pr5
print("NRMSE Pr5 KNN: {:.2f}" .format(nrmse_pr5_knn), "\n")


# Ridge Pr1
r_pr1 = Ridge()
r_pr1.fit(X_train_pr1, y_train_pr1)

# Cross Validation Block
# Tried with 0.01, 0.1, 1.0. Got best results with 1.0
ridge_parameters = {"alpha" : [1.0]}
gs_cv_block_pr1_ridge = GridSearchCV(r_pr1, ridge_parameters,
                                    scoring="neg_mean_squared_error", cv=KFold(n_splits=N, shuffle=False),
                                    n_jobs=-1, return_train_score=True)
gs_cv_block_pr1_ridge.fit(X_train_pr1, y_train_pr1)
print(gs_cv_block_pr1_ridge.best_params_)
y_hat_pr1_ridge = gs_cv_block_pr1_ridge.predict(X_test_pr1)
print("Test RMSE Block: %.2f" % rmse(y_test_pr1, y_hat_pr1_ridge), "\n")


# Cross Validation Shuffle
gs_cv_shuffle_pr1_ridge = GridSearchCV(r_pr1, ridge_parameters,
                                      scoring="neg_mean_squared_error", cv=KFold(n_splits=N, shuffle=True),
                                      n_jobs=-1, return_train_score=True)
gs_cv_shuffle_pr1_ridge.fit(X_train_pr1, y_train_pr1)
print(gs_cv_shuffle_pr1_ridge.best_params_)
y_hat_shuffle_pr1_ridge = gs_cv_shuffle_pr1_ridge.predict(X_test_pr1)
print("Test RMSE Shuffle: %.2f" % rmse(y_test_pr1, y_hat_shuffle_pr1_ridge), "\n")


# Cross Validation Time series
gs_cv_ts_pr1_ridge = GridSearchCV(r_pr1, ridge_parameters,
                                 scoring="neg_mean_squared_error", cv=TimeSeriesSplit(n_splits=N),
                                 n_jobs=-1, return_train_score=True)
gs_cv_ts_pr1_ridge.fit(X_train_pr1, y_train_pr1)
print(gs_cv_ts_pr1_ridge.best_params_)
y_hat_ts_pr1_ridge = gs_cv_ts_pr1_ridge.predict(X_test_pr1)
print("Test RMSE Time Series: %.2f" % rmse(y_test_pr1, y_hat_ts_pr1_ridge), "\n")


r_pr1 = Ridge(alpha=1.0)
r_pr1.fit(X_train_pr1, y_train_pr1)
r_pr1_hat = r_pr1.predict(X_test_pr1)
continuous_graph(r_pr1_hat, y_test_pr1, 'Pr1 power consumption in kW (Ridge)', 'hour')
step_graph(r_pr1_hat, y_test_pr1, 'Pr1 power consumption in kW (Ridge)', 'hour')
print("MAE Pr1 Ridge: {:.2f}" .format(mae(y_test_pr1, r_pr1_hat)))
nrmse_pr1_ridge = rmse(y_test_pr1, r_pr1_hat) / mean_pr1
print("NRMSE Pr1 Ridge: {:.2f}" .format(nrmse_pr1_ridge), "\n")


# Ridge Pr2
r_pr2 = Ridge()
r_pr2.fit(X_train_pr2, y_train_pr2)

# Cross Validation Block
gs_cv_block_pr2_ridge = GridSearchCV(r_pr2, ridge_parameters,
                                    scoring="neg_mean_squared_error", cv=KFold(n_splits=N, shuffle=False),
                                    n_jobs=-1, return_train_score=True)
gs_cv_block_pr2_ridge.fit(X_train_pr2, y_train_pr2)
print(gs_cv_block_pr2_ridge.best_params_)
y_hat_pr2_ridge = gs_cv_block_pr2_ridge.predict(X_test_pr2)
print("Test RMSE Block: %.2f" % rmse(y_test_pr2, y_hat_pr2_ridge), "\n")


# Cross Validation Shuffle
gs_cv_shuffle_pr2_ridge = GridSearchCV(r_pr2, ridge_parameters,
                                      scoring="neg_mean_squared_error", cv=KFold(n_splits=N, shuffle=True),
                                      n_jobs=-1, return_train_score=True)
gs_cv_shuffle_pr2_ridge.fit(X_train_pr2, y_train_pr2)
print(gs_cv_shuffle_pr2_ridge.best_params_)
y_hat_shuffle_pr2_ridge = gs_cv_shuffle_pr2_ridge.predict(X_test_pr2)
print("Test RMSE Shuffle: %.2f" % rmse(y_test_pr2, y_hat_shuffle_pr2_ridge), "\n")


# Cross Validation Time series
gs_cv_ts_pr2_ridge = GridSearchCV(r_pr2, ridge_parameters,
                                 scoring="neg_mean_squared_error", cv=TimeSeriesSplit(n_splits=N),
                                 n_jobs=-1, return_train_score=True)
gs_cv_ts_pr2_ridge.fit(X_train_pr2, y_train_pr2)
print(gs_cv_ts_pr2_ridge.best_params_)
y_hat_ts_pr2_ridge = gs_cv_ts_pr2_ridge.predict(X_test_pr2)
print("Test RMSE Time Series: %.2f" % rmse(y_test_pr2, y_hat_ts_pr2_ridge), "\n")


r_pr2 = Ridge(alpha=1.0)
r_pr2.fit(X_train_pr2, y_train_pr2)
r_pr2_hat = r_pr2.predict(X_test_pr2)
continuous_graph(r_pr2_hat, y_test_pr2, 'Pr2 power consumption in kW (Ridge)', 'hour')
step_graph(r_pr2_hat, y_test_pr2, 'Pr2 power consumption in kW (Ridge)', 'hour')
print("MAE Pr2 Ridge: {:.2f}" .format(mae(y_test_pr2, r_pr2_hat)))
nrmse_pr2_ridge = rmse(y_test_pr2, r_pr2_hat) / mean_pr2
print("NRMSE Pr2 Ridge: {:.2f}" .format(nrmse_pr2_ridge), "\n")


# Ridge Pr3
r_pr3 = Ridge()
r_pr3.fit(X_train_pr3, y_train_pr3)

# Cross Validation Block
gs_cv_block_pr3_ridge = GridSearchCV(r_pr3, ridge_parameters,
                                    scoring="neg_mean_squared_error", cv=KFold(n_splits=N, shuffle=False),
                                    n_jobs=-1, return_train_score=True)
gs_cv_block_pr3_ridge.fit(X_train_pr3, y_train_pr3)
print(gs_cv_block_pr3_ridge.best_params_)
y_hat_pr3_ridge = gs_cv_block_pr3_ridge.predict(X_test_pr3)
print("Test RMSE Block: %.2f" % rmse(y_test_pr3, y_hat_pr3_ridge), "\n")


# Cross Validation Shuffle
gs_cv_shuffle_pr3_ridge = GridSearchCV(r_pr3, ridge_parameters,
                                      scoring="neg_mean_squared_error", cv=KFold(n_splits=N, shuffle=True),
                                      n_jobs=-1, return_train_score=True)
gs_cv_shuffle_pr3_ridge.fit(X_train_pr3, y_train_pr3)
print(gs_cv_shuffle_pr3_ridge.best_params_)
y_hat_shuffle_pr3_ridge = gs_cv_shuffle_pr3_ridge.predict(X_test_pr3)
print("Test RMSE Shuffle: %.2f" % rmse(y_test_pr3, y_hat_shuffle_pr3_ridge), "\n")


# Cross Validation Time series
gs_cv_ts_pr3_ridge = GridSearchCV(r_pr3, ridge_parameters,
                                 scoring="neg_mean_squared_error", cv=TimeSeriesSplit(n_splits=N),
                                 n_jobs=-1, return_train_score=True)
gs_cv_ts_pr3_ridge.fit(X_train_pr3, y_train_pr3)
print(gs_cv_ts_pr3_ridge.best_params_)
y_hat_ts_pr3_ridge = gs_cv_ts_pr3_ridge.predict(X_test_pr3)
print("Test RMSE Time Series: %.2f" % rmse(y_test_pr3, y_hat_ts_pr3_ridge), "\n")


r_pr3 = Ridge(alpha=1.0)
r_pr3.fit(X_train_pr3, y_train_pr3)
r_pr3_hat = r_pr3.predict(X_test_pr3)
continuous_graph(r_pr3_hat, y_test_pr3, 'Pr3 power consumption in kW (Ridge)', 'hour')
step_graph(r_pr3_hat, y_test_pr3, 'Pr3 power consumption in kW (Ridge)', 'hour')
print("MAE Pr3 Ridge: {:.2f}" .format(mae(y_test_pr3, r_pr3_hat)))
nrmse_pr3_ridge = rmse(y_test_pr3, r_pr3_hat) / mean_pr3
print("NRMSE Pr3 Ridge: {:.2f}" .format(nrmse_pr3_ridge), "\n")


# Ridge Pr4
r_pr4 = Ridge()
r_pr4.fit(X_train_pr4, y_train_pr4)

# Cross Validation Block
gs_cv_block_pr4_ridge = GridSearchCV(r_pr4, ridge_parameters,
                                    scoring="neg_mean_squared_error", cv=KFold(n_splits=N, shuffle=False),
                                    n_jobs=-1, return_train_score=True)
gs_cv_block_pr4_ridge.fit(X_train_pr4, y_train_pr4)
print(gs_cv_block_pr4_ridge.best_params_)
y_hat_pr4_ridge = gs_cv_block_pr4_ridge.predict(X_test_pr4)
print("Test RMSE Block: %.2f" % rmse(y_test_pr4, y_hat_pr4_ridge), "\n")


# Cross Validation Shuffle
gs_cv_shuffle_pr4_ridge = GridSearchCV(r_pr4, ridge_parameters,
                                      scoring="neg_mean_squared_error", cv=KFold(n_splits=N, shuffle=True),
                                      n_jobs=-1, return_train_score=True)
gs_cv_shuffle_pr4_ridge.fit(X_train_pr4, y_train_pr4)
print(gs_cv_shuffle_pr4_ridge.best_params_)
y_hat_shuffle_pr4_ridge = gs_cv_shuffle_pr4_ridge.predict(X_test_pr4)
print("Test RMSE Shuffle: %.2f" % rmse(y_test_pr4, y_hat_shuffle_pr4_ridge), "\n")


# Cross Validation Time series
gs_cv_ts_pr4_ridge = GridSearchCV(r_pr4, ridge_parameters,
                                 scoring="neg_mean_squared_error", cv=TimeSeriesSplit(n_splits=N),
                                 n_jobs=-1, return_train_score=True)
gs_cv_ts_pr4_ridge.fit(X_train_pr4, y_train_pr4)
print(gs_cv_ts_pr4_ridge.best_params_)
y_hat_ts_pr4_ridge = gs_cv_ts_pr4_ridge.predict(X_test_pr4)
print("Test RMSE Time Series: %.2f" % rmse(y_test_pr4, y_hat_ts_pr4_ridge), "\n")


r_pr4 = Ridge(alpha=1.0)
r_pr4.fit(X_train_pr4, y_train_pr4)
r_pr4_hat = r_pr4.predict(X_test_pr4)
continuous_graph(r_pr4_hat, y_test_pr4, 'Pr4 power consumption in kW (Ridge)', 'hour')
step_graph(r_pr4_hat, y_test_pr4, 'Pr4 power consumption in kW (Ridge)', 'hour')
print("MAE Pr4 Ridge: {:.2f}" .format(mae(y_test_pr4, r_pr4_hat)))
nrmse_pr4_ridge = rmse(y_test_pr4, r_pr4_hat) / mean_pr4
print("NRMSE Pr4 Ridge: {:.2f}" .format(nrmse_pr4_ridge), "\n")


# Ridge Pr5
r_pr5 = Ridge()
r_pr5.fit(X_train_pr5, y_train_pr5)

# Cross Validation Block
gs_cv_block_pr5_ridge = GridSearchCV(r_pr5, ridge_parameters,
                                    scoring="neg_mean_squared_error", cv=KFold(n_splits=N, shuffle=False),
                                    n_jobs=-1, return_train_score=True)
gs_cv_block_pr5_ridge.fit(X_train_pr5, y_train_pr5)
print(gs_cv_block_pr5_ridge.best_params_)
y_hat_pr5_ridge = gs_cv_block_pr5_ridge.predict(X_test_pr5)
print("Test RMSE Block: %.2f" % rmse(y_test_pr5, y_hat_pr5_ridge), "\n")


# Cross Validation Shuffle
gs_cv_shuffle_pr5_ridge = GridSearchCV(r_pr5, ridge_parameters,
                                      scoring="neg_mean_squared_error", cv=KFold(n_splits=N, shuffle=True),
                                      n_jobs=-1, return_train_score=True)
gs_cv_shuffle_pr5_ridge.fit(X_train_pr5, y_train_pr5)
print(gs_cv_shuffle_pr5_ridge.best_params_)
y_hat_shuffle_pr5_ridge = gs_cv_shuffle_pr5_ridge.predict(X_test_pr5)
print("Test RMSE Shuffle: %.2f" % rmse(y_test_pr5, y_hat_shuffle_pr5_ridge), "\n")


# Cross Validation Time series
gs_cv_ts_pr5_ridge = GridSearchCV(r_pr5, ridge_parameters,
                                 scoring="neg_mean_squared_error", cv=TimeSeriesSplit(n_splits=N),
                                 n_jobs=-1, return_train_score=True)
gs_cv_ts_pr5_ridge.fit(X_train_pr5, y_train_pr5)
print(gs_cv_ts_pr5_ridge.best_params_)
y_hat_ts_pr5_ridge = gs_cv_ts_pr5_ridge.predict(X_test_pr5)
print("Test RMSE Time Series: %.2f" % rmse(y_test_pr5, y_hat_ts_pr5_ridge), "\n")


r_pr5 = Ridge(alpha=1.0)
r_pr5.fit(X_train_pr5, y_train_pr5)
r_pr5_hat = r_pr5.predict(X_test_pr5)
continuous_graph(r_pr5_hat, y_test_pr5, 'Pr5 power consumption in kW (Ridge)', 'hour')
step_graph(r_pr5_hat, y_test_pr5, 'Pr5 power consumption in kW (Ridge)', 'hour')
print("MAE Pr5 Ridge: {:.2f}" .format(mae(y_test_pr5, r_pr5_hat)))
nrmse_pr5_ridge = rmse(y_test_pr5, r_pr5_hat) / mean_pr5
print("NRMSE Pr5 Ridge: {:.2f}" .format(nrmse_pr5_ridge), "\n")


# Lasso Pr1
lasso_pr1 = Lasso()
lasso_pr1.fit(X_train_pr1, y_train_pr1)

# Cross Validation Block
# Tried with 0.01, 0.1, 1.0. Got best results with 0.01
lasso_parameters = {"alpha" : [0.01]}
gs_cv_block_pr1_lasso = GridSearchCV(lasso_pr1, lasso_parameters,
                                    scoring="neg_mean_squared_error", cv=KFold(n_splits=N, shuffle=False),
                                    n_jobs=-1, return_train_score=True)
gs_cv_block_pr1_lasso.fit(X_train_pr1, y_train_pr1)
print(gs_cv_block_pr1_lasso.best_params_)
y_hat_pr1_lasso = gs_cv_block_pr1_lasso.predict(X_test_pr1)
print("Test RMSE Block: %.2f" % rmse(y_test_pr1, y_hat_pr1_lasso), "\n")


# Cross Validation Shuffle
gs_cv_shuffle_pr1_lasso = GridSearchCV(lasso_pr1, lasso_parameters,
                                      scoring="neg_mean_squared_error", cv=KFold(n_splits=N, shuffle=True),
                                      n_jobs=-1, return_train_score=True)
gs_cv_shuffle_pr1_lasso.fit(X_train_pr1, y_train_pr1)
print(gs_cv_shuffle_pr1_lasso.best_params_)
y_hat_shuffle_pr1_lasso = gs_cv_shuffle_pr1_lasso.predict(X_test_pr1)
print("Test RMSE Shuffle: %.2f" % rmse(y_test_pr1, y_hat_shuffle_pr1_lasso), "\n")


# Cross Validation Time series
gs_cv_ts_pr1_lasso = GridSearchCV(lasso_pr1, lasso_parameters,
                                 scoring="neg_mean_squared_error", cv=TimeSeriesSplit(n_splits=N),
                                 n_jobs=-1, return_train_score=True)
gs_cv_ts_pr1_lasso.fit(X_train_pr1, y_train_pr1)
print(gs_cv_ts_pr1_lasso.best_params_)
y_hat_ts_pr1_lasso = gs_cv_ts_pr1_lasso.predict(X_test_pr1)
print("Test RMSE Time Series: %.2f" % rmse(y_test_pr1, y_hat_ts_pr1_lasso), "\n")

# Lasso Pr1
lasso_pr1 = Lasso(alpha=0.01)
lasso_pr1.fit(X_train_pr1, y_train_pr1)
lasso_pr1_hat = lasso_pr1.predict(X_test_pr1)
continuous_graph(lasso_pr1_hat, y_test_pr1, 'Pr1 power consumption in kW (Lasso)', 'hour')
step_graph(lasso_pr1_hat, y_test_pr1, 'Pr1 power consumption in kW (Lasso)', 'hour')
print("MAE Pr1 Lasso: {:.2f}" .format(mae(y_test_pr1, lasso_pr1_hat)))
nrmse_pr1_lasso = rmse(y_test_pr1, lasso_pr1_hat) / mean_pr1
print("NRMSE Pr1 Lasso: {:.2f}" .format(nrmse_pr1_lasso), "\n")


# Lasso Pr2
lasso_pr2 = Lasso()
lasso_pr2.fit(X_train_pr2, y_train_pr2)

# Cross Validation Block
gs_cv_block_pr2_lasso = GridSearchCV(lasso_pr2, lasso_parameters,
                                    scoring="neg_mean_squared_error", cv=KFold(n_splits=N, shuffle=False),
                                    n_jobs=-1, return_train_score=True)
gs_cv_block_pr2_lasso.fit(X_train_pr2, y_train_pr2)
print(gs_cv_block_pr2_lasso.best_params_)
y_hat_pr2_lasso = gs_cv_block_pr2_lasso.predict(X_test_pr2)
print("Test RMSE Block: %.2f" % rmse(y_test_pr2, y_hat_pr2_lasso), "\n")


# Cross Validation Shuffle
gs_cv_shuffle_pr2_lasso = GridSearchCV(lasso_pr2, lasso_parameters,
                                      scoring="neg_mean_squared_error", cv=KFold(n_splits=N, shuffle=True),
                                      n_jobs=-1, return_train_score=True)
gs_cv_shuffle_pr2_lasso.fit(X_train_pr2, y_train_pr2)
print(gs_cv_shuffle_pr2_lasso.best_params_)
y_hat_shuffle_pr2_lasso = gs_cv_shuffle_pr2_lasso.predict(X_test_pr2)
print("Test RMSE Shuffle: %.2f" % rmse(y_test_pr2, y_hat_shuffle_pr2_lasso), "\n")


# Cross Validation Time series
gs_cv_ts_pr2_lasso = GridSearchCV(lasso_pr2, lasso_parameters,
                                 scoring="neg_mean_squared_error", cv=TimeSeriesSplit(n_splits=N),
                                 n_jobs=-1, return_train_score=True)
gs_cv_ts_pr2_lasso.fit(X_train_pr2, y_train_pr2)
print(gs_cv_ts_pr2_lasso.best_params_)
y_hat_ts_pr2_lasso = gs_cv_ts_pr2_lasso.predict(X_test_pr2)
print("Test RMSE Time Series: %.2f" % rmse(y_test_pr2, y_hat_ts_pr2_lasso), "\n")

# Lasso Pr2
lasso_pr2 = Lasso(alpha=0.01)
lasso_pr2.fit(X_train_pr2, y_train_pr2)
lasso_pr2_hat = lasso_pr2.predict(X_test_pr2)
continuous_graph(lasso_pr2_hat, y_test_pr2, 'Pr2 power consumption in kW (Lasso)', 'hour')
step_graph(lasso_pr2_hat, y_test_pr2, 'Pr2 power consumption in kW (Lasso)', 'hour')
print("MAE Pr2 Lasso: {:.2f}" .format(mae(y_test_pr2, lasso_pr2_hat)))
nrmse_pr2_lasso = rmse(y_test_pr2, lasso_pr2_hat) / mean_pr2
print("NRMSE Pr2 Lasso: {:.2f}" .format(nrmse_pr2_lasso), "\n")


# Lasso Pr3
lasso_pr3 = Lasso()
lasso_pr3.fit(X_train_pr3, y_train_pr3)

# Cross Validation Block
gs_cv_block_pr3_lasso = GridSearchCV(lasso_pr3, lasso_parameters,
                                    scoring="neg_mean_squared_error", cv=KFold(n_splits=N, shuffle=False),
                                    n_jobs=-1, return_train_score=True)
gs_cv_block_pr3_lasso.fit(X_train_pr3, y_train_pr3)
print(gs_cv_block_pr3_lasso.best_params_)
y_hat_pr3_lasso = gs_cv_block_pr3_lasso.predict(X_test_pr3)
print("Test RMSE Block: %.2f" % rmse(y_test_pr3, y_hat_pr3_lasso), "\n")


# Cross Validation Shuffle
gs_cv_shuffle_pr3_lasso = GridSearchCV(lasso_pr3, lasso_parameters,
                                      scoring="neg_mean_squared_error", cv=KFold(n_splits=N, shuffle=True),
                                      n_jobs=-1, return_train_score=True)
gs_cv_shuffle_pr3_lasso.fit(X_train_pr3, y_train_pr3)
print(gs_cv_shuffle_pr3_lasso.best_params_)
y_hat_shuffle_pr3_lasso = gs_cv_shuffle_pr3_lasso.predict(X_test_pr3)
print("Test RMSE Shuffle: %.2f" % rmse(y_test_pr3, y_hat_shuffle_pr3_lasso), "\n")


# Cross Validation Time series
gs_cv_ts_pr3_lasso = GridSearchCV(lasso_pr3, lasso_parameters,
                                 scoring="neg_mean_squared_error", cv=TimeSeriesSplit(n_splits=N),
                                 n_jobs=-1, return_train_score=True)
gs_cv_ts_pr3_lasso.fit(X_train_pr3, y_train_pr3)
print(gs_cv_ts_pr3_lasso.best_params_)
y_hat_ts_pr3_lasso = gs_cv_ts_pr3_lasso.predict(X_test_pr3)
print("Test RMSE Time Series: %.2f" % rmse(y_test_pr3, y_hat_ts_pr3_lasso), "\n")

# Lasso Pr3
lasso_pr3 = Lasso(alpha=0.01)
lasso_pr3.fit(X_train_pr3, y_train_pr3)
lasso_pr3_hat = lasso_pr3.predict(X_test_pr3)
continuous_graph(lasso_pr3_hat, y_test_pr3, 'Pr3 power consumption in kW (Lasso)', 'hour')
step_graph(lasso_pr3_hat, y_test_pr3, 'Pr3 power consumption in kW (Lasso)', 'hour')
print("MAE Pr3 Lasso: {:.2f}" .format(mae(y_test_pr3, lasso_pr3_hat)))
nrmse_pr3_lasso = rmse(y_test_pr3, lasso_pr3_hat) / mean_pr3
print("NRMSE Pr3 Lasso: {:.2f}" .format(nrmse_pr3_lasso), "\n")


# Lasso Pr4
lasso_pr4 = Lasso()
lasso_pr4.fit(X_train_pr4, y_train_pr4)

# Cross Validation Block
gs_cv_block_pr4_lasso = GridSearchCV(lasso_pr4, lasso_parameters,
                                    scoring="neg_mean_squared_error", cv=KFold(n_splits=N, shuffle=False),
                                    n_jobs=-1, return_train_score=True)
gs_cv_block_pr4_lasso.fit(X_train_pr4, y_train_pr4)
print(gs_cv_block_pr4_lasso.best_params_)
y_hat_pr4_lasso = gs_cv_block_pr4_lasso.predict(X_test_pr4)
print("Test RMSE Block: %.2f" % rmse(y_test_pr4, y_hat_pr4_lasso), "\n")


# Cross Validation Shuffle
gs_cv_shuffle_pr4_lasso = GridSearchCV(lasso_pr4, lasso_parameters,
                                      scoring="neg_mean_squared_error", cv=KFold(n_splits=N, shuffle=True),
                                      n_jobs=-1, return_train_score=True)
gs_cv_shuffle_pr4_lasso.fit(X_train_pr4, y_train_pr4)
print(gs_cv_shuffle_pr4_lasso.best_params_)
y_hat_shuffle_pr4_lasso = gs_cv_shuffle_pr4_lasso.predict(X_test_pr4)
print("Test RMSE Shuffle: %.2f" % rmse(y_test_pr4, y_hat_shuffle_pr4_lasso), "\n")


# Cross Validation Time series
gs_cv_ts_pr4_lasso = GridSearchCV(lasso_pr4, lasso_parameters,
                                 scoring="neg_mean_squared_error", cv=TimeSeriesSplit(n_splits=N),
                                 n_jobs=-1, return_train_score=True)
gs_cv_ts_pr4_lasso.fit(X_train_pr4, y_train_pr4)
print(gs_cv_ts_pr4_lasso.best_params_)
y_hat_ts_pr4_lasso = gs_cv_ts_pr4_lasso.predict(X_test_pr4)
print("Test RMSE Time Series: %.2f" % rmse(y_test_pr4, y_hat_ts_pr4_lasso), "\n")

# Lasso Pr4
lasso_pr4 = Lasso(alpha=0.01)
lasso_pr4.fit(X_train_pr4, y_train_pr4)
lasso_pr4_hat = lasso_pr4.predict(X_test_pr4)
continuous_graph(lasso_pr4_hat, y_test_pr4, 'Pr4 power consumption in kW (Lasso)', 'hour')
step_graph(lasso_pr4_hat, y_test_pr4, 'Pr4 power consumption in kW (Lasso)', 'hour')
print("MAE Pr4 Lasso: {:.2f}" .format(mae(y_test_pr4, lasso_pr4_hat)))
nrmse_pr4_lasso = rmse(y_test_pr4, lasso_pr4_hat) / mean_pr4
print("NRMSE Pr4 Lasso: {:.2f}" .format(nrmse_pr4_lasso), "\n")


# Lasso Pr5
lasso_pr5 = Lasso()
lasso_pr5.fit(X_train_pr5, y_train_pr5)

# Cross Validation Block
gs_cv_block_pr5_lasso = GridSearchCV(lasso_pr5, lasso_parameters,
                                    scoring="neg_mean_squared_error", cv=KFold(n_splits=N, shuffle=False),
                                    n_jobs=-1, return_train_score=True)
gs_cv_block_pr5_lasso.fit(X_train_pr5, y_train_pr5)
print(gs_cv_block_pr5_lasso.best_params_)
y_hat_pr5_lasso = gs_cv_block_pr5_lasso.predict(X_test_pr5)
print("Test RMSE Block: %.2f" % rmse(y_test_pr5, y_hat_pr5_lasso), "\n")


# Cross Validation Shuffle
gs_cv_shuffle_pr5_lasso = GridSearchCV(lasso_pr5, lasso_parameters,
                                      scoring="neg_mean_squared_error", cv=KFold(n_splits=N, shuffle=True),
                                      n_jobs=-1, return_train_score=True)
gs_cv_shuffle_pr5_lasso.fit(X_train_pr5, y_train_pr5)
print(gs_cv_shuffle_pr5_lasso.best_params_)
y_hat_shuffle_pr5_lasso = gs_cv_shuffle_pr5_lasso.predict(X_test_pr5)
print("Test RMSE Shuffle: %.2f" % rmse(y_test_pr5, y_hat_shuffle_pr5_lasso), "\n")


# Cross Validation Time series
gs_cv_ts_pr5_lasso = GridSearchCV(lasso_pr5, lasso_parameters,
                                 scoring="neg_mean_squared_error", cv=TimeSeriesSplit(n_splits=N),
                                 n_jobs=-1, return_train_score=True)
gs_cv_ts_pr5_lasso.fit(X_train_pr5, y_train_pr5)
print(gs_cv_ts_pr5_lasso.best_params_)
y_hat_ts_pr5_lasso = gs_cv_ts_pr5_lasso.predict(X_test_pr5)
print("Test RMSE Time Series: %.2f" % rmse(y_test_pr5, y_hat_ts_pr5_lasso), "\n")

# Lasso Pr5
lasso_pr5 = Lasso(alpha=0.01)
lasso_pr5.fit(X_train_pr5, y_train_pr5)
lasso_pr5_hat = lasso_pr5.predict(X_test_pr5)
continuous_graph(lasso_pr5_hat, y_test_pr5, 'Pr5 power consumption in kW (Lasso)', 'hour')
step_graph(lasso_pr5_hat, y_test_pr5, 'Pr5 power consumption in kW (Lasso)', 'hour')
print("MAE Pr5 Lasso: {:.2f}" .format(mae(y_test_pr5, lasso_pr5_hat)))
nrmse_pr5_lasso = rmse(y_test_pr5, lasso_pr5_hat) / mean_pr5
print("NRMSE Pr5 Lasso: {:.2f}" .format(nrmse_pr5_lasso), "\n")


# Linear regression Pr1
lr_pr1 = LinearRegression()
lr_pr1.fit(X_train_pr1, y_train_pr1)
lr_pr1_hat = lr_pr1.predict(X_test_pr1)
continuous_graph(lr_pr1_hat, y_test_pr1, 'Pr1 power consumption in kW (LR)', 'hour')
step_graph(lr_pr1_hat, y_test_pr1, 'Pr1 power consumption in kW (LR)', 'hour')
print("MAE Pr1 LR: {:.2f}" .format(mae(y_test_pr1, lr_pr1_hat)))
nrmse_pr1_lr = rmse(y_test_pr1, lr_pr1_hat) / mean_pr1
print("NRMSE Pr1 LR: {:.2f}" .format(nrmse_pr1_lr), "\n")


# Linear regression Pr2
lr_pr2 = LinearRegression()
lr_pr2.fit(X_train_pr2, y_train_pr2)
lr_pr2_hat = lr_pr2.predict(X_test_pr2)
continuous_graph(lr_pr2_hat, y_test_pr2, 'Pr2 power consumption in kW (LR)', 'hour')
step_graph(lr_pr2_hat, y_test_pr2, 'Pr2 power consumption in kW (LR)', 'hour')
print("MAE Pr2 LR: {:.2f}" .format(mae(y_test_pr2, lr_pr2_hat)))
nrmse_pr2_lr = rmse(y_test_pr2, lr_pr2_hat) / mean_pr2
print("NRMSE Pr2 LR: {:.2f}" .format(nrmse_pr2_lr), "\n")


# Linear regression Pr3
lr_pr3 = LinearRegression()
lr_pr3.fit(X_train_pr3, y_train_pr3)
lr_pr3_hat = lr_pr3.predict(X_test_pr3)
continuous_graph(lr_pr3_hat, y_test_pr3, 'Pr3 power consumption in kW (LR)', 'hour')
step_graph(lr_pr3_hat, y_test_pr3, 'Pr3 power consumption in kW (LR)', 'hour')
print("MAE Pr3 LR: {:.2f}" .format(mae(y_test_pr3, lr_pr3_hat)))
nrmse_pr3_lr = rmse(y_test_pr3, lr_pr3_hat) / mean_pr3
print("NRMSE Pr3 LR: {:.2f}" .format(nrmse_pr3_lr), "\n")


# Linear regression Pr4
lr_pr4 = LinearRegression()
lr_pr4.fit(X_train_pr4, y_train_pr4)
lr_pr4_hat = lr_pr4.predict(X_test_pr4)
continuous_graph(lr_pr4_hat, y_test_pr4, 'Pr4 power consumption in kW (LR)', 'hour')
step_graph(lr_pr4_hat, y_test_pr4, 'Pr4 power consumption in kW (LR)', 'hour')
print("MAE Pr4 LR: {:.2f}" .format(mae(y_test_pr4, lr_pr4_hat)))
nrmse_pr4_lr = rmse(y_test_pr4, lr_pr4_hat) / mean_pr4
print("NRMSE Pr4 LR: {:.2f}" .format(nrmse_pr4_lr), "\n")


# Linear regression Pr5
lr_pr5 = LinearRegression()
lr_pr5.fit(X_train_pr5, y_train_pr5)
lr_pr5_hat = lr_pr5.predict(X_test_pr5)
continuous_graph(lr_pr5_hat, y_test_pr5, 'Pr5 power consumption in kW (LR)', 'hour')
step_graph(lr_pr5_hat, y_test_pr5, 'Pr5 power consumption in kW (LR)', 'hour')
print("MAE Pr5 LR: {:.2f}" .format(mae(y_test_pr5, lr_pr5_hat)))
nrmse_pr5_lr = rmse(y_test_pr5, lr_pr5_hat) / mean_pr5
print("NRMSE Pr5 LR: {:.2f}" .format(nrmse_pr5_lr), "\n")


# MLP Pr1
mlp_pr1 = MLPRegressor()
mlp_pr1.fit(X_train_pr1, y_train_pr1)
m_pr1_hat = mlp_pr1.predict(X_test_pr1)
continuous_graph(m_pr1_hat, y_test_pr1, 'Pr1 power consumption in kW (MLP)', 'hour')
step_graph(m_pr1_hat, y_test_pr1, 'Pr1 power consumption in kW (MLP)', 'hour')
print("MAE Pr1 MLP: {:.2f}" .format(mae(y_test_pr1, m_pr1_hat)))
nrmse_pr1_mlp = rmse(y_test_pr1, m_pr1_hat) / mean_pr1
print("NRMSE Pr1 MLP: {:.2f}" .format(nrmse_pr1_mlp), "\n")

# MLP Pr2
mlp_pr2 = MLPRegressor()
mlp_pr2.fit(X_train_pr2, y_train_pr2)
m_pr2_hat = mlp_pr2.predict(X_test_pr2)
continuous_graph(m_pr2_hat, y_test_pr2, 'Pr2 power consumption in kW (MLP)', 'hour')
step_graph(m_pr2_hat, y_test_pr2, 'Pr2 power consumption in kW (MLP)', 'hour')
print("MAE Pr2 MLP: {:.2f}" .format(mae(y_test_pr2, m_pr2_hat)))
nrmse_pr2_mlp = rmse(y_test_pr2, m_pr2_hat) / mean_pr2
print("NRMSE Pr2 MLP: {:.2f}" .format(nrmse_pr2_mlp), "\n")

# MLP Pr3
mlp_pr3 = MLPRegressor()
mlp_pr3.fit(X_train_pr3, y_train_pr3)
m_pr3_hat = mlp_pr3.predict(X_test_pr3)
continuous_graph(m_pr3_hat, y_test_pr3, 'Pr3 power consumption in kW (MLP)', 'hour')
step_graph(m_pr3_hat, y_test_pr3, 'Pr3 power consumption in kW (MLP)', 'hour')
print("MAE Pr3 MLP: {:.2f}" .format(mae(y_test_pr3, m_pr3_hat)))
nrmse_pr3_mlp = rmse(y_test_pr3, m_pr3_hat) / mean_pr3
print("NRMSE Pr3 MLP: {:.2f}" .format(nrmse_pr3_mlp), "\n")

# MLP Pr4
mlp_pr4 = MLPRegressor()
mlp_pr4.fit(X_train_pr4, y_train_pr4)
m_pr4_hat = mlp_pr4.predict(X_test_pr4)
continuous_graph(m_pr4_hat, y_test_pr4, 'Pr4 power consumption in kW (MLP)', 'hour')
step_graph(m_pr4_hat, y_test_pr4, 'Pr4 power consumption in kW (MLP)', 'hour')
print("MAE Pr4 MLP: {:.2f}" .format(mae(y_test_pr4, m_pr4_hat)))
nrmse_pr4_mlp = rmse(y_test_pr4, m_pr4_hat) / mean_pr4
print("NRMSE Pr4 MLP: {:.2f}" .format(nrmse_pr4_mlp), "\n")

# MLP Pr5
mlp_pr5 = MLPRegressor()
mlp_pr5.fit(X_train_pr5, y_train_pr5)
m_pr5_hat = mlp_pr5.predict(X_test_pr5)
continuous_graph(m_pr5_hat, y_test_pr5, 'Pr5 power consumption in kW (MLP)', 'hour')
step_graph(m_pr5_hat, y_test_pr5, 'Pr5 power consumption in kW (MLP)', 'hour')
print("MAE Pr5 MLP: {:.2f}" .format(mae(y_test_pr5, m_pr5_hat)))
nrmse_pr5_mlp = rmse(y_test_pr5, m_pr5_hat) / mean_pr5
print("NRMSE Pr5 MLP: {:.2f}" .format(nrmse_pr5_mlp), "\n")


# Gradient boosting regression Pr1
params = {'n_estimators': 500, 'max_depth': 6, 'min_samples_split': 2, 'learning_rate': 0.01, 'loss': 'ls'}
gbr_pr1 = GradientBoostingRegressor(**params)
gbr_pr1.fit(X_train_pr1, y_train_pr1)
gbr_pr1_hat = gbr_pr1.predict(X_test_pr1)
continuous_graph(gbr_pr1_hat, y_test_pr1, 'Pr1 power consumption in kW (GBR)', 'hour')
step_graph(gbr_pr1_hat, y_test_pr1, 'Pr1 power consumption in kW (GBR)', 'hour')
print("MAE Pr1 GBR: {:.2f}" .format(mae(y_test_pr1, gbr_pr1_hat)))
nrmse_pr1_gbr = rmse(y_test_pr1, gbr_pr1_hat) / mean_pr1
print("NRMSE Pr1 GBR: {:.2f}" .format(nrmse_pr1_gbr), "\n")

# Gradient boosting regression Pr2
gbr_pr2 = GradientBoostingRegressor(**params)
gbr_pr2.fit(X_train_pr2, y_train_pr2)
gbr_pr2_hat = gbr_pr2.predict(X_test_pr2)
continuous_graph(gbr_pr2_hat, y_test_pr2, 'Pr2 power consumption in kW (GBR)', 'hour')
step_graph(gbr_pr2_hat, y_test_pr2, 'Pr2 power consumption in kW (GBR)', 'hour')
print("MAE Pr2 GBR: {:.2f}" .format(mae(y_test_pr2, gbr_pr2_hat)))
nrmse_pr2_gbr = rmse(y_test_pr2, gbr_pr2_hat) / mean_pr2
print("NRMSE Pr2 GBR: {:.2f}" .format(nrmse_pr2_gbr), "\n")

# Gradient boosting regression Pr3
gbr_pr3 = GradientBoostingRegressor(**params)
gbr_pr3.fit(X_train_pr3, y_train_pr3)
gbr_pr3_hat = gbr_pr3.predict(X_test_pr3)
continuous_graph(gbr_pr3_hat, y_test_pr3, 'Pr3 power consumption in kW (GBR)', 'hour')
step_graph(gbr_pr3_hat, y_test_pr3, 'Pr3 power consumption in kW (GBR)', 'hour')
print("MAE Pr3 GBR: {:.2f}" .format(mae(y_test_pr3, gbr_pr3_hat)))
nrmse_pr3_gbr = rmse(y_test_pr3, gbr_pr3_hat) / mean_pr3
print("NRMSE Pr3 GBR: {:.2f}" .format(nrmse_pr3_gbr), "\n")

# Gradient boosting regression Pr4
gbr_pr4 = GradientBoostingRegressor(**params)
gbr_pr4.fit(X_train_pr4, y_train_pr4)
gbr_pr4_hat = gbr_pr4.predict(X_test_pr4)
continuous_graph(gbr_pr4_hat, y_test_pr4, 'Pr4 power consumption in kW (GBR)', 'hour')
step_graph(gbr_pr4_hat, y_test_pr4, 'Pr4 power consumption in kW (GBR)', 'hour')
print("MAE Pr4 GBR: {:.2f}" .format(mae(y_test_pr4, gbr_pr4_hat)))
nrmse_pr4_gbr = rmse(y_test_pr4, gbr_pr4_hat) / mean_pr4
print("NRMSE Pr4 GBR: {:.2f}" .format(nrmse_pr4_gbr), "\n")

# Gradient boosting regression Pr5
gbr_pr5 = GradientBoostingRegressor(**params)
gbr_pr5.fit(X_train_pr5, y_train_pr5)
gbr_pr5_hat = gbr_pr5.predict(X_test_pr5)
continuous_graph(gbr_pr5_hat, y_test_pr5, 'Pr5 power consumption in kW (GBR)', 'hour')
step_graph(gbr_pr5_hat, y_test_pr5, 'Pr5 power consumption in kW (GBR)', 'hour')
print("MAE Pr5 GBR: {:.2f}" .format(mae(y_test_pr5, gbr_pr5_hat)))
nrmse_pr5_gbr = rmse(y_test_pr5, gbr_pr5_hat) / mean_pr5
print("NRMSE Pr5 GBR: {:.2f}" .format(nrmse_pr5_gbr), "\n")


# To create a csv file of forecast which can be used for optimization
date_time = ["Date"]
for a in time_last_day:
    date_time.append(a)

# Pr1 via GBR
prediction_pr1 = ["Pr-1 load in kW"]
for a in gbr_pr1_hat:
    prediction_pr1.append(a)

# Pr2 via GBR
prediction_pr2 = ["Pr-2 load in kW"]
for a in gbr_pr2_hat:
    prediction_pr2.append(a)

# Pr3 via MLP
prediction_pr3 = ["Pr-3 load in kW"]
for a in m_pr3_hat:
    prediction_pr3.append(a)

# Pr4 via Lasso
prediction_pr4 = ["Pr-4 load in kW"]
for a in lasso_pr4_hat:
    prediction_pr4.append(a)

# Pr5 via GBR
prediction_pr5 = ["Pr-5 load in kW"]
for a in gbr_pr5_hat:
    prediction_pr5.append(a)

zip_time_and_forecast = zip(date_time, prediction_pr1, prediction_pr2, prediction_pr3, prediction_pr4, prediction_pr5)
x = tuple(zip_time_and_forecast)
with open('result_consumption.csv', 'w') as csvFile:
    for a in x:
        writer = csv.writer(csvFile)
        writer.writerow(a)
