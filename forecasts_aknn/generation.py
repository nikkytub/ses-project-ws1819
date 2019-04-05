# name: Generation forecast via LR, KNN, GBR, MLP, Ridge, Lasso and AKNN.
# author: Nikhil Singh (nikhil.singh@campus.tu-berlin.de)
# data-source: Pecan Street Dataset
# Reference: Some ideas and code taken from ISIS full tutorial

import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit, KFold, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from lpi_python import lpi_distance, lpi_mean
import numpy as np
import matplotlib.pyplot as plt
import csv

data = pd.read_csv('data1.csv', parse_dates=True)
data['Unnamed: 0'] = pd.to_datetime(data['Unnamed: 0'])
data["month_of_year"] = data['Unnamed: 0'].dt.month
data["hour_of_day"] = data['Unnamed: 0'].dt.hour
data["day_of_week"] = data['Unnamed: 0'].dt.dayofweek

# For AKNN
# 02/01/2015 to 30/11/2016
ini_data = data[18:16794]
# 30/11/2016
prev_day_data = data[16770:16794]
# 01/12/2016
last_day_data = data[16794:16818]
y_test_pv_01_12 = last_day_data['PV']
y_test_wind_01_12 = last_day_data['Wind']
generation_pv_30_11 = np.array(prev_day_data['PV'])
generation_wind_30_11 = np.array(prev_day_data['Wind'])
y_train_pv = np.array(ini_data['PV'])
y_train_wind = np.array(ini_data['Wind'])
chunks_pv = [y_train_pv[x:x + 24] for x in range(0, len(y_train_pv), 24)]
chunks_wind = [y_train_wind[x:x + 24] for x in range(0, len(y_train_wind), 24)]
time_last_day = np.array(last_day_data['Unnamed: 0'])


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


def prediction(generation, chunks):
    aknn_predicted_load_school = [aknn(generation, chunks)]
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


def continuous_graph(prediction, actual, ylabl, xlabl):
    plt.plot(prediction, label='Predicted')
    plt.plot(actual.values, label='Actual')
    plt.ylabel(ylabl)
    plt.xticks([0, 5, 10, 15, 20],
               ['00:00', '05:00', '10:00', '15:00', '20:00'])
    plt.xlabel(xlabl)
    plt.legend()
    return plt.show()


def rmse(y_true, y_pred):
    """Root Mean Square Error"""
    return np.sqrt(np.average((y_true - y_pred) ** 2))


def mae(y_true, y_pred):
    """Mean Absolute Error"""
    return np.average(np.abs(y_pred - y_true))


# AKNN PV
plot_values_pv = prediction(generation_pv_30_11, chunks_pv)
continuous_graph(plot_values_pv, y_test_pv_01_12, 'PV power generation in kW (AKNN)', 'hour')
step_graph(plot_values_pv, y_test_pv_01_12, 'PV power generation in kW (AKNN)', 'hour')
print("MAE PV AKNN: {:.2f}" .format(mae(y_test_pv_01_12, plot_values_pv)))
mean_pv = np.mean(y_test_pv_01_12)
nrmse_pv_aknn = rmse(y_test_pv_01_12, plot_values_pv) / mean_pv
print("NRMSE PV AKNN: {:.2f}" .format(nrmse_pv_aknn), "\n")

# AKNN Wind
plot_values_wind = prediction(generation_wind_30_11, chunks_wind)
continuous_graph(plot_values_wind, y_test_wind_01_12, 'Wind power generation in kW (AKNN)', 'hour')
step_graph(plot_values_wind, y_test_wind_01_12, 'Wind power generation in kW (AKNN)', 'hour')
print("MAE Wind AKNN: {:.2f}" .format(mae(y_test_wind_01_12, plot_values_wind)))
mean_wind = np.mean(y_test_wind_01_12)
nrmse_wind_aknn = rmse(y_test_wind_01_12, plot_values_wind) / mean_wind
print("NRMSE Wind AKNN: {:.2f}" .format(nrmse_wind_aknn), "\n")

# 01/01/2015 to 01/12/2016
data_till_01_12 = data[:16818]
load_wind = data_till_01_12.loc[:, 'Wind']
load_pv = data_till_01_12.loc[:, 'PV']
X_wind = pd.DataFrame(index=data_till_01_12.index)
X_pv = pd.DataFrame(index=data_till_01_12.index)
lags = [1, 2, 3, 4, 5, 6, 24, 48, 168]
for lag in lags:
    X_wind.loc[:, "lag_"+str(lag)] = load_wind.shift(lag)
for lag in lags:
    X_pv.loc[:, "lag_"+str(lag)] = load_pv.shift(lag)
X_wind.loc[:, "HoD"] = data_till_01_12["hour_of_day"]
X_wind.loc[:, "DoW"] = data_till_01_12["day_of_week"]
X_wind.loc[:, "MoY"] = data_till_01_12["month_of_year"]
X_wind.loc[:, "Temperature"] = data_till_01_12["temperature"]
X_pv.loc[:, "HoD"] = data_till_01_12["hour_of_day"]
X_pv.loc[:, "DoW"] = data_till_01_12["day_of_week"]
X_pv.loc[:, "MoY"] = data_till_01_12["month_of_year"]
X_pv.loc[:, "Temperature"] = data_till_01_12["temperature"]

# One-hot encoding
X_wind = pd.get_dummies(X_wind, columns=["HoD"])
X_wind = pd.get_dummies(X_wind, columns=["DoW"])
X_wind = pd.get_dummies(X_wind, columns=["MoY"])
X_pv = pd.get_dummies(X_pv, columns=["HoD"])
X_pv = pd.get_dummies(X_pv, columns=["DoW"])
X_pv = pd.get_dummies(X_pv, columns=["MoY"])
Y_wind = data_till_01_12['Wind']
Y_pv = data_till_01_12['PV']

# Train/Test Split
X_train_wind, X_test_wind, y_train_wind, y_test_wind = train_test_split(X_wind.iloc[168:, :], Y_wind.iloc[168:],
                                                                        test_size=0.0014, shuffle=False)
X_train_pv, X_test_pv, y_train_pv, y_test_pv = train_test_split(X_pv.iloc[168:, :], Y_pv.iloc[168:],
                                                                test_size=0.0014, shuffle=False)

# KNN PV
knn_pv = KNeighborsRegressor()
knn_pv.fit(X_train_pv, y_train_pv)

# KNN Wind
knn_wind = KNeighborsRegressor()
knn_wind.fit(X_train_wind, y_train_wind)

# Cross Validation Block
N = 4
# Tried with 3, 6, 9, 12, 15, 18, 24. Got best results with 3
knn_parameters = {"n_neighbors": [3]}

gs_cv_block_pv = GridSearchCV(knn_pv, knn_parameters,
                              scoring="neg_mean_squared_error", cv=KFold(n_splits=N, shuffle=False),
                              n_jobs=-1, return_train_score=True)
gs_cv_block_pv.fit(X_train_pv, y_train_pv)
print(gs_cv_block_pv.best_params_)
y_hat_pv = gs_cv_block_pv.predict(X_test_pv)
print("Test RMSE Block: %.2f" % rmse(y_test_pv, y_hat_pv), "\n")

gs_cv_block_wind = GridSearchCV(knn_wind, knn_parameters,
                                scoring="neg_mean_squared_error", cv=KFold(n_splits=N, shuffle=False),
                                n_jobs=-1, return_train_score=True)
gs_cv_block_wind.fit(X_train_wind, y_train_wind)
print(gs_cv_block_wind.best_params_)
y_hat_wind = gs_cv_block_wind.predict(X_test_wind)
print("Test RMSE Block: %.2f" % rmse(y_test_wind, y_hat_wind), "\n")

# Cross Validation Shuffle
gs_cv_shuffle_pv = GridSearchCV(knn_pv, knn_parameters,
                                scoring="neg_mean_squared_error", cv=KFold(n_splits=N, shuffle=True),
                                n_jobs=-1, return_train_score=True)
gs_cv_shuffle_pv.fit(X_train_pv, y_train_pv)
print(gs_cv_shuffle_pv.best_params_)
y_hat_shuffle_pv = gs_cv_shuffle_pv.predict(X_test_pv)
print("Test RMSE Shuffle: %.2f" % rmse(y_test_pv, y_hat_shuffle_pv), "\n")

gs_cv_shuffle_wind = GridSearchCV(knn_wind, knn_parameters,
                                  scoring="neg_mean_squared_error", cv=KFold(n_splits=N, shuffle=True),
                                  n_jobs=-1, return_train_score=True)
gs_cv_shuffle_wind.fit(X_train_pv, y_train_pv)
print(gs_cv_shuffle_wind.best_params_)
y_hat_shuffle_wind = gs_cv_shuffle_wind.predict(X_test_wind)
print("Test RMSE Shuffle: %.2f" % rmse(y_test_wind, y_hat_shuffle_wind), "\n")


# Cross Validation Time series
gs_cv_ts_pv = GridSearchCV(knn_pv, knn_parameters,
                           scoring="neg_mean_squared_error", cv=TimeSeriesSplit(n_splits=N),
                           n_jobs=-1, return_train_score=True)
gs_cv_ts_pv.fit(X_train_pv, y_train_pv)
print(gs_cv_ts_pv.best_params_)
y_hat_ts_pv = gs_cv_ts_pv.predict(X_test_pv)
print("Test RMSE Time Series: %.2f" % rmse(y_test_pv, y_hat_ts_pv), "\n")

gs_cv_ts_wind = GridSearchCV(knn_wind, knn_parameters,
                             scoring="neg_mean_squared_error", cv=TimeSeriesSplit(n_splits=N),
                             n_jobs=-1, return_train_score=True)
gs_cv_ts_wind.fit(X_train_wind, y_train_wind)
print(gs_cv_ts_wind.best_params_)
y_hat_ts_wind = gs_cv_ts_wind.predict(X_test_wind)
print("Test RMSE Time Series: %.2f" % rmse(y_test_wind, y_hat_ts_wind), "\n")

knn_pv = KNeighborsRegressor(n_neighbors=3)
knn_pv.fit(X_train_pv, y_train_pv)
y_pv_hat = knn_pv.predict(X_test_pv)
continuous_graph(y_pv_hat, y_test_pv, 'PV power generation in kW (KNN)', 'hour')
step_graph(y_pv_hat, y_test_pv, 'PV power generation in kW (KNN)', 'hour')
print("MAE PV KNN: {:.2f}" .format(mae(y_test_pv, y_pv_hat)))
nrmse_pv_knn = rmse(y_test_pv, y_pv_hat) / mean_pv
print("NRMSE PV KNN: {:.2f}" .format(nrmse_pv_knn), "\n")

knn_wind = KNeighborsRegressor(n_neighbors=3)
knn_wind.fit(X_train_wind, y_train_wind)
y_wind_hat = knn_wind.predict(X_test_wind)
continuous_graph(y_wind_hat, y_test_wind, 'Wind power generation in kW (KNN)', 'hour')
step_graph(y_wind_hat, y_test_wind, 'Wind power generation in kW (KNN)', 'hour')
print("MAE Wind KNN: {:.2f}" .format(mae(y_test_wind, y_wind_hat)))
nrmse_wind_knn = rmse(y_test_wind, y_wind_hat) / mean_wind
print("NRMSE Wind KNN: {:.2f}" .format(nrmse_wind_knn), "\n")


# Ridge PV
r_pv = Ridge()
r_pv.fit(X_train_pv, y_train_pv)

# Ridge Wind
r_wind = Ridge()
r_wind.fit(X_train_wind, y_train_wind)

# Cross Validation Block
# Tried with 0.01, 0.1, 1.0. Got best results with 1.0
ridge_parameters = {"alpha" : [1.0]}
gs_cv_block_pv_ridge = GridSearchCV(r_pv, ridge_parameters,
                                    scoring="neg_mean_squared_error", cv=KFold(n_splits=N, shuffle=False),
                                    n_jobs=-1, return_train_score=True)
gs_cv_block_pv_ridge.fit(X_train_pv, y_train_pv)
print(gs_cv_block_pv_ridge.best_params_)
y_hat_pv_ridge = gs_cv_block_pv_ridge.predict(X_test_pv)
print("Test RMSE Block: %.2f" % rmse(y_test_pv, y_hat_pv_ridge), "\n")

gs_cv_block_wind_ridge = GridSearchCV(r_wind, ridge_parameters,
                                      scoring="neg_mean_squared_error", cv=KFold(n_splits=N, shuffle=False),
                                      n_jobs=-1, return_train_score=True)
gs_cv_block_wind_ridge.fit(X_train_wind, y_train_wind)
print(gs_cv_block_wind_ridge.best_params_)
y_hat_wind_ridge = gs_cv_block_wind_ridge.predict(X_test_wind)
print("Test RMSE Block: %.2f" % rmse(y_test_wind, y_hat_wind_ridge), "\n")

# Cross Validation Shuffle
gs_cv_shuffle_pv_ridge = GridSearchCV(r_pv, ridge_parameters,
                                      scoring="neg_mean_squared_error", cv=KFold(n_splits=N, shuffle=True),
                                      n_jobs=-1, return_train_score=True)
gs_cv_shuffle_pv_ridge.fit(X_train_pv, y_train_pv)
print(gs_cv_shuffle_pv_ridge.best_params_)
y_hat_shuffle_pv_ridge = gs_cv_shuffle_pv_ridge.predict(X_test_pv)
print("Test RMSE Shuffle: %.2f" % rmse(y_test_pv, y_hat_shuffle_pv_ridge), "\n")

gs_cv_shuffle_wind_ridge = GridSearchCV(r_wind, ridge_parameters,
                                        scoring="neg_mean_squared_error", cv=KFold(n_splits=N, shuffle=True),
                                        n_jobs=-1, return_train_score=True)
gs_cv_shuffle_wind_ridge.fit(X_train_pv, y_train_pv)
print(gs_cv_shuffle_wind_ridge.best_params_)
y_hat_shuffle_wind_ridge = gs_cv_shuffle_wind_ridge.predict(X_test_wind)
print("Test RMSE Shuffle: %.2f" % rmse(y_test_wind, y_hat_shuffle_wind_ridge), "\n")


# Cross Validation Time series
gs_cv_ts_pv_ridge = GridSearchCV(r_pv, ridge_parameters,
                                 scoring="neg_mean_squared_error", cv=TimeSeriesSplit(n_splits=N),
                                 n_jobs=-1, return_train_score=True)
gs_cv_ts_pv_ridge.fit(X_train_pv, y_train_pv)
print(gs_cv_ts_pv_ridge.best_params_)
y_hat_ts_pv_ridge = gs_cv_ts_pv_ridge.predict(X_test_pv)
print("Test RMSE Time Series: %.2f" % rmse(y_test_pv, y_hat_ts_pv_ridge), "\n")

gs_cv_ts_wind_ridge = GridSearchCV(r_wind, ridge_parameters,
                                   scoring="neg_mean_squared_error", cv=TimeSeriesSplit(n_splits=N),
                                   n_jobs=-1, return_train_score=True)
gs_cv_ts_wind_ridge.fit(X_train_wind, y_train_wind)
print(gs_cv_ts_wind_ridge.best_params_)
y_hat_ts_wind_ridge = gs_cv_ts_wind_ridge.predict(X_test_wind)
print("Test RMSE Time Series: %.2f" % rmse(y_test_wind, y_hat_ts_wind_ridge), "\n")

r_pv = Ridge(alpha=1.0)
r_pv.fit(X_train_pv, y_train_pv)
r_pv_hat = r_pv.predict(X_test_pv)
continuous_graph(r_pv_hat, y_test_pv, 'PV power generation in kW (Ridge)', 'hour')
step_graph(r_pv_hat, y_test_pv, 'PV power generation in kW (Ridge)', 'hour')
print("MAE PV Ridge: {:.2f}" .format(mae(y_test_pv, r_pv_hat)))
nrmse_pv_ridge = rmse(y_test_pv, r_pv_hat) / mean_pv
print("NRMSE PV Ridge: {:.2f}" .format(nrmse_pv_ridge), "\n")

# Ridge Wind
r_wind = Ridge(alpha=1.0)
r_wind.fit(X_train_wind, y_train_wind)
r_wind_hat = r_wind.predict(X_test_wind)
continuous_graph(r_wind_hat, y_test_wind, 'Wind power generation in kW (Ridge)', 'hour')
step_graph(r_wind_hat, y_test_wind, 'Wind power generation in kW (Ridge)', 'hour')
print("MAE Wind Ridge: {:.2f}" .format(mae(y_test_wind, r_wind_hat)))
nrmse_wind_ridge = rmse(y_test_wind, r_wind_hat) / mean_wind
print("NRMSE Wind Ridge: {:.2f}" .format(nrmse_wind_ridge), "\n")

# Lasso PV
lasso_pv = Lasso()
lasso_pv.fit(X_train_pv, y_train_pv)

# Lasso Wind
lasso_wind = Lasso()
lasso_wind.fit(X_train_wind, y_train_wind)

# Cross Validation Block
# Tried with 0.01, 0.1, 1.0. Got best results with 0.01
lasso_parameters = {"alpha" : [0.01]}
gs_cv_block_pv_lasso = GridSearchCV(lasso_pv, lasso_parameters,
                                    scoring="neg_mean_squared_error", cv=KFold(n_splits=N, shuffle=False),
                                    n_jobs=-1, return_train_score=True)
gs_cv_block_pv_lasso.fit(X_train_pv, y_train_pv)
print(gs_cv_block_pv_lasso.best_params_)
y_hat_pv_lasso = gs_cv_block_pv_lasso.predict(X_test_pv)
print("Test RMSE Block: %.2f" % rmse(y_test_pv, y_hat_pv_lasso), "\n")

gs_cv_block_wind_lasso = GridSearchCV(lasso_wind, lasso_parameters,
                                      scoring="neg_mean_squared_error", cv=KFold(n_splits=N, shuffle=False),
                                      n_jobs=-1, return_train_score=True)
gs_cv_block_wind_lasso.fit(X_train_wind, y_train_wind)
print(gs_cv_block_wind_lasso.best_params_)
y_hat_wind_lasso = gs_cv_block_wind_lasso.predict(X_test_wind)
print("Test RMSE Block: %.2f" % rmse(y_test_wind, y_hat_wind_lasso), "\n")

# Cross Validation Shuffle
gs_cv_shuffle_pv_lasso = GridSearchCV(lasso_pv, lasso_parameters,
                                      scoring="neg_mean_squared_error", cv=KFold(n_splits=N, shuffle=True),
                                      n_jobs=-1, return_train_score=True)
gs_cv_shuffle_pv_lasso.fit(X_train_pv, y_train_pv)
print(gs_cv_shuffle_pv_lasso.best_params_)
y_hat_shuffle_pv_lasso = gs_cv_shuffle_pv_lasso.predict(X_test_pv)
print("Test RMSE Shuffle: %.2f" % rmse(y_test_pv, y_hat_shuffle_pv_lasso), "\n")

gs_cv_shuffle_wind_lasso = GridSearchCV(lasso_wind, lasso_parameters,
                                        scoring="neg_mean_squared_error", cv=KFold(n_splits=N, shuffle=True),
                                        n_jobs=-1, return_train_score=True)
gs_cv_shuffle_wind_lasso.fit(X_train_pv, y_train_pv)
print(gs_cv_shuffle_wind_lasso.best_params_)
y_hat_shuffle_wind_lasso = gs_cv_shuffle_wind_lasso.predict(X_test_wind)
print("Test RMSE Shuffle: %.2f" % rmse(y_test_wind, y_hat_shuffle_wind_lasso), "\n")

# Cross Validation Time series
gs_cv_ts_pv_lasso = GridSearchCV(lasso_pv, lasso_parameters,
                                 scoring="neg_mean_squared_error", cv=TimeSeriesSplit(n_splits=N),
                                 n_jobs=-1, return_train_score=True)
gs_cv_ts_pv_lasso.fit(X_train_pv, y_train_pv)
print(gs_cv_ts_pv_lasso.best_params_)
y_hat_ts_pv_lasso = gs_cv_ts_pv_lasso.predict(X_test_pv)
print("Test RMSE Time Series: %.2f" % rmse(y_test_pv, y_hat_ts_pv_lasso), "\n")

gs_cv_ts_wind_lasso = GridSearchCV(lasso_wind, lasso_parameters,
                                   scoring="neg_mean_squared_error", cv=TimeSeriesSplit(n_splits=N),
                                   n_jobs=-1, return_train_score=True)
gs_cv_ts_wind_lasso.fit(X_train_wind, y_train_wind)
print(gs_cv_ts_wind_lasso.best_params_)
y_hat_ts_wind_lasso = gs_cv_ts_wind_lasso.predict(X_test_wind)
print("Test RMSE Time Series: %.2f" % rmse(y_test_wind, y_hat_ts_wind_lasso), "\n")

# Lasso PV
lasso_pv = Lasso(alpha=0.01)
lasso_pv.fit(X_train_pv, y_train_pv)
lasso_pv_hat = lasso_pv.predict(X_test_pv)
continuous_graph(lasso_pv_hat, y_test_pv, 'PV power generation in kW (Lasso)', 'hour')
step_graph(lasso_pv_hat, y_test_pv, 'PV power generation in kW (Lasso)', 'hour')
print("MAE PV Lasso: {:.2f}" .format(mae(y_test_pv, lasso_pv_hat)))
nrmse_pv_lasso = rmse(y_test_pv, lasso_pv_hat) / mean_pv
print("NRMSE PV Lasso: {:.2f}" .format(nrmse_pv_lasso), "\n")

# Lasso Wind
lasso_wind = Lasso(alpha=0.01)
lasso_wind.fit(X_train_wind, y_train_wind)
lasso_wind_hat = lasso_wind.predict(X_test_wind)
continuous_graph(lasso_wind_hat, y_test_wind, 'Wind power generation in kW (Lasso)', 'hour')
step_graph(lasso_wind_hat, y_test_wind, 'Wind power generation in kW (Lasso)', 'hour')
print("MAE Wind Lasso: {:.2f}" .format(mae(y_test_wind, lasso_wind_hat)))
nrmse_wind_lasso = rmse(y_test_wind, lasso_wind_hat) / mean_wind
print("NRMSE Wind Lasso: {:.2f}" .format(nrmse_wind_lasso), "\n")

# Linear regression PV
lr_pv = LinearRegression()
lr_pv.fit(X_train_pv, y_train_pv)
lr_pv_hat = lr_pv.predict(X_test_pv)
continuous_graph(lr_pv_hat, y_test_pv, 'PV power generation in kW (LR)', 'hour')
step_graph(lr_pv_hat, y_test_pv, 'PV power generation in kW (LR)', 'hour')
print("MAE PV LR: {:.2f}" .format(mae(y_test_pv, lr_pv_hat)))
nrmse_pv_lr = rmse(y_test_pv, lr_pv_hat) / mean_pv
print("NRMSE PV LR: {:.2f}" .format(nrmse_pv_lr), "\n")

# Linear regression Wind
lr_wind = LinearRegression()
lr_wind.fit(X_train_wind, y_train_wind)
lr_wind_hat = lr_wind.predict(X_test_wind)
continuous_graph(lr_wind_hat, y_test_wind, 'Wind power generation in kW (LR)', 'hour')
step_graph(lr_wind_hat, y_test_wind, 'Wind power generation in kW (LR)', 'hour')
print("MAE Wind LR: {:.2f}" .format(mae(y_test_wind, lr_wind_hat)))
nrmse_wind_lr = rmse(y_test_wind, lr_wind_hat) / mean_wind
print("NRMSE Wind LR: {:.2f}" .format(nrmse_wind_lr), "\n")

# MLP PV
mlp_pv = MLPRegressor()
mlp_pv.fit(X_train_pv, y_train_pv)
m_pv_hat = mlp_pv.predict(X_test_pv)
continuous_graph(m_pv_hat, y_test_pv, 'PV power generation in kW (MLP)', 'hour')
step_graph(m_pv_hat, y_test_pv, 'PV power generation in kW (MLP)', 'hour')
print("MAE PV MLP: {:.2f}" .format(mae(y_test_pv, m_pv_hat)))
nrmse_pv_mlp = rmse(y_test_pv, m_pv_hat) / mean_pv
print("NRMSE PV MLP: {:.2f}" .format(nrmse_pv_mlp), "\n")

# MLP Wind
mlp_wind = MLPRegressor()
mlp_wind.fit(X_train_wind, y_train_wind)
m_wind_hat = mlp_wind.predict(X_test_wind)
continuous_graph(m_wind_hat, y_test_wind, 'Wind power generation in kW (MLP)', 'hour')
step_graph(m_wind_hat, y_test_wind, 'Wind power generation in kW (MLP)', 'hour')
print("MAE Wind MLP: {:.2f}" .format(mae(y_test_wind, m_wind_hat)))
nrmse_wind_mlp = rmse(y_test_wind, m_wind_hat) / mean_wind
print("NRMSE Wind MLP: {:.2f}" .format(nrmse_wind_mlp), "\n")

# Gradient boosting regression PV
params = {'n_estimators': 500, 'max_depth': 6, 'min_samples_split': 2, 'learning_rate': 0.01, 'loss': 'ls'}
gbr_pv = GradientBoostingRegressor(**params)
gbr_pv.fit(X_train_pv, y_train_pv)
gbr_pv_hat = gbr_pv.predict(X_test_pv)
continuous_graph(gbr_pv_hat, y_test_pv, 'PV power generation in kW (GBR)', 'hour')
step_graph(gbr_pv_hat, y_test_pv, 'PV power generation in kW (GBR)', 'hour')
print("MAE PV GBR: {:.2f}" .format(mae(y_test_pv, gbr_pv_hat)))
nrmse_pv_gbr = rmse(y_test_pv, gbr_pv_hat) / mean_pv
print("NRMSE PV GBR: {:.2f}" .format(nrmse_pv_gbr), "\n")

# Gradient boosting regression Wind
gbr_wind = GradientBoostingRegressor(**params)
gbr_wind.fit(X_train_wind, y_train_wind)
gbr_wind_hat = gbr_wind.predict(X_test_wind)
continuous_graph(gbr_wind_hat, y_test_wind, 'Wind power generation in kW (GBR)', 'hour')
step_graph(gbr_wind_hat, y_test_wind, 'Wind power generation in kW (GBR)', 'hour')
print("MAE Wind GBR: {:.2f}" .format(mae(y_test_wind, gbr_wind_hat)))
nrmse_wind_gbr = rmse(y_test_wind, gbr_wind_hat) / mean_wind
print("NRMSE Wind GBR: {:.2f}" .format(nrmse_wind_gbr), "\n")

# Comparison PV using step graph
hour = []
for i in range(24):
    hour.append(i)
plt.step(hour, lr_pv_hat, label='Predicted LR')
plt.step(hour, y_test_pv.values, label='Actual')
plt.step(hour, y_pv_hat, label='Predicted KNN')
plt.step(hour, gbr_pv_hat, label='Predicted GBR')
plt.step(hour, m_pv_hat, label='Predicted MLP')
plt.step(hour, r_pv_hat, label='Predicted Ridge')
plt.step(hour, lasso_pv_hat, label='Predicted Lasso')
plt.step(hour, plot_values_pv, label='Predicted AKNN')
plt.ylabel('Power in kW')
plt.xticks([0, 5, 10, 15, 20],
           ['00:00', '05:00', '10:00', '15:00', '20:00'])
plt.xlabel('hour')
plt.legend()
plt.show()

# Comparison Wind using step graph
plt.step(hour, lr_wind_hat, label='Predicted LR')
plt.step(hour, y_test_wind.values, label='Actual')
plt.step(hour, y_wind_hat, label='Predicted KNN')
plt.step(hour, gbr_wind_hat, label='Predicted GBR')
plt.step(hour, m_wind_hat, label='Predicted MLP')
plt.step(hour, r_wind_hat, label='Predicted Ridge')
plt.step(hour, lasso_wind_hat, label='Predicted Lasso')
plt.step(hour, plot_values_wind, label='Predicted AKNN')
plt.ylabel('Power in kW')
plt.xticks([0, 5, 10, 15, 20],
           ['00:00', '05:00', '10:00', '15:00', '20:00'])
plt.xlabel('hour')
plt.legend()
plt.show()


# To create a csv file of forecast which can be used for optimization
date_time = ["Day"]
for a in time_last_day:
    date_time.append(a)

# For PV, KNN gave the best prediction based on lowest NRMSE. Hence we are using it.
prediction_pv = ["PV power generation in kW"]
for a in y_wind_hat:
    prediction_pv.append(a)

# For wind, MLP gave the best prediction based on lowest NRMSE. Hence we are using it.
prediction_wind = ["Wind power generation in kW"]
for a in m_wind_hat:
    prediction_wind.append(a)

zip_time_and_forecast = zip(date_time, prediction_pv, prediction_wind)
x = tuple(zip_time_and_forecast)
with open('result_generation.csv', 'w') as csvFile:
    for a in x:
        writer = csv.writer(csvFile)
        writer.writerow(a)
