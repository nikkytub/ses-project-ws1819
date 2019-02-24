# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 17:39:16 2019

@author: Hanggai
"""

import pandas as pd # work with data
import numpy as np # work with matrices and vectors
from sqlalchemy import create_engine # database
import requests # rest web services
import pytz # timezones

# machine learning
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split, TimeSeriesSplit, KFold, GridSearchCV
from sklearn.multioutput import MultiOutputRegressor, RegressorChain

from quantile import QuantileRegressor
from features import *

from scipy.stats import multivariate_normal, norm, uniform
import scipy.interpolate as interpolate

# plotting libraries
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import seaborn as sns

# database connection and queries
engine = create_engine('postgresql://wCwvMcbXb2hT:m8hfhgl14gOo@dataport.pecanstreet.org:5434/postgres')
query= "SELECT local_15min, use, gen, car1 FROM university.electricity_egauge_15min WHERE local_15min >= \'%s\' and local_15min < \'%s\' and dataid=%s order by local_15min"
weather_query = "select localhour, (temperature - 32) /  1.8 as temperature, (apparent_temperature - 32) /  1.8 as apparent_temperature, dew_point, humidity, visibility, pressure, wind_speed, cloud_cover, wind_bearing, precip_intensity, precip_probability from university.weather where localhour >= \'%s\' and localhour <= \'%s\' and latitude = %f order by localhour"

# download meta data table
meta_df = pd.read_sql_query("SELECT * FROM university.metadata WHERE dataid=6547", con=engine)
meta_df["egauge_max_time"] = pd.to_datetime(meta_df["egauge_max_time"], utc=True)
meta_df["egauge_min_time"] = pd.to_datetime(meta_df["egauge_min_time"], utc=True)

# filter meta data for date and available data
start_date = "2015-01-01"
end_date = "2017-01-01"
tz = pytz.timezone("US/Central")
latitude = 30.292432       # (30.292432,-97.699662)
#latitude = 65.654867
meta_df = meta_df[(meta_df.egauge_max_time >= end_date) & (meta_df.egauge_min_time <= start_date)]
meta_df = meta_df[(meta_df.city == "Austin")&(meta_df.use == "yes")&(meta_df.pv == "yes")&(meta_df.car1 == "yes")]

load = pd.read_sql_query(query % (start_date, end_date, 1642), con=engine) # household 1642

# make datetime the index
load["time"] = load.iloc[:, 0]
load["time"] = pd.to_datetime(load["time"])
load = load.set_index(load["time"])

# some handling of summer time erros
load.index = load.index.tz_localize(tz, ambiguous='NaT')
load = load[~load.index.duplicated(keep='first')]
load = load.reindex(index=pd.date_range(load.index[0], load.index[-1], freq="15min", tz=tz))

# drop colums that are now index
load.drop(['time', 'local_15min'], 1, inplace=True) 

# fix smaller holes
load = load.interpolate(limit=12)

# fix larger holes
for c in load:
    load[c] = load[c].fillna(value=load[c].shift(96)) 

# household load is use minus electrical vehicle
load["household"] = load.use - load.car1
load = load.drop("use", axis=1)
load.columns = ["PV", "EV", "Load"]

weather = pd.read_sql_query(weather_query % (start_date, end_date, latitude), con=engine)
    
# make proper timeindex 
weather["localhour"] = pd.to_datetime(weather["localhour"], utc=True)
weather = weather.set_index(weather["localhour"])

# handle time shift errors
weather.index = weather.index.tz_convert(tz=tz)
weather = weather.drop(["localhour"], axis=1)
weather = weather[~weather.index.duplicated(keep='first')]
weather = weather.reindex(index=pd.date_range(weather.index[0], weather.index[-1], freq="H", tz=tz))

# clean data
weather = weather.interpolate(limit=3) # small holes

# larger holes
for c in weather:
    weather[c] = weather[c].fillna(value=weather[c].shift(24)) 
    
data = [go.Scatter(x=load.index.tz_localize(None), y=load.Load, name="Load"),
        go.Scatter(x=load.index.tz_localize(None), y=load.PV, name="PV"),
        go.Scatter(x=load.index.tz_localize(None), y=load.EV, name="EV")]

token = '47bdf2cf899352c96c0dff33806d07d72107fb54'
api_base = 'https://www.renewables.ninja/api/'

s = requests.session()
s.headers = {'Authorization': 'Token ' + token}
url = api_base + 'data/wind'

dfs = []
for year in [2015, 2016]:
    args = {
        'lat': 30.292432,
        'lon': -97.699662,
        'date_from': '%d-01-01' % (year),
        'date_to': '%d-01-01' % (year+1),
        'capacity': 5.0,
        'height': 10,
        'turbine': 'Vestas V80 2000',
        'format': 'json',
        'metadata': False,
        'raw': False
    }

    r = s.get(url, params=args)
    dfs.append(pd.read_json(r.text, orient='index'))
wind = pd.concat(dfs)
wind.index = wind.index.tz_localize("UTC").tz_convert(tz)
wind = wind[~wind.index.duplicated(keep='first')]
wind.columns = ["Wind"]

# Concat the data
#data = pd.concat([load,wind.resample("15min").interpolate(), weather.resample("15min").interpolate(), ], axis=1)
data = pd.concat([load,wind.resample("15min").interpolate(), weather.resample("15min").interpolate(), ], axis=1)
#data = data["2015":"2016"]
data = data["2015"]

# General feature engineering
data["day_of_week"] = data.index.day
data["hour_of_day"] = data.index.hour
data["month_of_year"] = data.index.month
data["trend"] = np.arange(0, data.shape[0])

# Add lags
for lag in range(1, 97):
    data["lag_" + str(lag)] = data["Load"].shift(lag)

#lags for PV
for lag in range(1,97):
    data["lag_pv_" + str(lag)] = data["PV"].shift(lag)
    
#lags for Wind
for lag in range(1,97):
    data["lag_wind_" + str(lag)] = data["Wind"].shift(lag)
    
# or to dummy encoding with:
pd.get_dummies(data["hour_of_day"])

load_transformer = make_column_transformer(
    (make_pipeline(HourOfDay(), OneHotEncoder(sparse=False, categories=[range(24)])), ["Load"]),
    (make_pipeline(DayOfWeek(), OneHotEncoder(sparse=False, categories=[range(7)])), ["Load"]),
    (make_pipeline(MonthOfYear(), OneHotEncoder(sparse=False, categories=[range(12)])), ["Load"]),
    (make_pipeline(Forecast(H=24), StandardScaler()), ["temperature"]),
    (make_pipeline(Lags(lags=[1,2,3,4,5,6,96,672]), StandardScaler()), ["Load"]), remainder="drop"   
)
X = pd.DataFrame(load_transformer.fit_transform(data), index=data.index)   
X

#X for PV
load_transformer_pv = make_column_transformer(
    (make_pipeline(HourOfDay(), OneHotEncoder(sparse=False, categories=[range(24)])), ["PV"]),
    (make_pipeline(DayOfWeek(), OneHotEncoder(sparse=False, categories=[range(7)])), ["PV"]),
    (make_pipeline(MonthOfYear(), OneHotEncoder(sparse=False, categories=[range(12)])), ["PV"]),
    (make_pipeline(Forecast(H=24), StandardScaler()), ["dew_point"]),
    (make_pipeline(Lags(lags=[1,2,3,4,5,6,96,672]), StandardScaler()), ["PV"]), remainder="drop"   
)
X_pv = pd.DataFrame(load_transformer_pv.fit_transform(data), index=data.index)  

#X for wind
load_transformer_wind = make_column_transformer(
    (make_pipeline(HourOfDay(), OneHotEncoder(sparse=False, categories=[range(24)])), ["Wind"]),
    (make_pipeline(DayOfWeek(), OneHotEncoder(sparse=False, categories=[range(7)])), ["Wind"]),
    (make_pipeline(MonthOfYear(), OneHotEncoder(sparse=False, categories=[range(12)])), ["Wind"]),
    (make_pipeline(Forecast(H=24), StandardScaler()), ["wind_speed"]),
    (make_pipeline(Lags(lags=[1,2,3,4,5,6,96,672]), StandardScaler()), ["Wind"]), remainder="drop"   
)
X_wind = pd.DataFrame(load_transformer_wind.fit_transform(data), index=data.index) 

def rmse(y_true, y_pred):
    '''Root Mean Square Error'''
    return np.sqrt(np.average((y_true - y_pred) ** 2))

def mae(y_true, y_pred):
    '''Mean Absolute Error'''
    return np.average(np.abs(y_pred - y_true))

def nrmse(y_true, y_pred):
    '''Normalised Root Mean Square Error'''
    print(np.mean((np.mean(y_pred))))
    return np.sqrt(np.average((y_true - y_pred) ** 2))/np.mean((np.mean(y_pred)))

# Only full hour
X = X[X.index.minute==0]

# Transform y
y = LabelToHorizon(24).fit_transform(data.Load)
y = y[y.index.minute==0]


# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X.iloc[672:,:], y.iloc[672:,:], test_size=0.25, shuffle=False)


# Only full hour
X_wind = X_wind[X_wind.index.minute==0]

# Transform y
y_wind = LabelToHorizon(24).fit_transform(data.Wind)
y_wind = y_wind[y_wind.index.minute==0]


# Train/Test Split
X_wind_train, X_wind_test, y_wind_train, y_wind_test = train_test_split(X_wind.iloc[672:,:], y_wind.iloc[672:,:], test_size=0.25, shuffle=False)

# Only full hour
X_pv = X_pv[X_pv.index.minute==0]

# Transform y
y_pv = LabelToHorizon(24).fit_transform(data.PV)
y_pv = y_pv[y_pv.index.minute==0]


# Train/Test Split
X_pv_train, X_pv_test, y_pv_train, y_pv_test = train_test_split(X_pv.iloc[672:,:], y_pv.iloc[672:,:], test_size=0.25, shuffle=False)

# Fit an ANN for PV
mlp = MLPRegressor()
mlp.fit(X_pv_train, y_pv_train)
y_pv_hat = mlp.predict(X_pv_test)
print("RMSE: %.2f" % rmse(y_pv_test, y_pv_hat))
print("MAE: %.2f" % mae(y_pv_test, y_pv_hat))
print("NRMSE: %.2f" % nrmse(y_pv_test, y_pv_hat))

#######plot the graph of the last day
Y = []
for i in range(1998,2022):
    Y.append(y_pv_test.iloc[i,0])

plt.plot(y_pv_hat[1998:2022,0], label="prediction")
plt.plot(Y, label="actual")
plt.xlabel('hour')
plt.ylabel('PV power generation in kw (ANN)')
plt.legend()

# Fit an ANN for Wind
mlp = MLPRegressor()
mlp.fit(X_wind_train, y_wind_train)
y_wind_hat = mlp.predict(X_wind_test)
print("RMSE: %.2f" % rmse(y_wind_test, y_wind_hat))
print("MAE: %.2f" % mae(y_wind_test, y_wind_hat))
print("NRMSE: %.2f" % nrmse(y_wind_test, y_wind_hat))

#######plot the graph of the last day
Y_wind = []
for i in range(1998,2022):
    Y_wind.append(y_wind_test.iloc[i,0])
#Y

plt.plot(y_wind_hat[1998:2022,0], label="prediction")
plt.plot(Y_wind, label="actual")
plt.xlabel('hour')
plt.ylabel('wind power generation in kw (ANN)')
#plt.title('Wind forecast ANN')
plt.legend()