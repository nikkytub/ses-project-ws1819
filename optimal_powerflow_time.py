# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 15:11:57 2018

@author: draz
"""
import pandapower as pp
from pandapower.plotting.plotly import simple_plotly
from pandapower.plotting.plotly import vlevel_plotly
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


NOMINAL_POWER_BATTERY = -10
MIN_POWER_BATTERY = -10
MAX_POWER_BATTERY = 0


NOMINAL_POWER_CHP = -20
MIN_POWER_CHP = -20
MAX_POWER_CHP = 0

# Import data
day = []
hour = []
PV_cost = []  # Euro/kWh
wind_cost = []  # Euro/kWh
CHP_cost = []  # Euro/kWh
ext_cost = []  # Euro/kWh
PV_gen = []  # kW
wind_gen = []  # kW
TEL_load = []
charging_station_load = []  #kw

p_kw_PV = []
q_kvar_PV = []

p_kw_CHP = []
q_kvar_CHP = []

p_kw_wind = []
q_kvar_wind = []


p_kw_battery = []
q_kvar_battery = []

p_kw_ext = []
q_kvar_ext = []

p_kw_charging_station = []
q_kvar_charging_station = []

p_kw_TEL = []
q_kvar_TEL = []

total_cost = []


with open('task3_energy_cost.csv', 'r', encoding='UTF-8') as f:
    for line in f:
        break
    for line in f:
        row = line.replace(' ', ',')
        row = row.split(',')
        day.append(row[0])
        hour.append(row[1])
        PV_cost.append(float(row[2]))
        wind_cost.append(float(row[3]))
        CHP_cost.append(float(row[4]))
        ext_cost.append(float(row[5]))

with open('task3_pv_and_wind.csv', 'r', encoding='UTF-8') as f:
    for line in f:
        break
    for line in f:
        row = line.replace(' ', ',')
        row = row.split(',')
        PV_gen.append(float(row[2]))
        wind_gen.append(float(row[3]))

with open('results_house_1minute.csv', 'r', encoding='UTF-8') as f:
    for line in f:
        row = line.split(';')
        TEL_load.append(float(row[1]))

with open('EV_6LPload_week_1minute.csv', 'r', encoding='UTF-8') as f:
    for line in f:
        break
    for i, line in enumerate(f):
        if i % 60 == 0:
            row = line.split(',')
            charging_station_load.append(float(row[1]))

n = len(charging_station_load)
#n = 37
cost_demand =[]
cost_supply = []

for i in range(0, 24):
    print('--------------------------------',i,'----------------------------------')
    # creat an empty network
    net = pp.create_empty_network()

    # define buses
    bus0 = pp.create_bus(net, index=0, name="Trafo1", vn_kv=20, type='b', zone='A', vm_max_pu=1.05, vm_min_pu=0.95,
                         in_service=True)
    bus1 = pp.create_bus(net, index=1, name="Trafo2", vn_kv=0.4, type='b', zone='A', vm_max_pu=1.05, vm_min_pu=0.95,
                         in_service=True)
    bus2 = pp.create_bus(net, index=2, name="PV", vn_kv=0.4, type='b', zone='A', vm_max_pu=1.05, vm_min_pu=0.95,
                         in_service=True)
    bus3 = pp.create_bus(net, index=3, name="EV", vn_kv=0.4, type='b', zone='A', vm_max_pu=1.05, vm_min_pu=0.95,
                         in_service=True)
    bus4 = pp.create_bus(net, index=4, name="Battery", vn_kv=0.4, type='b', zone='A', vm_max_pu=1.05, vm_min_pu=0.95,
                         in_service=True)
    bus5 = pp.create_bus(net, index=5, name="CHP", vn_kv=0.4, type='b', zone='A', vm_max_pu=1.05, vm_min_pu=0.95,
                         in_service=True)
    bus6 = pp.create_bus(net, index=6, name="Wind", vn_kv=0.4, type='b', zone='A', vm_max_pu=1.05, vm_min_pu=0.95,
                         in_service=True)
    bus7 = pp.create_bus(net, index=7, name="T_building", vn_kv=0.4, type='b', zone='A', vm_max_pu=1.05, vm_min_pu=0.95,
                         in_service=True)

    # define system lines
    pp.create_line(net, bus1, bus2, length_km=0.2, std_type='NAYY 4x50 SE', name='line1', index=1, in_service=True,
                   max_loading_percent=50)
    pp.create_line(net, bus2, bus3, length_km=0.2, std_type='NAYY 4x50 SE', name='line2', index=2, in_service=True,
                   max_loading_percent=50)
    pp.create_line(net, bus3, bus4, length_km=0.3, std_type='NAYY 4x50 SE', name='line3', index=3, in_service=True,
                   max_loading_percent=50)
    pp.create_line(net, bus4, bus5, length_km=0.2, std_type='NAYY 4x50 SE', name='line4', index=4, in_service=True,
                   max_loading_percent=50)
    pp.create_line(net, bus5, bus6, length_km=0.3, std_type='NAYY 4x50 SE', name='line5', index=5, in_service=True,
                   max_loading_percent=50)
    pp.create_line(net, bus6, bus7, length_km=0.4, std_type='NAYY 4x50 SE', name='line6', index=6, in_service=True,
                   max_loading_percent=50)
    pp.create_line(net, bus7, bus1, length_km=0.4, std_type='NAYY 4x50 SE', name='line7', index=7, in_service=True,
                   max_loading_percent=50)

    # tranformer
    pp.create_transformer(net, bus0, bus1, name='trafoDAI', std_type="0.63 MVA 20/0.4 kV", max_loading_percent=50)

    # switches
    sw1 = pp.create_switch(net, bus0, element=0, et='t', closed="False", type="CB", name="CBT1")
    sw2 = pp.create_switch(net, bus0, element=0, et='t', closed="False", type="CB", name="CBT2")

    # create generators
    grid = pp.create_ext_grid(net, bus0)
    solar = pp.create_sgen(net, bus2, p_kw=-PV_gen[i], min_p_kw=-max(PV_gen), max_p_kw=0, min_q_kvar=-1,
                           max_q_kvar=0., controllable=False, index=1)
    battery = pp.create_storage(net, bus4, p_kw=NOMINAL_POWER_BATTERY, min_e_kwh=0, max_e_kwh=30, min_q_kvar=-1,
                                max_q_kvar=0., min_p_kw=MIN_POWER_BATTERY, max_p_kw=MAX_POWER_BATTERY, controllable=True, index=3)
    chp = pp.create_sgen(net, bus5, p_kw=NOMINAL_POWER_CHP, min_p_kw=MIN_POWER_CHP, min_q_kvar=-1, max_q_kvar=0., max_p_kw=MAX_POWER_CHP,
                         controllable=True, index=5)
    wind = pp.create_sgen(net, bus6, p_kw=-wind_gen[i], min_p_kw=-max(wind_gen), min_q_kvar=-1, max_q_kvar=0., max_p_kw=0,
                          controllable=False, index=4)

    # loads
    charging_station = pp.create_load(net, bus3, p_kw=charging_station_load[i], name='charging station', min_p_kw=min(charging_station_load), max_p_kw=max(charging_station_load),
                                      min_q_kvar=0., max_q_kvar=1.0, controllable=False, index=0)
    T_buidling = pp.create_load(net, bus7, p_kw=TEL_load[i], name='T_building', min_p_kw=min(TEL_load), max_p_kw=max(TEL_load), min_q_kvar=0.,
                                max_q_kvar=2.0, controllable=True, index=1)

    # generator cost function
    pp.create_polynomial_cost(net, grid, 'ext_grid', np.array([-ext_cost[i], 0]))
    pp.create_polynomial_cost(net, solar, 'sgen', np.array([-PV_cost[i], 0]))
    pp.create_polynomial_cost(net, battery, 'storage', np.array([-ext_cost[i], 0]))
    pp.create_polynomial_cost(net, chp, 'sgen', np.array([-CHP_cost[i], 0]))
    pp.create_polynomial_cost(net, wind, 'sgen', np.array([-wind_cost[i], 0]))

    # load cost function
    #pp.create_polynomial_cost(net, charging_station, 'load', np.array([-0.4, 0]))
    #pp.create_polynomial_cost(net, T_buidling, 'load', np.array([-0.4, 0]))


    # run power flow and optimal power flow

    # pp.runpp(net)
    pp.runopp(net, verbose=False, numba=False)
    # pp.rundcpp(net)
    #pp.rundcopp(net, verbose=False)

# visualize system topology

# (python-igraph is needed if you do not assign the coordinates to grid elements)
# for non-windows machines you can download it using: $ pip install python-igraph
# for windows machines, you should do that manually because python-igraph is not supported for windows.
# step1: open this link https://www.lfd.uci.edu/~gohlke/pythonlibs/#python-igraph
# Step2: download the appropriate wheel igraph version according to your machine and python version
# sptep3: pip install <wheel file name and path>
# step 4: run the following command simple_plotly(net) or vlevel_plotly (net)

#simple_plotly(net)
# vlevel_plotly(net, respect_switches=True, use_line_geodata=None, colors_dict=None, on_map=False, projection=None, map_style='basic', figsize=1, aspectratio='auto', line_width=2, bus_size=10)

# print results
    pd.set_option('display.max_columns', 15)
    # power flow in the lines
    #print (net.res_line)
    # power, voltage and angles at  buses
    #print (net.res_bus)
    # power from generators and battery
    print(net.res_sgen)
    #print ("only p_kw"+str(net.res_sgen.at[1, 'p_kw']))
    #print("only p_kw" + net.res_sgen[1])
    p_kw_PV.append(net.res_sgen.at[1, 'p_kw'])
    q_kvar_PV.append(net.res_sgen.at[1, 'q_kvar'])

    p_kw_CHP.append(net.res_sgen.at[5, 'p_kw'])
    q_kvar_CHP.append(net.res_sgen.at[5, 'q_kvar'])

    p_kw_wind.append(net.res_sgen.at[4, 'p_kw'])
    q_kvar_wind.append(net.res_sgen.at[4, 'q_kvar'])


    print(net.res_storage)
    p_kw_battery.append(net.res_storage.at[3, 'p_kw'])
    q_kvar_battery.append(net.res_storage.at[3, 'q_kvar'])

    # power from/to external grid
    print (net.res_ext_grid)
    p_kw_ext.append(net.res_ext_grid.at[0, 'p_kw'])
    q_kvar_ext.append(net.res_ext_grid.at[0, 'q_kvar'])

    print(net.res_load)
    p_kw_charging_station.append(net.res_load.at[0, 'p_kw'])
    q_kvar_charging_station.append(net.res_load.at[0, 'q_kvar'])

    p_kw_TEL.append(net.res_load.at[1, 'p_kw'])
    q_kvar_TEL.append(net.res_load.at[1, 'q_kvar'])

    # total energy supply cost
    print(net.res_cost)
    total_cost.append(net.res_cost)

    cost_demand.append( net.res_cost / (net.res_load.at[0, 'p_kw'] + net.res_load.at[1, 'p_kw']))
    cost_supply.append( net.res_cost / (net.res_sgen.at[1, 'p_kw'] + net.res_sgen.at[5, 'p_kw'] +
                                        net.res_sgen.at[4, 'p_kw'] + net.res_storage.at[3, 'p_kw']
                                        + net.res_ext_grid.at[0, 'p_kw']))


    print("#########cost###########")
    print(net.res_cost / (net.res_load.at[0, 'p_kw'] + net.res_load.at[1, 'p_kw']))

total_gen = [x + y + z + e for x, y, z, e in zip(p_kw_PV, p_kw_wind, p_kw_CHP, p_kw_ext)]
total_cons = [x + y for x, y in zip(p_kw_TEL, p_kw_charging_station)]
x = np.arange(24)

f, axarr = plt.subplots(3, sharex=True)

axarr[0].plot(x, total_gen)
axarr[0].plot(x, total_cons)
#axarr[0].plot(x, cost2)
axarr[0].legend(['supply','demand'], loc='upper left')
axarr[0].set_ylabel('power (kw)')



#ax2 = axarr[0].twinx()
#axarr[2].plot(x, total_cost)


axarr[1].plot(x, p_kw_PV)
axarr[1].plot(x, p_kw_wind)
axarr[1].plot(x, p_kw_CHP)
axarr[1].plot(x, p_kw_ext)
#axarr[1].plot(x, total_cost)
axarr[1].legend(['PV','Wind','CHP', 'Ext','cost'], loc='lower left')
axarr[1].set_ylabel('power (kw)')

axarr[2].plot(x,cost_demand)
#axarr[2].plot(x,cost_supply)

#axarr[2].legend(['cost for consumption', 'cost for generation'], loc='upper left')
axarr[2].set_ylabel('price (€/kwh)')
axarr[2].set_xlabel('time (hour)')
plt.show()
