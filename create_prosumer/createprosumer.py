#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 10:16:12 2019

@author: luisarahn
"""
import time
#Import packages:
import pandapower as pp
from pandapower.plotting.plotly import simple_plotly
from pandapower.plotting.plotly import vlevel_plotly
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from threading import Timer
from create_prosumer.store_grids import store_grid, update_grid
from database import get_grids
#global variables:
names=[]
loc_lat=[]
loc_long=[]
month=1
day=1
hour=1
grids=[]
SOC=[]
price_current=[]
price_next=[]
alpha_current=[]
alpha_next=[]
available=[]
p_charging_station=[]

pv_cost = []  # Euro/kWh
wind_cost = []  # Euro/kWh
ext_cost = []  # Euro/kWh
total_cost = [] # Euro/kWh
charging_station_household=[]
charging_station_supermarket=[]
charging_station_office2=[]
charging_station_office1=[]
PV_gen_normalized=[]  
wind_gen_normalized=[]
Office_load=[]
house_load=[]
supermarket_load=[]
    

def initializeprosumer (prosumer_type, amount):

    #create vector including all gird-names (e.g.: household1, household2, ...):
    number=np.arange(0,amount,1)
    for line in number:
        names.append(prosumer_type+'_'+str(line+1))
    
    #Create random location:
        min_lat = 52.500430
        max_lat = 52.527547
        min_long = 13.301881
        max_long = 13.386467

        loc_lat.append(np.random.uniform(min_lat,max_lat))
        loc_long.append(np.random.uniform(min_long,max_long))
    
    #SOC of battery (initial condition)
        SOC.append(0.5)
    #price and alpha_current(initial condition)
        price_current.append(0)
        price_next.append(0)
        alpha_current.append(0)
        alpha_next.append(0)
        available.append(0)
        p_charging_station.append(0)
        
    #Initialized time: (current-time)
    month=0
    day=0
    hour=0

def createprosumer(month,day,hour):
#execute: createprosumer(grid_type=[household or supermarket or office],x,t, SOC of timestep before)
    #define parameters:
    charging_station1=[]
    charging_station2=[]
    p_charging_station_total=[]
    current_grids=[]
    p_kw_wind = []
    q_kvar_wind = []
    p_kw_PV = []
    q_kvar_PV = []
    p_kw_battery = []
    q_kvar_battery = []
    p_kw_ext = []
    q_kvar_ext = []
    p_kw_charging_station = []
    q_kvar_charging_station = []
    p_kw_building = []
    q_kvar_building = []
    costs = []
    price_demand = []

    available1 = []
    available2 = []
            
    #Charging station types
    P_charging1=4.6 #in kW
    P_charging2=22  #in kW

    
    for j in range(0,len(names)):
        print('--------------------------------',names[j],'----------------------------------')   
        
        #Prosumer_type
        prosumer_type = names[j].split('_')[0]
        #Checking current state for availability:
        # index according to current timestep
        index0=month*30*24+day*24+hour
        
        #Prosumer Type dependencies:
        if prosumer_type == 'household':
            number_chargingstation1=2
            number_chargingstation2=0
            charging_station1=charging_station_household[index0]
            charging_station2=0
            
        if prosumer_type == 'office':
            number_chargingstation1=10
            number_chargingstation2=0
            charging_station1=charging_station_office1[index0]
            charging_station2=0#charging_station_office2[index0]
            
        if prosumer_type == 'supermarket':
            number_chargingstation1=0
            number_chargingstation2=3
            charging_station1=0
            charging_station2=charging_station_supermarket[index0]        
            
        #Charging station
        #Dependent on prosumer_type -> availability and power of charging station 
        if charging_station1 == number_chargingstation1:
            available1=0
        else:
            available1=1
        if charging_station2 == number_chargingstation2:
            available2=0
        else:
            available2=1
            
        if available2 == 1:
            p_charging_station[j]=P_charging2
            available[j]=1
        elif available1 == 1:
            p_charging_station[j]=P_charging1
            available[j]=1
        else:
            p_charging_station[j]=0
            available[j]=0
                 
        # index according to next timestep (+1h)
        # if optimization is run for multiple hours (if hourmax>1: [index:index+hourmax] instead of [index])
        index1=month*30*24+day*24+hour+1 
        
        if prosumer_type == 'household':
            Pn_PV=5
            Pn_wind=0
            pv_gen=PV_gen_normalized[index1]*Pn_PV
            wind_gen=wind_gen_normalized[index1]*Pn_wind        
            c_battery=10 #in kWh
            building_load=house_load[index1]
            number_chargingstation1=2
            number_chargingstation2=0
            charging_station1=charging_station_household[index1]
            charging_station2=0
        
        if prosumer_type == 'office':
            Pn_PV=10
            Pn_wind=5
            pv_gen=PV_gen_normalized[index1]*Pn_PV
            wind_gen=wind_gen_normalized[index1]*Pn_wind
            c_battery=20 #in kWh
            building_load=Office_load[index1]
            #print("current load: ", Office_load[index1])
            number_chargingstation1=4
            number_chargingstation2=0
            charging_station1=charging_station_office1[index1]
            charging_station2=0#charging_station_office2[index1]
            
        if prosumer_type == 'supermarket':
            Pn_PV=20
            Pn_wind=5
            pv_gen=PV_gen_normalized[index1]*Pn_PV
            wind_gen=wind_gen_normalized[index1]*Pn_wind
            c_battery=30 #in kWh
            building_load=supermarket_load[index1]
            number_chargingstation1=0
            number_chargingstation2=3
            charging_station1=0
            charging_station2=charging_station_supermarket[index1]

        
        #Run optimization for hourmax (if hourmax>1: add [] for pv_gen)
        hourmax=1
        for i in range(0, hourmax):
            print('--------------------------------',i,'----------------------------------')
            
            #creat an empty network (all ne)
            net = pp.create_empty_network()
            #define buses (20 kV: high voltage side, 0.4 kV low voltage side)
            bus0 = pp.create_bus(net, index=0, name="trafo1", vn_kv=20, type='b', zone='A', vm_max_pu=1.05, vm_min_pu=0.95,
                                 in_service=True)
            bus1 = pp.create_bus(net, index=1, name="trafo2", vn_kv=0.4, type='b', zone='A', vm_max_pu=1.05, vm_min_pu=0.95,
                                 in_service=True)
            bus2 = pp.create_bus(net, index=2, name="pv", vn_kv=0.4, type='b', zone='A', vm_max_pu=1.05, vm_min_pu=0.95,
                                 in_service=True)
            bus3 = pp.create_bus(net, index=3, name="wind", vn_kv=0.4, type='b', zone='A', vm_max_pu=1.05, vm_min_pu=0.95,
                                 in_service=True)
            bus4 = pp.create_bus(net, index=4, name="battery", vn_kv=0.4, type='b', zone='A', vm_max_pu=1.05, vm_min_pu=0.95,
                                 in_service=True)
            bus5 = pp.create_bus(net, index=5, name="ev", vn_kv=0.4, type='b', zone='A', vm_max_pu=1.05, vm_min_pu=0.95,
                                 in_service=True)
            bus6 = pp.create_bus(net, index=6, name="building", vn_kv=0.4, type='b', zone='A', vm_max_pu=1.05, vm_min_pu=0.95,
                                 in_service=True)
            # define system lines
            pp.create_line(net, bus1, bus2, length_km=0.3, std_type='NAYY 4x150 SE', name='line1', index=1, in_service=True,
                           max_loading_percent=50)
            pp.create_line(net, bus2, bus3, length_km=0.3, std_type='NAYY 4x150 SE', name='line2', index=2, in_service=True,
                           max_loading_percent=50)
            pp.create_line(net, bus3, bus4, length_km=0.3, std_type='NAYY 4x150 SE', name='line3', index=3, in_service=True,
                           max_loading_percent=50)
            pp.create_line(net, bus4, bus5, length_km=0.3, std_type='NAYY 4x150 SE', name='line4', index=4, in_service=True,
                           max_loading_percent=50)
            pp.create_line(net, bus5, bus6, length_km=0.3, std_type='NAYY 4x150 SE', name='line5', index=5, in_service=True,
                           max_loading_percent=50)
            pp.create_line(net, bus6, bus1, length_km=0.3, std_type='NAYY 4x150 SE', name='line6', index=6, in_service=True,
                           max_loading_percent=50)
        
            #create transformer
            pp.create_transformer(net, bus0, bus1, name='trafoDAI', std_type="0.63 MVA 20/0.4 kV", max_loading_percent=50)
            #create switches
            sw1 = pp.create_switch(net, bus0, element=0, et='t', closed="False", type="CB", name="CBT1")
            sw2 = pp.create_switch(net, bus0, element=0, et='t', closed="False", type="CB", name="CBT2")
            
            #create generators (p_kW<0 -> generation)
            grid = pp.create_ext_grid(net, bus0, index=0)
            pv = pp.create_sgen(net, bus2, p_kw=-pv_gen, # for multiple hours: p_kw=-pv_gen[i] min_p_kw=-max(pv_gen), max_p_kw=0,
                                min_q_kvar=-1, max_q_kvar=0., controllable=False, index=1)
            wind = pp.create_sgen(net, bus4, p_kw=-wind_gen, # for multiple hours: p_kw=-wind_gen[i] min_p_kw=-max(wind_gen), 
                                  min_q_kvar=-1, max_q_kvar=0., max_p_kw=0, controllable=False, index=2)
            
            #battery time-dependency (SOC): decision making on whether to charge (a=1) or discharge battery (a=-1)
            ext_cost_average=sum(ext_cost)/len(ext_cost) 
            if ((SOC[j] < 1) and (ext_cost[index0] <= ext_cost_average)): #for multiple hours: ext_cost[i]
                a=1
            elif (SOC[j] > 0) and (ext_cost[index0] > ext_cost_average): #for multiple hours: ext_cost[i]
                a=-1
            else: a=0
            
            #create battery (SOC is timedependent), momentary real power of the storage (positive for charging, negative for discharging)
            battery = pp.create_storage(net, bus6, p_kw=a*1/10*c_battery, max_e_kwh=c_battery, controllable=False, index=3)
            
            #Total power of charging station for grid optimization:
            p_charging_station_total=charging_station1*P_charging1+charging_station2*P_charging2
            # for multiple hours: p_charging_station_total[i]=charging_station1[i]*P_charging1+charging_station2[i]*P_charging2 
           
            #Creating loads: postive value -> load, negative value -> generation
            charging_station = pp.create_load(net, bus3, p_kw=p_charging_station_total,
                                              # for multiple hours: p_kw=p_charging_station_total[i], 
                                              name='charging station',#min_p_kw=min(p_charging_station_total), max_p_kw=max(p_charging_station_total),
                                              min_q_kvar=0., max_q_kvar=1.0, controllable=False, index=0)
            
            buidling = pp.create_load(net, bus5, p_kw=building_load, #for multiple hours: p_kw=building_load[i],
                                      name='building',# min_p_kw=min(building_load), max_p_kw=max(building_load),
                                      min_q_kvar=0., max_q_kvar=2.0, controllable=False, index=1)
        
            #generator cost function (costs in Euro/kWh)
            pp.create_polynomial_cost(net, grid, 'ext_grid', np.array([-ext_cost[index1], 0]))
            pp.create_polynomial_cost(net, pv, 'sgen', np.array([(-pv_cost[index1]), 0]))
            pp.create_polynomial_cost(net, wind, 'sgen', np.array([(-wind_cost[index1]), 0]))
            pp.create_polynomial_cost(net, battery, 'storage', np.array([-ext_cost[index1], 0]))
            #load cost function
            pp.create_polynomial_cost(net, charging_station, 'load', np.array([-0.4, 0]))
            pp.create_polynomial_cost(net, buidling, 'load', np.array([-0.4, 0]))
        
            # run power flow and optimal power flow
            #pp.runpp(net)
            
            
            ##pp.runopp(net, verbose=False, numba=False)
            ##pp.runopp(net)
            
            ##pp.rundcpp(net)
            pp.rundcopp(net, verbose=False)
            print("RES!!!!!!")
            print(net.res_sgen)
            print(net.res_storage)
            print(net.res_load)
            print(net.res_ext_grid)
            #print(net.)
            
            #power distribution
            p_kw_PV.append(net.res_sgen.at[1, 'p_kw'])
            q_kvar_PV.append(net.res_sgen.at[1, 'q_kvar'])
            p_kw_wind.append(net.res_sgen.at[2, 'p_kw'])
            q_kvar_wind.append(net.res_sgen.at[2, 'q_kvar'])
            #for battery: positive for charging, negative for discharging
            p_kw_battery.append(net.res_storage.at[3, 'p_kw'])
            q_kvar_battery.append(net.res_storage.at[3, 'q_kvar'])
            
            # power from/to external grid
            p_kw_ext.append(net.res_ext_grid.at[0, 'p_kw'])
            q_kvar_ext.append(net.res_ext_grid.at[0, 'q_kvar'])
            
            #power consumption
            p_kw_charging_station.append(net.res_load.at[0, 'p_kw'])
            q_kvar_charging_station.append(net.res_load.at[0, 'q_kvar'])
            p_kw_building.append(net.res_load.at[1, 'p_kw'])
            q_kvar_building.append(net.res_load.at[1, 'q_kvar'])
            
            #resulting costs
            costs.append(net.res_cost)
            
            #share of own generated 'green' energy:
            alpha_current[j]=alpha_next[j]

            # if prosumer_type == 'household':
            #     next_alpha = (p_kw_PV[i] + p_kw_wind[i]) / (p_kw_PV[i] + p_kw_wind[i] + p_kw_ext[i]) + 0.1
            # elif prosumer_type == 'office':
            #     next_alpha = (p_kw_PV[i] + p_kw_wind[i]) / (p_kw_PV[i] + p_kw_wind[i] + p_kw_ext[i]) + 0.2
            # else:
            next_alpha = (p_kw_PV[i] + p_kw_wind[i]) / (p_kw_PV[i] + p_kw_wind[i] + p_kw_ext[i])

            if next_alpha > 1:
                alpha_next[j] = 1
            elif next_alpha < 0:
                alpha_next[j] = 0
            else:
                alpha_next[j] = next_alpha

            #(for charging: a=1)
            if a == 1:
                price_demand = -costs[j] / (p_kw_charging_station[j] + p_kw_building[j] + p_kw_battery[j])
            else:
                price_demand = -costs[j] / (p_kw_charging_station[j] + p_kw_building[j])
            
            #charging price in Euros/kWh (including 10% benefit)
            price_current[j]=price_next[j]

            # if prosumer_type == 'household':
            #     price_demand = price_demand + 0.1
            # elif prosumer_type == 'office':
            #     price_demand = price_demand + 0.2

            if price_demand < 0:
                price_next[j] = price_demand*(-1.1)
            elif net.res_cost > 0:
                price_next[j] = price_demand*(-0.9)
            else:
                price_next[j] = 0.1

            #new SOC (currently only makes sence for one hour optimization):
            SOC[j]=(SOC[j]*c_battery+p_kw_battery[i])/c_battery
            if SOC[j] > 1:
                SOC[j] = 1
            elif SOC[j] < 0:
                SOC[j] = 0

            print('costs ',net.res_cost)
            print('price for current timestep' + str(price_current[j]))
            print('price for next timestep' + str(price_next[j]))
            print('new state of charge' + str(SOC[j]))

        current_grids.append('{"name":"' + names[j] + '","availability":' + str(available[j]) + ',"lat":' + str(loc_lat[j])+',"lon":' + str(loc_long[j]) + ',"price":' + str(price_current[j]) +',"alpha":' + str(alpha_current[j])
         + ',"p_charging_station":' + str(p_charging_station[j]) +',"price_next":' + str(price_next[j]) +',"alpha_next":' + str(alpha_next[j])+'}')
    grids.append(current_grids)

    if(len(get_grids())==0):
        store_grid(current_grids)
    else:
        update_grid(current_grids)


    print('grids',grids)
    return current_grids


def main(month,day,hour):
    initializeprosumer ('household', 2)
    initializeprosumer ('office', 2)
    initializeprosumer ('supermarket', 2)
    
    #input-forecasting: 
    with open('PV_generationNew.csv', 'r', encoding='UTF-8') as f:
        for line in f:
            PV_gen_normalized.append(float(line))
    
    with open('Wind_generationNew.csv', 'r', encoding='UTF-8') as f:
        for line in f:
            wind_gen_normalized.append(float(line)+1.0)
    
    with open('LoadOfficeNew.csv', 'r', encoding='UTF-8') as f:
        for line in f:
            Office_load.append((float(line)/10))
    
    with open('loadHouseNew.csv', 'r', encoding='UTF-8') as f:
        for line in f:
            house_load.append(float(line)) 
    
    #input further variables based on assumptions:
    with open('supermarket.csv', 'r', encoding='UTF-8') as f:
        for line in f:
                row = line.replace('.', ',')
                row = row.split(',')
                supermarket_load.append(float(row[2]))    
    
    #input-cost over time
    with open('task3_energy_cost.csv', 'r', encoding='UTF-8') as f:
        for i in range(0,6):
            for line in f:
                    break
            for line in f:
                row = line.replace(' ', ',')
                row = row.split(',')
                #day.append(row[0])
                #hour.append(row[1])
                pv_cost.append(float(row[2]))
                wind_cost.append(float(row[3]))
                ext_cost.append(float(row[5]))
    
    with open('chargingStation.csv', 'r', encoding='UTF-8') as f:
        for line in f:
            break
        for line in f:
            row = line.split(',')
            charging_station_household.append(float(row[0]))
            charging_station_supermarket.append(float(row[1]))          
            charging_station_office2.append(float(row[2]))
            charging_station_office1.append(float(row[3])) 
    
    #time delay = time for grid optimization for each timestep (=1h)
    for i in range(0,2000,1):
        print('timestep'+str(i))
        #t = Timer(5.0, createprosumer(month,day,hour))
        #t.start()
        time.sleep(4)
        createprosumer(month, day, hour)
        if hour<23:
            hour=hour+1
        else:
            hour=0
            if day<29:
                day=day+1
            else: 
                day=0
                month=month+1

main(0,0,0)
