# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 13:11:30 2019

@author: Leo77
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 11:35:08 2019

@author: Leo77
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 10:16:12 2019

@author: luisarahn
"""

#Import packages:
import pandapower as pp
from pandapower.plotting.plotly import simple_plotly
from pandapower.plotting.plotly import vlevel_plotly
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from threading import Timer
import random

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
pv_levelized = 0.1
pv_margin = 0.01
wind_levelized = 0.19
wind_margin = 0.01
battery_levelized = 0.1
battery_margin = 0.01
external = 0.2944
external_margin = 0.01

#CO2 index
extIndex = 0.53096
pvIndex = 0.00615
windIndex = 0.00074




zoo_load=[]
house_load=[]
school_load=[]
gym_load=[]
hall_load=[]
priceAll = np.zeros((5,1))  #5*n array. Each row indicates one prosumer. At the end of each time step, for each row, only the last two values are needed to be pushed to the database.
co2All = np.zeros((5,1))    #And the same for the price.
    

def initializeprosumer (prosumer_type, amount):

    #create vector including all gird-names (e.g.: household1, household2, ...):
    number=np.arange(0,amount,1)
    for line in number:
        names.append(prosumer_type+'_'+str(line+1))
    
    #Create random location:
        min_lat=52
        min_long=12.9
        delta_lat=np.random.random(1)
        delta_long=np.random.random(1)
        loc_lat.append(min_lat+delta_lat[0])
        loc_long.append(min_long+delta_long[0])
    
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

    #define parameters:
    charging_station1=[]
    charging_station2=[]
    p_charging_station_total=[]
 
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
    price = []
    price_demand = []
    co2 = []

    available1 = []
    available2 = []
            
    #Charging station types
    P_charging1=4.6 #in kW
    P_charging2=22  #in kW
    fluc_external = random.uniform(-external_margin, external_margin)
    
    for j in range(0,len(names)):
        print('--------------------------------',names[j],'----------------------------------')   
        
        #Prosumer_type
        prosumer_type = names[j].split('_')[0]
        #Checking current state for availability:
        # index according to current timestep
        index1=month*30*24+day*24+hour+1 
        
        if prosumer_type == 'p1':
            Pn_PV=5
            Pn_wind=0
            pv_gen=PV_gen_normalized[index1]*Pn_PV
            wind_gen=wind_gen_normalized[index1]*Pn_wind        
            c_battery=2.5 #in kWh
            building_load=house_load[index1]
            number_chargingstation1=1
            number_chargingstation2=0
            charging_station1=random.choice([0,1])
            charging_station2=0
        
        if prosumer_type == 'p2':
            Pn_PV=20
            Pn_wind=6
            pv_gen=PV_gen_normalized[index1]*Pn_PV
            wind_gen=wind_gen_normalized[index1]*Pn_wind
            c_battery=15 #in kWh
            building_load=school_load[index1]
            #print("current load: ", Office_load[index1])
            number_chargingstation1=5
            number_chargingstation2=0
            charging_station1=random.choice([0,1,2,3,4,5])
            charging_station2=random.choice([0,1])
            
        if prosumer_type == 'p3':
            Pn_PV=60
            Pn_wind=15
            pv_gen=PV_gen_normalized[index1]*Pn_PV
            wind_gen=wind_gen_normalized[index1]*Pn_wind
            c_battery=35 #in kWh
            building_load=zoo_load[index1]
            number_chargingstation1=5
            number_chargingstation2=3
            charging_station1=random.choice([0,1,2,3,4,5])
            charging_station2=random.choice([0,1,2,3])
            
        if prosumer_type == 'p4':
            Pn_PV=5
            Pn_wind=0
            pv_gen=PV_gen_normalized[index1]*Pn_PV
            wind_gen=wind_gen_normalized[index1]*Pn_wind
            c_battery=2.5 #in kWh
            building_load=gym_load[index1]
            number_chargingstation1=0
            number_chargingstation2=2
            charging_station1=0
            charging_station2=random.choice([0,1,2])
            
        if prosumer_type == 'p5':
            Pn_PV=10
            Pn_wind=0
            pv_gen=PV_gen_normalized[index1]*Pn_PV
            wind_gen=wind_gen_normalized[index1]*Pn_wind
            c_battery=5 #in kWh
            building_load=hall_load[index1]
            number_chargingstation1=6
            number_chargingstation2=0
            charging_station1=random.choice([0,1,2,3,4,5,6])
            charging_station2=0

            
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
                 
        #Index according to next timestep (+1h)
        #If optimization is run for multiple hours (if hourmax>1: [index:index+hourmax] instead of [index])
        
        
        #Run optimization for hourmax (if hourmax>1: add [] for pv_gen)
        
            
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
#        sw1 = pp.create_switch(net, bus0, element=0, et='t', closed="False", type="CB", name="CBT1")
#        sw2 = pp.create_switch(net, bus0, element=0, et='t', closed="False", type="CB", name="CBT2")
        
        #create generators (p_kW<0 -> generation)
        grid = pp.create_ext_grid(net, bus0, index=0)
        pv = pp.create_sgen(net, bus2, p_kw=-pv_gen, # for multiple hours: p_kw=-pv_gen[i] min_p_kw=-max(pv_gen), max_p_kw=0,
                            min_q_kvar=-1, max_q_kvar=0., controllable=False, index=1)
        wind = pp.create_sgen(net, bus4, p_kw=-wind_gen, # for multiple hours: p_kw=-wind_gen[i] min_p_kw=-max(wind_gen), 
                              min_q_kvar=-1, max_q_kvar=0., max_p_kw=0, controllable=False, index=2)
        
        
        #generate cost profile
        pvCost = pv_levelized + random.uniform(-pv_margin, pv_margin)
        windCost = wind_levelized + random.uniform(-wind_margin, wind_margin)
        batteryCost = battery_levelized + random.uniform(-battery_margin, battery_margin)
        externalCost = external + fluc_external
        
#        #battery time-dependency (SOC): decision making on whether to charge (a=1) or discharge battery (a=-1)
        #ext_cost_average=0.2944
        
        #Total power of charging station for grid optimization:
        p_charging_station_total=charging_station1*P_charging1+charging_station2*P_charging2
        
        # for multiple hours: p_charging_station_total[i]=charging_station1[i]*P_charging1+charging_station2[i]*P_charging2 
       
        #Creating loads: postive value -> load, negative value -> generation
        
        charging_station = pp.create_load(net, bus3, p_kw=p_charging_station_total,
                                          # for multiple hours: p_kw=p_charging_station_total[i], 
                                          name='charging station', #min_p_kw = 4.6, max_p_kw = 90,#min_p_kw=min(p_charging_station_total), max_p_kw=max(p_charging_station_total),
                                          min_q_kvar=0., max_q_kvar=1.0, controllable=False, index=0)
        
        buidling = pp.create_load(net, bus5, p_kw=building_load, #for multiple hours: p_kw=building_load[i],
                                  name='building', min_p_kw=building_load*0.9, max_p_kw=building_load*1.1,
                                  min_q_kvar=0., max_q_kvar=2.0, controllable=True, index=1)
        
        #if ((SOC[j] < 0.9) and (externalCost <= ext_cost_average)): #for multiple hours: ext_cost[i]
        if  ((SOC[j] < 0.9) and (building_load+p_charging_station_total <= pv_gen + wind_gen)):
            a=1 
            min_p_kw = 0
            max_p_kw = 1/8*c_battery
       # elif (SOC[j] > 0.1) and (externalCost > ext_cost_average): #for multiple hours: ext_cost[i]
        elif ((SOC[j] > 0.1) and (building_load+p_charging_station_total > pv_gen + wind_gen)):
            a=-1
            min_p_kw = -1/8*c_battery
            max_p_kw = 0
        else:
            a=0
            min_p_kw = 0
            max_p_kw = 0
            
        
        print('a', a)
    
        #create battery (SOC is timedependent), momentary real power of the storage (positive for charging, negative for discharging)
        battery = pp.create_storage(net, bus6, p_kw=a*1/10*c_battery, max_e_kwh=c_battery, controllable=True, max_p_kw =max_p_kw
                                    , min_p_kw =min_p_kw, max_q_kvar= max_p_kw, min_q_kvar= min_p_kw, index=3)
        
        
        
        
    
        #generator cost function (costs in Euro/kWh)
        pp.create_polynomial_cost(net, grid, 'ext_grid', np.array([-externalCost,0]))
        pp.create_polynomial_cost(net, pv, 'sgen', np.array([-pvCost,0]))
        pp.create_polynomial_cost(net, wind, 'sgen', np.array([-windCost,0]))
        pp.create_polynomial_cost(net, battery, 'storage', np.array([-batteryCost,0]))
        
        '''
        pp.create_polynomial_cost(net, grid, 'ext_grid', np.array([-ext_cost[index1], 0]))
        pp.create_polynomial_cost(net, pv, 'sgen', np.array([(-pv_cost[index1]), 0]))
        pp.create_polynomial_cost(net, wind, 'sgen', np.array([(-wind_cost[index1]), 0]))
        pp.create_polynomial_cost(net, battery, 'storage', np.array([-ext_cost[index1], 0]))
        '''
        
        #load cost function
        #pp.create_polynomial_cost(net, charging_station, 'load', np.array([-0.4, 0]))
        #pp.create_polynomial_cost(net, buidling, 'load', np.array([-0.4, 0]))
    
        # run power flow and optimal power flow
        #pp.runpp(net)
        
        
        ##pp.runopp(net, verbose=False, numba=False)
        ##pp.runopp(net)
        
        ##pp.rundcpp(net)
        pp.rundcopp(net, OPF_ALG_DC=200, verbose=False)
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
        #
        print('p_kw_bat', str(net.res_storage.at[3, 'p_kw']))
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
        #alpha_current[j]=alpha_next[j]
        
        #(for charging: a=1)
        if a == 1:
            price_demand = -costs[j] / (p_kw_charging_station[j] + p_kw_building[j] + p_kw_battery[j])
            #alpha_next[j]=(abs(p_kw_PV[j]) + abs(p_kw_wind[j])) / (abs(p_kw_charging_station[j])+ abs(p_kw_building[j])+ abs(p_kw_battery[j]))
            
        else:
            price_demand = -costs[j] / (p_kw_charging_station[j] + p_kw_building[j])
            #alpha_next[j]=(abs(p_kw_PV[j])+abs(p_kw_wind[j])+abs(p_kw_battery[j])) /(abs(p_kw_charging_station[j])+ abs(p_kw_building[j]))
        
        
        #charging price in Euros/kWh (including 10% benefit)
        price_current[j]=price_next[j]
        
        if price_demand < 0:
            price_next[j]=price_demand*(-1.1)
        elif net.res_cost > 0:
            price_next[j]=price_demand*(-0.9)
        else:
            price_next[j]=0.1
        
        price.append(price_next[j])
        if net.res_ext_grid.at[0, 'p_kw'] > 0:
            extPower = 0
        else:
            extPower = net.res_ext_grid.at[0, 'p_kw']
        
        co2.append((abs(pvIndex*net.res_sgen.at[1, 'p_kw'])+abs(windIndex*net.res_sgen.at[2, 'p_kw'])+abs(extIndex*extPower))/(abs(net.res_sgen.at[1, 'p_kw'])+abs(net.res_sgen.at[2, 'p_kw'])+abs(extPower)))
        #new SOC (currently only makes sence for one hour optimization):
        SOC[j]=(SOC[j]*c_battery+p_kw_battery[j])/c_battery
        #print('p_kw_bat' + str(p_kw_battery[j]))
        if SOC[j]>1:
            SOC[j]=1
        elif SOC[j]<0:
            SOC[j]=0

        print('costs ',net.res_cost)
        print('price for current timestep' + str(price_current[j]))
        print('price for next timestep' + str(price_next[j]))
        print('new state of charge' + str(SOC[j]))
            
        grids.append('{"name":"' + names[j] + '","availability":' + str(available[j]) + ',"lat":' + str(loc_lat[j])+',"lon":' + str(loc_long[j]) + ',"price":' + str(price_current[j]) +',"alpha":' + str(alpha_current[j])
         + ',"p_charging_station":' + str(p_charging_station[j]) +',"price_next":' + str(price_next[j]) +',"alpha_next":' + str(alpha_next[j])+'}')
    print('grids',grids)
    global priceAll
    global co2All
    priceAll = np.c_[priceAll, price]
    co2All = np.c_[co2All,co2]
    print(priceAll)
    
def main(month,day,hour):
    initializeprosumer ('p1', 1)
    initializeprosumer ('p2', 1)
    initializeprosumer ('p3', 1)
    initializeprosumer ('p4', 1)
    initializeprosumer ('p5', 1)
    
    
    #input-forecasting: 
    with open('PV_generationNew.csv', 'r', encoding='UTF-8') as f:
        for line in f:
            PV_gen_normalized.append(float(line)/4)
  
    with open('Wind_generationNew.csv', 'r', encoding='UTF-8') as f:
        for line in f:
            wind_gen_normalized.append(float(line)/2.2)
    
   # with open('result_load_school_zoo_garden.csv', 'r', encoding='UTF-8') as f:
    with open('result_load_school_zoo_gym_hall_garden.csv', 'r', encoding='UTF-8') as f:
        for line in f:
            break;
        for line in f:
            row = line.split(',')
            school_load.append(float(row[1])) 
            zoo_load.append(float(row[2])/2) 
            gym_load.append(float(row[3])) 
            hall_load.append(float(row[4])) 
    
    with open('loadHouseNew.csv', 'r', encoding='UTF-8') as f:
        for line in f:
            house_load.append(float(line)) 
    
    

    
    
    #time delay = time for grid optimization for each timestep (=1h)
    for i in range(0,23,1):
        print('timestep'+str(i))
        t = Timer(1.0, createprosumer(month,day,hour))
        print('--------------------------------',i,'----------------------------------')
        t.start()
        if hour<23:
            hour=hour+1
        else:
            hour=0
            if day<29:
                day=day+1
            else: 
                day=0
                month=month+1

main(0,0,0) #input: month,day,hour
