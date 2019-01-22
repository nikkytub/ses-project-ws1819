# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 10:09:38 2018

@author: claud
"""
import numpy as np;
from math import sin, cos, sqrt, atan2, radians,inf;

class Autocar:   
    def __init__(self,AC_location = 0,state_of_charge = -1,min_lat = 52,min_long = 12.9,capacity = 75, powerpdist = 14):
        #map constraint
        self.AC_location = np.random.random(2) + (min_lat,min_long)
        self.min_lat = min_lat;
        self.min_long = min_long;
        # car representet as random distribution funktion --> random langitude and latidude
        #AC_long = np.random.random(1);
        #AC_lat = np.random.random(1);
        # alternative: random function generates random vector that represent a car with only 20% of charge
        # random number amount between 0 and 1
        # car needs parameters
        self.capacity  = capacity; # example for Tessla S modal, BMWi3 ; smart60kW fortwo 17,6kWh
        #possible_distance = 400 # Tessla s Modal; 260km BMWi3 (daily distance)
        #Stromverbrauch in kWh/100 km: 14,6 - 14,0
        self.powerpdist = powerpdist;#power per distance --> kWh per 100km
        #powtodist = powerpdist/100; #power [kW] need for 1km; theoretically something that might be measured in the car andoptimized by the car itself
        self.powerkm = powerpdist/100
        if state_of_charge < 0:    
            self.state_of_charge = np.random.random(1); # state of battery between 0 and 1 --> if it is smaller 0.2 it will check where it can charge its battery
        else:
            self.state_of_charge = state_of_charge;
            
            
    def show_attributes(self):
        print('location:',self.AC_location,' (min_lat:', self.min_lat,'min_long:', self.min_long,')\ncapacity [kWh]:',
              self.capacity,'\npower consumption per km:', self.powerkm,'\nstate of charge [% of capacity]:',self.state_of_charge*100 )
        
    # start condition for an autonomous car popping up
    def calc_energy_need(self):
            left_capacity = self.state_of_charge *self.capacity;
            energy_need = self.capacity-left_capacity;#ask for energy
            return left_capacity,energy_need
    #availability of grid with constrains is it reachable (charge > energy_need*distance)
    #f(available?,price?,energy_need)
    
    #distance to grid; output: distance in m
    #def distance_to_grid(glong,glat,clong,clat):
    #   distance =math.sqrt(math.fabs(math.pow(glong-clong,2)+math.pow(glat-clat,2)));
    #  return distance
    def find_distance_km(lat1,lng1,lat2,lng2):
        EARTH_RADIUS = 6373.0
        lat1 = radians(lat1)
        lng1 = radians(lng1)
        lat2 = radians(lat2)
        lng2 = radians(lng2)
        
        dlon = lng2 - lng1
        dlat = lat2 - lat1
        
        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        
        return EARTH_RADIUS * c 
    
    # optimize Cost
    def optimize_cost(self,grid,grid_energy_price,grid_capacity = inf):
        cost_opt = 0.99*self.capacity; # price in €
        energy_need = Autocar.calc_energy_need(self)[1];
        print("The amount of needed energy is aboout ",energy_need,"kWh");
        for n in range(len(grid)):
            cost = energy_need * grid_energy_price[n];
            print("cost [€]for charging @ grid number",n,":", cost);
            if (cost < cost_opt) & (grid_capacity > energy_need):
                cost_opt = cost;
                grid_number = n;
                print('optimal cost',cost_opt);
                print('grid number',grid_number);  
        return cost_opt, grid_number
    
    # optimize to the min_distance 
    def optimize_distance(self,grid):
        decission = Autocar.calc_energy_need(self)[0] / self.powerkm; # grid has to  be in a reachable distance
        print('reachable distance in km:', decission)
        lat = 0; long = 1;
        for n in range(len(grid)):
            check = Autocar.find_distance_km(grid[n][lat],grid[n][long],self.AC_location[0],self.AC_location[1]);
            #debug:
            print('distance [km] to grid number',n,':',check);
            #grid = matrix of different grid with long and lat in one row(n*2 Matrix)
            if check < decission:
                decission = check;
                grid_number = n;
               # debug: print('car position',self.AC_location);
               # debug: print('grid positions',grid);
                
        return decission,grid_number   
            # constraints cost for standing (charging)/time (when parameters of grid changes need)
            
            # price for charging: distance to charginstation + chargin itself --> cost comparision --> minimize
            
    def optimize_ChargingTime(self,grid,grid_chargingStation,avg_velocity,t_charging_max):
        # grid and grid_charginStations information delivered by grid modal; avg_velocity estimation/measurement, t_chagring_max from user choosed constrain
        max_distance = Autocar.calc_energy_need(self)[0] / self.powerkm; # grid has to be in reachable distace according to state of charge in km
        print('estimate reachable distance in km:', max_distance);
        lat = 0; long = 1; # defined by grid parameter -->has to be changed if information will be delivered in a different way
        t_charge_opt = t_charging_max; # time alway calculated in h --> for a better use can be changed in min (*60)
        if (avg_velocity == 0): # velocity vannnot be 0 because division by 0 is not possible
            avg_velocity = 50; # there could also be the average velocity for the area where the car is located
        change = False;
        for n in range(len(grid)):
            if (grid_chargingStation[n] == 0): 
                continue; # if there is no power at the charging station the grid will be ignored
            t_reach = Autocar.find_distance_km(grid[n][lat],grid[n][long],self.AC_location[0],self.AC_location[1]) / avg_velocity;
            t_charging = Autocar.calc_energy_need(self)[1] / grid_chargingStation[n];
            t_charge = t_charging + t_reach;
            print("reach_time [h]:", t_reach,"charging time[h]:",t_charging, "grid_number[h]:",n)
            if (t_charge < t_charge_opt):
                t_charge_opt = t_charge;
                grid_number = n;
                change = True;
                print(t_charge)
        if (change):
            return t_charge_opt, grid_number;
        else:
           # message = 
            return print("There is no charging station in reachable distance that fits to your maximum charging time")
#+time optimization
