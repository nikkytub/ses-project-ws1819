# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 10:09:38 2018

@author: claud
"""
import numpy as np;
from math import sin, cos, sqrt, atan2, radians;

#map constraint
max_long = 100; # 52-53
max_lat = 100; #12.9-13.9
min_long = 52;
min_lat = 12.9;
# car representet as random distribution funktion --> random langitude and latidude
#AC_long = np.random.random(1);
#AC_lat = np.random.random(1);
# alternative: random function generates random vector that represent a car with only 20% of charge
# random number amount between 0 and 1
AC_place = np.random.random(2) + (min_long,min_lat);

# car needs parameters
capacity  = 75; # example for Tessla S modal, BMWi3 ; smart60kW fortwo 17,6kWh
possible_distance = 400 # Tessla s Modal; 260km BMWi3 (daily distance)
#Stromverbrauch in kWh/100 km: 14,6 - 14,0
powerpdist = 14;#power per distance --> kWh per 100km
powtodist = powerpdist/100; #power [kW] need for 1km; theoretically something that might be measured in the car andoptimized by the car itself

# start condition for an autonomous car popping up
charge = 0.2*capacity;
energy_need = capacity-charge;#ask for energy
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

# optimize to the min_distance
def decide(grid):
    decission = charge / powtodist; # grid has to  be in a reachable distance
    for n in range(len(grid)):
        check = find_distance_km(grid[n][0],grid[n][1],AC_place[0],AC_place[1]);
        print('distance to grid number',n,':',check,'km');
        #grid = matrix of different grid with long and lat in one row(n*2 Matrix)
        if check < decission:
            decission = check;
            grid_number = n;
    print('car position',AC_place);
    print('grid positions',grid);
    return decission,grid_number   
# constraints cost for standing (charging)/time (when parameters of grid changes need)

# price for charging: distance to charginstation + chargin itself --> cost comparision --> minimize



