import numpy as np;
from math import sin, cos, sqrt, atan2, radians, inf;
import random;
from apscheduler.schedulers.background import BackgroundScheduler as scheduler
import apscheduler
import random
import pandas as pd
from pulp import *


class Autocar:
    def __init__(self, AC_location=0, state_of_charge=-1, min_lat=52,
                 min_long=12.9, capacity=75, powerpdist=14):
        # map constraint
        self.AC_location = np.random.random(2) + (min_lat, min_long)
        self.min_lat = min_lat;
        self.min_long = min_long;
        # car representet as random distribution funktion --> random langitude and latidude
        # AC_long = np.random.random(1);
        # AC_lat = np.random.random(1);
        # alternative: random function generates random vector that represent a car with only 20% of charge
        # random number amount between 0 and 1
        # car needs parameters
        self.capacity = capacity;  # example for Tessla S modal, BMWi3 ; smart60kW fortwo 17,6kWh
        # possible_distance = 400 # Tessla s Modal; 260km BMWi3 (daily distance)
        # Stromverbrauch in kWh/100 km: 14,6 - 14,0
        self.powerpdist = powerpdist;  # power per distance --> kWh per 100km
        # powtodist = powerpdist/100; #power [kWh] need for 1km; theoretically something that might be measured in the car andoptimized by the car itself
        self.powerkm = powerpdist / 100
        if state_of_charge < 0:
            self.state_of_charge = round(random.uniform(0.2, 1),
                                         3);  # state of battery between 0 and 1 --> if it is smaller 0.2 it will check where it can charge its battery
        else:
            self.state_of_charge = state_of_charge;

    def show_attributes(self):
        print('location:', self.AC_location, ' (min_lat:', self.min_lat, 'min_long:', self.min_long,
              ')\ncapacity [kWh]:',
              self.capacity, '\npower consumption per km:', self.powerkm, '\nstate of charge [% of capacity]:',
              self.state_of_charge * 100)


    def find_distance_km(self, lat1, lng1, lat2, lng2):
        EARTH_RADIUS = 6373.0
        lat1 = radians(lat1)
        lng1 = radians(lng1)
        lat2 = radians(lat2)
        lng2 = radians(lng2)

        dlon = lng2 - lng1
        dlat = lat2 - lat1

        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        return EARTH_RADIUS * c


class Grid:
    def __init__(self, grid_location=0, total_charge_needed_at_grid=0, dist=0, price=0, alpha=0, supercharge=1,
                 name="default"):
        # map constraint
        self.grid_location = np.random.random(2) + (min_lat, min_long)
        self.total_charge_needed_at_grid = total_charge_needed_at_grid
        self.dist = dist
        self.price = price
        self.name = name
        self.alpha = random.uniform(0, 1)
        # 0.5 means charging time is 1/2 and 1 means normal charging time
        self.supercharge = np.random.choice([0.5, 1])


#############################
### linear Programming#######
#############################

def optimize(grids, mode):
    prob = LpProblem("AC", LpMinimize)
    variables = []
    alphas = []
    distances = []
    p_chargingStation = []
    totalcharges = []
    prices = []
    timeToGrids = []

    ### every x represents a boolean variable wether to pick a grid
    for x in range(len(grids)):
        variables.append(LpVariable(str(x), 0, 1, LpInteger))
        alphas.append(grids[x].alpha)
        distances.append(grids[x].dist)
        p_chargingStation.append(grids[x].p_charging_station)
        totalcharges.append(grids[x].total_charge_needed_at_grid)
        prices.append(grids[x].price)
        timeToGrids.append((grids[x].dist / 50) * 60)
    ### constraints
    if mode == "eco_mode":
        alphacharge = np.multiply(alphas, totalcharges)
        prob += lpSum(lpDot(variables, alphacharge))
    if mode == "costSaving_mode":
        priceAndCharge = np.multiply(prices, totalcharges)
        prob += lpSum(lpDot(variables, priceAndCharge))
    if mode == "chargingtime_mode":
        stationChargingTime = np.divide(totalcharges, p_chargingStation)
        chargingTime = np.add(timeToGrids, stationChargingTime)
        prob += lpSum(lpDot(variables, chargingTime))

    prob += lpSum(variables) == 1
    prob.writeLP("AC.lp")
    prob.solve()

    for x in range(len(variables)):
        if (value(variables[x]) == 1):
            print("Driving to grid at... ", grids[x].grid_location)
            print("Car fully charged! 100% battery!")
            return grids[x].grid_location


min_lat = 52
min_long = 12.9
deduce = 0.1

def create_vehicles(num):
    acs = []
    for x in range(num):
        acs.append(Autocar(capacity = random.uniform(50,120), powerpdist = random.uniform(5, 20)))
    print (acs)
    return acs

def create_grids(num):
    grids_loc = []
    for x in range(num):
        grids_loc.append(Grid(name="grid "))
    print (grids_loc)
    return grids_loc

def reachable_grids(ac, grids):
    final_grids = []
    for x in range(len(grids)):
        ### calculate total energy to grid considering distance
        grids[x].dist = ac.find_distance_km(grids[x].grid_location[0], grids[x].grid_location[1], ac.AC_location[0], ac.AC_location[1])
        c_to_grid = ac.powerkm*grids[x].dist
        ### consider only reachable grids
        if (c_to_grid <= (ac.state_of_charge*ac.capacity)):
            ### add the energy needed to reach grid to the total deficit
            grids[x].total_charge_needed_at_grid = c_to_grid+(ac.capacity-(ac.state_of_charge*ac.capacity))
            #energy_deficit.append(total_charge[0])
            grids[x].price = random.uniform(0.4,0.8)
            ### location, total energy needed at grid location, distance, price
            final_grids.append(grids[x])
    print("Grids within reach: ", len(final_grids))
    return final_grids

def drive_vehicle(ac, grids, modus):
        # assuming v = 50km/h
        soc = random.uniform(0.4,1)
        code = 0
        grid_loc = [0,0]
        ### available energy
        powerstate = ac.state_of_charge*ac.capacity
        ### needed energy
        consumption = 0.9*ac.powerpdist
        if (powerstate-consumption > 0):
            ### available energy - needed energy
            powerupdate = powerstate-consumption
            ### percentage battery left (new state of charge)
            soc_update = round((powerupdate/ac.capacity), 2)
            if (soc_update <= 0):
                print(soc_update)
                print("Battery below 1%.. Randomly resetting SOC to ", soc)
            if (soc_update <= 0.2):
                print("Searching for a charging station..")
                grid_loc = optimize(grids,  modus)
                soc = 1.0
                ### code for update within reach
                code = 1
                ac.AC_location = grid_loc
            else:
                soc = soc_update
                # update AC object values
                ac.state_of_charge = soc
                ac.AC_location[0] = ac.AC_location[0]-deduce
                ac.AC_location[1] = ac.AC_location[1]-deduce
        else:
            print("Car out of power.. Randomly resetting SOC to ", soc)
        ac.state_of_charge = soc
        # make sure car never leaves Berlin area :)
        if (ac.AC_location[0]< min_lat or ac.AC_location[1] < min_long):
            ac.AC_location = np.random.random(2) + (min_lat,min_long)
        #print("new ac location at " , ac.AC_location)
        # return the code for wether to update the location of the car, the current soc and the chosen grid_loc
        return [code, grid_loc]


def main(numAC, numGrids, modus):
    print("Enter 'b' for stopping and 's' for starting the vehicle. Enter 'r' to restart the program!")
    ### Create vehicles
    vehicles = create_vehicles(numAC)
    ### Create grids
    grids = create_grids(numGrids)
    ### choose one car to work with
    ac = vehicles[0]
    ### Find grids which can be reached with current state of charge
    grids_within_reach = reachable_grids(ac, grids)

    ### scedule Loop
    sched = scheduler()
    sched.start()

    # scheuler listener, listenes to scheduler events
    def listener(event):
        # !!!! successfull iteration of the job !!!!
        if (event.code == 4096):
            ###########################################
            # store and print updated State of Charge ##
            ###########################################
            current_SOC = event.retval[1]
            ### update reachable grids
            grids_within_reach = reachable_grids(ac, grids)
            print("CURRENT SOC: ", ac.state_of_charge)
            # print(ac.state_of_charge)

            if (event.retval[1][0] != 0):
                ########################################################
                ##### location of chosen grid during optimization ######
                ########################################################
                chosen_grid_location = event.retval[1]

                # if event.retval[0]==1:
            # randomly update location after charging process
            # ac.AC_location = np.random.random(2) + (min_lat,min_long)

    sched.add_job(drive_vehicle, 'interval', args=[ac, grids_within_reach, modus], seconds=1)
    sched.add_listener(listener)

    while True:
        if input() == 'b':
            sched.pause()
            print("Stopping Operation. Shutting down vehicle...")
        if input() == 's':
            sched.resume()
            print("Starting Operation. Powering up vehicle...")
        if input() == 'r':
            sched.shutdown(wait=False)
            print("total shutdown...")
            main()
            print("re-powerying the system...")




############################################################
###### RUN SIMULATION ######################################
############################################################
###### Number of vehicles, Number of Grids, Modus ##########
############################################################
## Modi: "eco_mode", "costSaving_mode","chargingtime_mode"##
############################################################

main(1, 5, "chargingtime_mode")
#TEST
