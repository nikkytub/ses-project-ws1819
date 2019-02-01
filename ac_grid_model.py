import numpy as np
from math import sin, cos, sqrt, atan2, radians,inf
from database import get_cars, get_grids, get_car, get_grid
from math import sin, cos, sqrt, atan2, radians, inf
import random
from apscheduler.schedulers.background import BackgroundScheduler as scheduler
import apscheduler
import pandas as pd
from pulp import *


def find_distance_km(lat1, lng1, lat2, lng2):
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


def calc_energy_need(state_of_charge, capacity):
    left_capacity = state_of_charge * capacity
    energy_need = capacity - left_capacity  # ask for energy
    return left_capacity, energy_need


def optimize_cost(car, grids):
    grid_number = -1
    cost_opt = 0.99 * car['capacity']  # price in €
    energy_need = calc_energy_need(car['soc'], car['capacity'])[1]
    print("The amount of needed energy is about ", energy_need, "kWh")
    for n in range(len(grids)):
        cost = energy_need * grids[n]['price']
        print("cost [€]for charging @ grid number", n, ":", cost)
        if (cost < cost_opt) & (grids[n]['capacity'] > energy_need):
            cost_opt = cost
            grid_number = n+1
            print('optimal cost', cost_opt)
            print('grid number', grid_number)
    return cost_opt, grid_number


def optimize_distance(car, grids):
    grid_number = -1
    # grid has to  be in a reachable distance
    decision = calc_energy_need(car['soc'], car['capacity'])[0] / car['powerPD']
    print('reachable distance in km:', decision)

    for n in range(len(grids)):
        check = find_distance_km(grids[n]['lat'], grids[n]['lon'], car['lat'], car['lon'])
        # debug:
        print('distance [km] to grid number', n, ':', check)
        # grid = matrix of different grid with long and lat in one row(n*2 Matrix)
        if check < decision:
            decision = check
            grid_number = n+1
        # debug: print('car position',self.AC_location);
        # debug: print('grid positions',grid);

    return decision, grid_number


def get_optimal(car, mode):
    grid_number = -1
    grids = get_grids()
    if mode == "Time Saving":
        decision, grid_number = optimize_distance(car, grids)
        print(decision,"---",grid_number)

    elif mode == "Money Saving":
        cost_opt, grid_number = optimize_cost(car, grids)
        print(cost_opt,"---",grid_number)

    else:
        print("Wrong mode")
    return get_grid(grid_number)


def reachable_grids(ac, grids):
    final_grids = []
    for x in range(len(grids)):
        # calculate total energy to grid considering distance
        grids[x]['dist'] = find_distance_km(grids[x]['lat'], grids[x]['lon'], ac['lat'], ac['lon'])
        c_to_grid = ac['powerKm']*grids[x]['dist']
        # consider only reachable grids
        if c_to_grid <= (ac['soc']*ac['capacity']):
            # add the energy needed to reach grid to the total deficit
            grids[x]['total_charge_needed_at_grid'] = c_to_grid+(ac['capacity']-(ac['soc']*ac['capacity']))
            # energy_deficit.append(total_charge[0])
            grids[x]['price'] = random.uniform(0.4, 0.8)
            # location, total energy needed at grid location, distance, price
            final_grids.append(grids[x])
    return final_grids


def optimize(grids, mode):
    prob = LpProblem("AC", LpMinimize)
    variables = []
    alphas = []
    distances = []
    supercharges = []
    totalcharges = []
    prices = []
    timeToGrids = []
    ### every x represents a boolean variable wether to pick a grid
    for x in range(len(grids)):
        variables.append(LpVariable(str(x), 0, 1, LpInteger))
        alphas.append(grids[x]["alpha"])
        distances.append(grids[x]["dist"])
        supercharges.append(grids[x]["super_charge"] * 120)
        totalcharges.append(grids[x]["total_charge_needed_at_grid"])
        prices.append(grids[x]["price"])
        timeToGrids.append((grids[x]["dist"] / 50) * 60)
    ### constraints
    if mode == "eco_mode":
        alphadist = np.multiply(alphas, distances)
        prob += lpSum(lpDot(variables, alphadist))
    if mode == "costSaving_mode":
        priceAndCharge = np.multiply(prices, totalcharges)
        prob += lpSum(lpDot(variables, priceAndCharge))
    # we assume a general charging time of 120 min, if supercharge true it is only 60 min
    if mode == "chargingtime_mode":
        superchargeTime = np.add(timeToGrids, supercharges)
        prob += lpSum(lpDot(variables, superchargeTime))

    prob += lpSum(variables) == 1
    prob.writeLP("AC.lp")
    prob.solve()

    for x in range(len(variables)):
        if (value(variables[x]) == 1):
            print("Driving to grid at... ", grids[x]["lat"], grids[x]["lon"])
            print("Car fully charged! 100% battery!")
            return grids[x]
