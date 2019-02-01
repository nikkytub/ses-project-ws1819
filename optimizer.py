import numpy as np
from math import sin, cos, sqrt, atan2, radians,inf
from database import get_cars, get_grids, get_car, get_grid


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
