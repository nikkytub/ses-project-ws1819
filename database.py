import mysql.connector
import json
config = {
    'user': 'root',
    'password': 'root',
    'host': '127.0.0.1',
    'database': 'infrastructure',
    'raise_on_warnings': True
}


def get_cars():
    cnx = mysql.connector.connect(**config)
    cursor = cnx.cursor()
    query = "SELECT * FROM car"
    cursor.execute(query)

    cars_json = [dict((cursor.description[i][0], value)
                      for i, value in enumerate(row)) for row in cursor.fetchall()]

    return cars_json


def get_grids():
    cnx = mysql.connector.connect(**config)
    cursor = cnx.cursor()
    # to change to the previous grids just replace grid2 with grid
    query = "SELECT * FROM grid2"
    cursor.execute(query)
    grids_json = [dict((cursor.description[i][0], value)
                       for i, value in enumerate(row)) for row in cursor.fetchall()]

    return grids_json


def get_car(id):
    cnx = mysql.connector.connect(**config)
    cursor = cnx.cursor()
    query = "SELECT * FROM car WHERE id = " + str(id)
    cursor.execute(query)
    car_json = [dict((cursor.description[i][0], value)
                      for i, value in enumerate(row)) for row in cursor.fetchall()]

    return car_json[0]


def get_grid(id):
    cnx = mysql.connector.connect(**config)
    cursor = cnx.cursor()
    # to change to the previous grids just replace grid2 with grid
    query = "SELECT * FROM grid2 WHERE id = " + str(id)
    cursor.execute(query)
    grid_json = [dict((cursor.description[i][0], value)
                      for i, value in enumerate(row)) for row in cursor.fetchall()]

    return grid_json[0]


def change_car_mode(car):
    cnx = mysql.connector.connect(**config)
    cursor = cnx.cursor()
    query = "UPDATE car SET mode = %s WHERE id = %s"
    val = (car['mode'], car['id'])
    cursor.execute(query,val)
    cnx.commit()
    cursor.close()
    cnx.close()

