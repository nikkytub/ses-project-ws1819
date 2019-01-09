import mysql.connector

config = {
    'user': 'root',
    'password': '',
    'host': '127.0.0.1',
    'database': 'infrastructure',
    'raise_on_warnings': True
}


def get_cars():
    cnx = mysql.connector.connect(**config)
    cursor = cnx.cursor()
    query = "SELECT * FROM car"
    cursor.execute(query)
    cars_json = [dict((cursor.description[i][0], float(value))
                      for i, value in enumerate(row)) for row in cursor.fetchall()]

    return cars_json


def get_grids():
    cnx = mysql.connector.connect(**config)
    cursor = cnx.cursor()
    query = "SELECT * FROM grid"
    cursor.execute(query)
    grids_json = [dict((cursor.description[i][0], float(value))
                       for i, value in enumerate(row)) for row in cursor.fetchall()]

    return grids_json
