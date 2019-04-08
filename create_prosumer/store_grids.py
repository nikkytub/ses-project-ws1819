import mysql.connector
import json
config = {
    'user': 'root',
    'password': 'root',
    'host': '127.0.0.1',
    'database': 'infrastructure',
    'raise_on_warnings': True
}


def store_grid(grids):
    for grid1 in grids:
        grid = json.loads(grid1)
        n = grid ["name"]
        cnx = mysql.connector.connect(**config)
        cursor = cnx.cursor()
        sql = "INSERT INTO grid(name, availability, lat, lon,price, alpha, p_charging_station,price_next, alpha_next)" \
              "VALUES (%s, %s, %s, %s, %s, %s,%s, %s, %s)"
        val = (str(grid["name"]), str(grid["availability"]), str(grid["lat"]), str(grid["lon"])
               , str(round(grid["price"],2)), str(round(grid["alpha"],2)), str(round(grid["p_charging_station"],2)), str(round(grid["price_next"],2))
               , str(round(grid["alpha_next"],2)))

        cursor.execute(sql, val)
        cnx.commit()
        cursor.close()
        cnx.close()

def update_grid(grids):
    for grid1 in grids:
        grid = json.loads(grid1)
        n = grid ["name"]
        cnx = mysql.connector.connect(**config)
        cursor = cnx.cursor()
        sql = """UPDATE grid SET  availability = %s, price=%s, alpha=%s, p_charging_station=%s,price_next=%s, alpha_next=%s
              WHERE name = %s"""
        val =(str(grid["availability"])
               , str(round(grid["price"],2)), str(round(grid["alpha"],2)), str(round(grid["p_charging_station"],2)), str(round(grid["price_next"],2))
               , str(round(grid["alpha_next"],2)),str(grid["name"]))

        cursor.execute(sql,val)
        cnx.commit()
        cursor.close()
        cnx.close()


'''
grids format
grids = ['{"name":"' + a + '","availability":' + str(b) + ',"lat":' + str(b)+',"lon":' + str(b)
         + ',"price":' + str(b)+',"total_charge_needed_at_grid":' + str(b)+',"alpha":' + str(b)
         + ',"p_charging_station":' + str(b)+',"dist":' + str(b)+',"price_next":' + str(b)+',"alpha_next":' + str(b)+'}']
'''


