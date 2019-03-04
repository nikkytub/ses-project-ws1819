from flask import Flask, render_template, request, jsonify
from database import get_cars, get_grids, get_car, change_car_mode

from optimizer import get_optimal
import json
from ac_grid_model import reachable_grids, optimize
app = Flask(__name__)


@app.route('/', methods=["GET", "POST"])
def index():
    cars = get_cars()
    return render_template('simulation.html', car=cars[0])


@app.route('/simulation')
def simulation():
    s = str(request).split('?')
    car_id = s[1].split('\'')[0].split('=')[1]
    car = get_car(car_id)
    grids = get_grids()
    return render_template('simulation.html', car=car, grids=grids)


@app.route('/postCar', methods=["POST"])
def post_javascript_data():
    response = request.get_data()
    response = response.decode("utf-8").replace("'", '"')
    json_response = json.loads(response)
    car = get_car(int(json_response['id']))
    mode = json_response['mode']
    grids = reachable_grids(car, get_grids())
    #print(grids)
    optimal_grid = get_optimal(car, mode)
    return jsonify(optimal_grid)


@app.route('/change_mode', methods=["POST"])
def change_mode():
    response = request.get_data()
    response = response.decode("utf-8").replace("'", '"')
    json_response = json.loads(response)
    change_car_mode(json_response)

    return jsonify(get_car(1))


@app.route('/postCar_getGrid', methods=["POST"])
def post_javascript_getGrid_data():
    response = request.get_data()
    response = response.decode("utf-8").replace("'", '"')
    json_response = json.loads(response)
    car = get_car(int(json_response['id']))
    mode = json_response['mode']
    reach_grids = reachable_grids(json_response, get_grids())
    #print(reach_grids)
    #visualize_alpha(reach_grids)
    optimal_grid = optimize(reach_grids, mode)
    #print("optimal grid is " , optimal_grid)
    return jsonify(optimal_grid)


@app.route('/postGrids_getOptimal', methods=["POST"])
def postGrids_getOptimal():
    response = request.get_data()
    response = response.decode("utf-8").replace("'", '"')
    reach_grids = json.loads(response)
    #visualize_alpha(reach_grids)
    optimal_grid = optimize(reach_grids, get_car(1)['mode'])
    print ("reach_grids",reach_grids)
    print ("mode", get_car(1)['mode'])
    print ("optimal_grid", optimal_grid)
    return jsonify(optimal_grid)


@app.route('/load_grids', methods=["POST"])
def post_grids():
    response = request.get_data()
    response = response.decode("utf-8").replace("'", '"')
    json_response = json.loads(response)
    grids = reachable_grids(json_response, get_grids())
    #print("car", json_response)
    #print ("grids",grids)
    #print ("car" ,json_response )
    #print ("reachable grids " ,grids)
    return jsonify(get_grids())


if __name__ == "__main__":

    app.run(debug=True)
