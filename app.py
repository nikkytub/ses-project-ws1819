from flask import Flask, render_template, request, jsonify
from database import get_cars, get_grids, get_car

from optimizer import get_optimal
import json
from ac_grid_model import reachable_grids
app = Flask(__name__)


@app.route('/', methods=["GET", "POST"])
def index():
    cars = get_cars()
    grids = get_grids()
    return render_template('index.html', cars=cars, grids=grids)


@app.route('/simulation')
def simulation():
    s = str(request).split('?')
    car_id = s[1].split('\'')[0].split('=')[1]
    car = get_car(car_id)
    grids = reachable_grids(car,get_grids())
    return render_template('simulation.html', car=car, grids=grids)


@app.route('/postCar', methods=["POST"])
def post_javascript_data():
    response = request.get_data()
    response = response.decode("utf-8").replace("'", '"')
    json_response = json.loads(response)
    car = get_car(int(json_response['id']))
    mode = json_response['mode']
    optimal_grid = get_optimal(car, mode)
    return jsonify(optimal_grid)


@app.route('/postCar_getGrid', methods=["POST"])
def post_javascript_getGrid_data():
    response = request.get_data()
    response = response.decode("utf-8").replace("'", '"')
    json_response = json.loads(response)
    car = get_car(int(json_response['id']))
    mode = json_response['mode']
    optimal_grid = get_optimal(car, mode)



if __name__ == "__main__":

    app.run(debug=True)
