from flask import Flask, render_template
from database import get_cars, get_grids
app = Flask(__name__)


@app.route('/', methods=["GET", "POST"])
def index():
    cars = get_cars()
    grids = get_grids()
    return render_template('index.html', cars=cars, grids=grids)


if __name__ == "__main__":

    app.run(debug=True )
