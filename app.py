import subprocess
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
import atexit

from flask import Flask, jsonify, request, Response
from flask_cors import CORS, cross_origin
import pandas as pd
import numpy as np

import covid19_model as model


def run_training():
    print("Training started")
    model.fetch_data()
    model.detect_growth()
    model.calculate_forecast()


scheduler = BackgroundScheduler()
scheduler.start()

scheduler.add_job(
    func=run_training,
    trigger='cron',
    hour='2',
    minute='00')

atexit.register(lambda: scheduler.shutdown())

app = Flask(__name__)
CORS(app)


@app.route("/forecast", methods=['GET'])
def forecast():
    try:
        country = request.args.get('country')
        df = pd.read_csv('data/covid19_forecast_data_' + country + '_cases.csv', parse_dates=True)
        df = df.drop(df.columns[[0]], axis=1)

        result = df.to_json(orient='records', date_format='iso')
        return result
    except FileNotFoundError:
        return Response(status=404)


if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5001)