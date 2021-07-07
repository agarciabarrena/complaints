from flask import Flask
from requests import request
from utils.app_utils import forecast_batch
import pandas as pd

application = Flask(__name__)

@application.route('/')
def main_page():
    return('Yes, The server is up!')

@application.route('/predict', methods=['Post'])
def prediction_server_API_call():
    try:
        data_json = request.get_json()
        d = pd.read_json(data_json, orient='records')
    except Exception as e:
        raise e

    if d.empty:
        return('FAILED')
    else:
        df = forecast_batch(data=d)
        return_data = df.to_json(orient='records')
        return return_data

if __name__ == "__main__":
    application.run()
