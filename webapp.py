"""Web service for serving prediction requests based on trained models"""

import os
import json
import uuid

from flask import Flask, request
from gevent.pywsgi import WSGIServer as Geventwsgiserver
import joblib

from dataset import LAN_ENCODING

app = Flask('ML-prediction-service')
app.secret_key = str(uuid.uuid4())
app.debug = True
wsgiapp = app.wsgi_app


def singleton(cls, *args, **kwargs):
    instances = {}

    def _singleton():
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return _singleton


@singleton
class LanModel:
    def __init__(self, modelfile: str = 'data/models/lan_pred_1.joblib.gz'):
        self._load_model(modelfile)

    def predict(self, text):
        return self.model.predict(text)

    def _load_model(self, modelfile: str = None):
        if os.path.exists(modelfile):
            self.model = joblib.load(modelfile)
        else:
            raise FileNotFoundError('Model file not exists! The model should be'
                                    'place under ./data/models/')


def predict_language(info: dict) -> str:
    """ An info comes in as a json (dict) in the following format, use title and 
    description for prediction.
    {
            "key": "",
            "postAt": "",
            "modifiedAt": "",
            "title": "Achieving 100k connections per second with Elixir",
            "url": ,
            "description": "",
            "poster": "",
            "inAggregations": [ ],
            "tags": [ ],
            "creators": [ ],
            "language": "en"
    }

    Returns
    -------
    str of the language acronym, en or cn
    """
    text = f"{info['title']}. {info['description']}"
    predicted = LAN_MODEL.predict([text])[0]

    for lan, enc in LAN_ENCODING.items():
        if enc == predicted:
            return lan

    return 'unknown_lan'


@app.route('/predictions/language', methods=['POST'])
def pred_lan():
    if request.method == 'POST':
        info = request.get_json()
        lan_pred = predict_language(info)
        resp = json.dumps({'language': lan_pred})
        return resp


LAN_MODEL = LanModel()

if __name__ == '__main__':
    # use gevent wsgi server
    httpserver = Geventwsgiserver(('0.0.0.0', 8090), wsgiapp)
    httpserver.serve_forever()

    # with open('data/cache/infos_80_90.json', 'r') as f:
    #     infos = json.load(f)
    #     for info in infos['content']:
    #         print(predict_language(lan_model, info))
