"""Web service for serving prediction requests based on trained models"""

from abc import ABC, abstractmethod
import os
import json
import uuid

import numpy as np
import pandas as pd
from flask import Flask, request, send_from_directory
from gevent.pywsgi import WSGIServer as Geventwsgiserver
import joblib
import torch
from transformers import AutoTokenizer, AutoModel
import nltk

from mltb.bert import download_once_pretrained_transformers
import extractor
from dataset import LAN_ENCODING


app = Flask('ML-prediction-service')
app.secret_key = str(uuid.uuid4())
app.debug = False
wsgiapp = app.wsgi_app

nltk.download('punkt')

# PRETRAINED_BERT_WEIGHTS = "./data/models/google/"
# PRETRAINED_BERT_WEIGHTS = "./data/models/google/bert_uncased_L-2_H-128_A-2/"
# PRETRAINED_BERT_WEIGHTS = "google/bert_uncased_L-2_H-128_A-2"
# PRETRAINED_BERT_WEIGHTS = download_once_pretrained_transformers(
#     "google/bert_uncased_L-4_H-256_A-4")
PRETRAINED_BERT_WEIGHTS = "./data/models/bert_mini_finetuned_tagthr_20/"

MODEL_FILE = 'data/models/tags_textbased_pred_6.joblib.gz'
MLB_FILE = 'data/models/tags_textbased_pred_6_mlb.joblib.gz'


def singleton(cls, *args, **kwargs):
    instances = {}

    def _singleton(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return _singleton


class PredictModel(ABC):
    def __init__(self, modelfile: str):
        self.model = self._load_model(modelfile)
        super().__init__()

    @abstractmethod
    def predict(self, text):
        raise NotImplementedError(
            'users must define __str__ to use this base class')

    def _load_model(self, modelfile: str = None):
        if os.path.exists(modelfile):
            model = joblib.load(modelfile)
            return model
        else:
            raise FileNotFoundError('Model file not exists! The model should be'
                                    'place under ./data/models/')


@singleton
class TagsTestModel(PredictModel):
    def __init__(self):
        pass

    def predict(self, text):
        return [['test-tags', 'machine-learning', 'python']]


@singleton
class LanModel(PredictModel):

    def predict(self, text):
        return self.model.predict(text)


@singleton
class TagsTextModel(PredictModel):
    def __init__(self, modelfile: str, mlb_fiile: str):
        super().__init__(modelfile)

        if os.path.exists(mlb_fiile):
            mlb = joblib.load(mlb_fiile)
        else:
            raise FileNotFoundError('MLB Model file not exists! The model should be'
                                    'place under ./data/models/')
        self.mlb = mlb

    def predict(self, text):
        # return self.model.predict(text)
        pred = self.model.predict(text)
        pred_transformed = self.mlb.inverse_transform(pred)
        return pred_transformed


@singleton
class TagsTextModelV2(PredictModel):
    def __init__(self, modelfile: str):
        super().__init__(modelfile)

        self.tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_BERT_WEIGHTS)
        self.feat_model = AutoModel.from_pretrained(PRETRAINED_BERT_WEIGHTS)

        # ds = dataset.ds_info_tags(from_batch_cache='fulltext', content_length_threshold=100, lan='en',
        #                           filter_tags_threshold=2, partial_len=3000, total_size=None)

        if os.path.exists(MLB_FILE):
            mlb = joblib.load(MLB_FILE)
        else:
            raise FileNotFoundError('MLB Model file not exists! The model should be'
                                    'place under ./data/models/')
        self.mlb = mlb

    def predict(self, text):
        list_len = []
        for i in text:
            list_len.append(len(nltk.word_tokenize(i)))

        max_length = max(list_len)
        if max_length > 512:
            max_length = 512

        input_ids = []
        attention_masks = []
        for i in text:
            encoded = self.tokenizer.encode_plus(i, add_special_tokens=True,
                                                 pad_to_max_length=True,
                                                 return_attention_mask=True,
                                                 max_length=max_length,
                                                 return_tensors='pt')

            input_ids.append(encoded['input_ids'])
            attention_masks.append(encoded['attention_mask'])

        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        # input_ids = torch.tensor(padded)
        # attention_mask = torch.tensor(attention_mask)
        features = []

        with torch.no_grad():
            last_hidden_states = self.feat_model(
                input_ids, attention_mask=attention_masks)
            features = last_hidden_states[0][:, 0, :].numpy()

        pred = self.model.predict(features)
        pred_transformed = self.mlb.inverse_transform(pred)
        return pred_transformed


@singleton
class TagsTextModelV3(PredictModel):
    def __init__(self, modelfile: str):
        super().__init__(modelfile)

        if os.path.exists(MLB_FILE):
            mlb = joblib.load(MLB_FILE)
        else:
            raise FileNotFoundError('MLB Model file not exists! The model should be'
                                    'place under ./data/models/')
        self.mlb = mlb

    def predict(self, text):
        col_text = 'description'
        features = pd.DataFrame(text, columns=[col_text])

        pred = self.model.predict(features)
        pred_transformed = self.mlb.inverse_transform(pred)
        return pred_transformed


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
    # predicted = LAN_MODEL.predict([text])[0]

    # for lan, enc in LAN_ENCODING.items():
    #     if enc == predicted:
    #         return lan

    return 'unknown_lan'


def predict_tags(info: dict) -> str:
    """ An info comes in as a json (dict), use 
    description or fulltext (if presence) for prediction.

    Returns
    -------
    str of the tags acronym
    """
    if 'fulltext' in info.keys():
        text = info['fulltext']
    else:
        text = info['description']

    if 'title' in info.keys():
        text = f"{info['title']}. {text}"

    predicted = TAGS_MODEL.predict([text])
    # inverse transform tags
    return predicted[0]


def predict_tags_by_url(info: dict) -> str:
    """ An info comes in as a json (dict), use the url sent in to extract text
     for prediction.

    Returns
    -------
    List of str of the tagID
    """
    if 'url' in info.keys():
        infourl = info['url']
    else:
        return None

    info = extractor.extract_info_from_url(infourl)

    return predict_tags(info)


@app.route('/predictions/language', methods=['POST'])
def pred_lan():
    if request.method == 'POST':
        info = request.get_json()
        lan_pred = predict_language(info)
        resp = json.dumps({'language': lan_pred})
        return resp


@app.route('/predictions/tags', methods=['POST'])
def pred_tags():
    if request.method == 'POST':
        info = request.get_json()

        # if multiple url args with the same key, only the 1st will be returned
        by_url = request.args.get('by_url', None)
        if by_url is not None and by_url in [True, 'true', 'True', 1, '1']:
            tags_pred = predict_tags_by_url(info)
        else:
            tags_pred = predict_tags(info)
        resp = json.dumps({'tags': tags_pred})
        return resp


@app.route('/', methods=['GET'])
def home():
    return send_from_directory('vuejs', 'home.html')


# LAN_MODEL = LanModel(modelfile='data/models/lan_pred_1.joblib.gz')
# TAGS_MODEL = TagsTextModel(
#     modelfile='data/models/tags_textbased_pred_5.joblib.gz',
#     mlb_fiile='data/models/tags_textbased_pred_5_mlb.joblib.gz')
TAGS_MODEL = TagsTextModelV3(
    modelfile=MODEL_FILE)
# TAGS_MODEL = TagsTestModel()

if __name__ == '__main__':
    # use gevent wsgi server
    httpserver = Geventwsgiserver(('0.0.0.0', 5000), wsgiapp)
    httpserver.serve_forever()

    # with open('data/cache/infos_80_90.json', 'r') as f:
    #     infos = json.load(f)
    #     for info in infos['content']:
    #         print(predict_language(lan_model, info))
