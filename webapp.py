"""Web service for serving prediction requests based on trained models"""

from abc import ABC, abstractmethod
import os
import json
import uuid
from datetime import datetime

# import numpy as np
import pandas as pd
# from flask import Flask, request
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.wsgi import WSGIMiddleware
from pydantic import BaseModel
import requests
import joblib
# import torch
# from transformers import AutoTokenizer, AutoModel
# import nltk
# from nltk.tokenize.treebank import TreebankWordTokenizer, TreebankWordDetokenizer
from typing import List, Tuple, Optional

# from mltb.mltb.bert import download_once_pretrained_transformers
# from dataset import LAN_ENCODING
from dataapp import data_app, MOUNT_PATH


# app = Flask('ML-prediction-service')
# app.secret_key = str(uuid.uuid4())
# app.debug = False
# wsgiapp = app.wsgi_app
app = FastAPI()
app.mount(MOUNT_PATH, WSGIMiddleware(data_app.server))


# PRETRAINED_BERT_WEIGHTS = "./data/models/google/"
# PRETRAINED_BERT_WEIGHTS = "./data/models/google/bert_uncased_L-2_H-128_A-2/"
# PRETRAINED_BERT_WEIGHTS = "google/bert_uncased_L-2_H-128_A-2"
# PRETRAINED_BERT_WEIGHTS = download_once_pretrained_transformers(
#     "google/bert_uncased_L-4_H-256_A-4")
PRETRAINED_BERT_WEIGHTS = "./data/models/bert_mini_finetuned_tagthr_20/"

MODEL_FILE = 'data/models/tags_textbased_pred_8.joblib.gz'
MLB_FILE = 'data/models/tags_textbased_pred_8_mlb.joblib.gz'


async def lazy_load():
    # print('start to load model and data')
    global TAGS_MODEL, TAGS_LIST, TAGS_MAP

    # yield
    if not TAGS_MODEL:
        TAGS_MODEL = TagsTextModelV3(modelfile=MODEL_FILE)
    if not TAGS_LIST:
        TAGS_LIST = get_tags_list()
    if not TAGS_MAP:
        TAGS_MAP = get_tags_map()


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
            'users must define `predict` to use this base class')

    def _load_model(self, modelfile: str):
        if os.path.exists(modelfile):
            model = joblib.load(modelfile)
            return model
        else:
            cwd = os.getcwd()
            raise FileNotFoundError(f'Model file not exists! The model should be'
                                    'place under ./data/models/. CWD: {cwd}')


@singleton
class TagsTestModel(PredictModel):
    def __init__(self):
        pass

    def predict(self, text):
        # return [[]]
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
            list_len.append(len(self.tokenizer.tokenize(i)))
            # list_len.append(len(nltk.word_tokenize(i)))

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


def get_tags_map() -> dict:
    tags = {"ML": "machine-learning"}
    headers = {'Accept': 'application/json'}
    resp = requests.get(
        'https://www.linkedinfo.co/tag-map', headers=headers)

    if resp.status_code == 200:
        try:
            tags = resp.json()
            return tags
        except ValueError as e:
            return tags
    return tags


def get_tags_list() -> List[dict]:
    tags = [{'tagID': 'python', 'label': 'Python'}]
    headers = {'Accept': 'application/json'}
    resp = requests.get(
        'https://www.linkedinfo.co/tags', headers=headers)

    if resp.status_code == 200:
        try:
            tags = resp.json()
            return tags
        except ValueError as e:
            return tags
    return tags


def append_map_tags(tags: List[str], info: dict) -> List[str]:
    # global TAGS_LIST, TAGS_MAP
    # if not TAGS_LIST:
    #     TAGS_LIST = get_tags_list()
    # if not TAGS_MAP:
    #     TAGS_MAP = get_tags_map()

    map_tags: List[str] = []
    # print(info)
    if info.get('fulltext'):
        text = info['fulltext']
    else:
        text = info['description']

    if info.get('title'):
        text = f"{info['title']}. {text}"

    # print(text)
    # toks = nltk.word_tokenize(text)
    toks = text.split(' ,.!?')
    if len(toks) > 20:
        toks = toks[:20]

    # print(toks)

    # d = TreebankWordDetokenizer()
    # text = d.detokenize(toks).lower()

    # print(text)
    for tag in TAGS_LIST:
        if tag['label'] in toks:
            map_tags.append(tag['tagID'])
        if len(tag['label']) > 9:
            if tag['label'] in text:
                map_tags.append(tag['tagID'])

    for k, v in TAGS_MAP.items():
        if k in text:
            map_tags.append(v)

    # print(tags)
    # print(map_tags)
    if map_tags:
        tags = list(tags) + map_tags

    return list(set(tags))


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


async def predict_tags(info: dict) -> List[str]:
    """ An info comes in as a json (dict), use
    description or fulltext (if presence) for prediction.

    Returns
    -------
    List of str of the tagsID
    """
    # global TAGS_MODEL

    if not TAGS_MODEL:
        #     TAGS_MODEL = TagsTextModelV3(modelfile=MODEL_FILE)
        # lazy loading models and data
        await lazy_load()

    # print('after load')

    if info.get('fulltext'):
        text = info['fulltext']
    else:
        text = info['description']

    if info.get('title'):
        text = f"{info['title']}. {text}"

    predicted = TAGS_MODEL.predict([text])
    # inverse transform tags
    return predicted[0]


async def predict_tags_by_url(info: dict) -> List[str]:
    """ An info comes in as a json (dict), use the url sent in to extract text
     for prediction.

    Returns
    -------
    List of str of the tagID
    """
    import extractor

    # if 'url' in info.keys():
    infourl = info['url']

    if infourl is None:
        raise KeyError('url missing in post data')
    # print(infourl)
    # else:
    #     return []

    try:
        info = extractor.extract_info_from_url(infourl)
    except TypeError:
        raise ValueError("URL is wrong or not fetchable")

    return await predict_tags(info)


def check_valid_request(info: dict, by_url: bool = False, only_model: bool = False) -> Tuple[bool, str]:
    if by_url:
        infourl = info['url']
        if infourl is None:
            return False, 'URL missing in post data.'
    else:
        toks = info['description'].split(' ')
        # toks = nltk.word_tokenize(info['description'])
        if len(toks) < 5:
            return False, 'Too short text for prediction.'
    return True, ''


class Info(BaseModel):
    url: Optional[str]
    title: Optional[str]
    description: Optional[str]
    fulltext: Optional[str]


@app.post('/predictions/language')
async def pred_lan(info: Info):
    """Not implemented yet"""
    # info = request.get_json()
    lan_pred = predict_language(info.dict())
    # resp = json.dumps({'language': lan_pred})
    return {'language': lan_pred}


class TagSuggestions(BaseModel):
    url: Optional[str]
    description: Optional[str]
    tags: str
    tags_suggest: str


@app.post('/tag-suggestions')
async def tag_suggestions(info: TagSuggestions):
    post_time = datetime.now().isoformat()
    data = {
        'postAt': post_time,
        'title': '',
        'url': info.url if info.url is not None else '',
        'description': info.description if info.description is not None else '',
        'poster': 'info.poster',
        'tags': info.tags,
        'tags_suggest': info.tags_suggest,
    }
    # print(data)
    # headers = {'Content-type': 'application/json'}
    resp = requests.post(
        'https://www.linkedinfo.co/tag-suggestions', json=data)
    # print(resp.status_code)
    # print(resp.text)

    return f'You submitted {info.tags_suggest}'


class PredTags(BaseModel):
    tags: List[str] = []


@app.post('/predictions/tags', response_model=PredTags)
async def pred_tags(info: Info, by_url: bool = False, only_model: bool = False):
    """ Accept POST request with data in application/json. The data body should
    contain either `description`, `fulltext` or `url`. When intend to predict by
    `url`, the requesting url should include parameter `by_url=[True, true, 1,
    on, yes]`.
    """
    # if request.method == 'POST':
    #     info = request.get_json()

    # print('-------------- before --------------')
    # print(TAGS_MODEL.__class__.__name__)
    # if multiple url args with the same key, only the 1st will be returned
    # by_url = request.args.get('by_url', None)
    valid_req, msg = check_valid_request(info.dict(), by_url, only_model)
    if not valid_req:
        raise HTTPException(status_code=400, detail=f'Value error: {msg}')

    try:
        if by_url:
            tags_pred = await predict_tags_by_url(info.dict())
        else:
            tags_pred = await predict_tags(info.dict())
    except KeyError as e:
        raise HTTPException(status_code=400,
                            detail=f"Data key missing: {e}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f'Value error: {e}')

    if not only_model:
        tags_pred = append_map_tags(tags_pred, info.dict())

    resp = PredTags()
    resp.tags = tags_pred

    # print(TAGS_MODEL.__class__.__name__)
    return resp


@app.get('/', responses={200: {
    "content": {"text/html": {}},
    "description": "Return the home page of the app."}})
async def home():
    return FileResponse('vuejs/home-bootstrap-vue.html', media_type='text/html')


# LAN_MODEL = LanModel(modelfile='data/models/lan_pred_1.joblib.gz')
# TAGS_MODEL = TagsTextModel(
#     modelfile='data/models/tags_textbased_pred_5.joblib.gz',
#     mlb_fiile='data/models/tags_textbased_pred_5_mlb.joblib.gz')
# TAGS_MODEL = TagsTextModelV3(
#     modelfile=MODEL_FILE)
# TAGS_MAP = get_tags_map()
# TAGS_LIST = get_tags_list()

TAGS_MODEL = None
# TAGS_MODEL = TagsTestModel()

TAGS_MAP = {}
TAGS_LIST = []

if __name__ == '__main__':
    # use gevent wsgi server
    # httpserver = Geventwsgiserver(('0.0.0.0', 5000), wsgiapp)
    # httpserver.serve_forever()

    import uvicorn

    uvicorn.run('webapp:app', host="127.0.0.1", port=5000,
                reload=True, log_level="debug")

    # with open('data/cache/infos_80_90.json', 'r') as f:
    #     infos = json.load(f)
    #     for info in infos['content']:
    #         print(predict_language(lan_model, info))
