"""Web service for serving prediction requests based on trained models"""

# from abc import ABC, abstractmethod
import os
import json
import uuid
from datetime import datetime

# import numpy as np
# import pandas as pd
# from flask import Flask, request
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.wsgi import WSGIMiddleware
from pydantic import BaseModel
import requests
# import joblib
# import torch
# from transformers import AutoTokenizer, AutoModel
# import nltk
# from nltk.tokenize.treebank import TreebankWordTokenizer, TreebankWordDetokenizer
from typing import List, Tuple, Optional

# from mltb.mltb.bert import download_once_pretrained_transformers
# from dataset import LAN_ENCODING
import predictor
from dataapp import data_app, MOUNT_PATH


# app = Flask('ML-prediction-service')
# app.secret_key = str(uuid.uuid4())
# app.debug = False
# wsgiapp = app.wsgi_app
app = FastAPI()
app.mount(MOUNT_PATH, WSGIMiddleware(data_app.server))


TAG_PRED = predictor.get_tag_predictor(
    # test_model=True
)


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


async def predict_tags(info: dict, entity_tags: bool = True) -> List[str]:
    """ An info comes in as a json (dict), use
    description or fulltext (if presence) for prediction.

    Returns
    -------
    List of str of the tagsID
    """
    # global TAGS_MODEL

    # if not TAGS_MODEL:
    #     #     TAGS_MODEL = TagsTextModelV3(modelfile=MODEL_FILE)
    #     # lazy loading models and data
    #     await lazy_load()
    if not TAG_PRED.initialized:
        TAG_PRED.init()

    # print('after load')

    if info.get('fulltext'):
        text = info['fulltext']
    else:
        text = info['description']

    if info.get('title'):
        text = f"{info['title']}. {text}"

    # predicted = TAGS_MODEL.predict([text])
    predicted = TAG_PRED.predict(text, entity_tags=entity_tags)
    # inverse transform tags
    return predicted


async def predict_tags_by_url(info: dict, entity_tags: bool = True) -> List[str]:
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

    return await predict_tags(info, entity_tags=entity_tags)


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

    if only_model:
        entity_tags = False
    else:
        entity_tags = True

    try:
        if by_url:
            tags_pred = await predict_tags_by_url(info.dict(), entity_tags=entity_tags)
        else:
            tags_pred = await predict_tags(info.dict(), entity_tags=entity_tags)
    except KeyError as e:
        raise HTTPException(status_code=400,
                            detail=f"Data key missing: {e}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f'Value error: {e}')

    # if not only_model:
    #     tags_pred = append_map_tags(tags_pred, info.dict())

    resp = PredTags()
    resp.tags = tags_pred

    # print(TAGS_MODEL.__class__.__name__)
    return resp


class ModelInit(BaseModel):
    model_name: str = 'all'


@app.put('/predictions/init')
async def init_model(data: ModelInit):
    """Initializa the prediction model."""
    if not TAG_PRED.initialized:
        TAG_PRED.init()
    return 'Model initialized.'


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

# TAGS_MODEL = None
# # TAGS_MODEL = TagsTestModel()

# TAGS_MAP = {}
# TAGS_LIST = []

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
