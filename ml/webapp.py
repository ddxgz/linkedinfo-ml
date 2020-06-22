"""Web service for serving prediction requests based on trained models"""

import os
import json
import uuid
from datetime import datetime

# from flask import Flask, request
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.wsgi import WSGIMiddleware
from pydantic import BaseModel
import requests
from typing import List, Tuple, Optional

# from dataset import LAN_ENCODING
from .models import predictor
from .dataapp import data_app, MOUNT_PATH
from ml.models import files


# files.init_model_files()

# app.secret_key = str(uuid.uuid4())
# wsgiapp = app.wsgi_app
app = FastAPI()
# app.debug = True
app.mount(MOUNT_PATH, WSGIMiddleware(data_app.server))


TAG_PRED = predictor.get_tag_predictor(
    init=False,
    # test_model=True
)

LAN_PRED = predictor.get_lan_predictor(
    init=True,
)

KEYWORD_PRED = predictor.get_keyword_predictor()


def info2text(info: dict) -> str:
    fulltext = info.get('fulltext', None)
    if fulltext:
        text = fulltext
    else:
        text = info.get('description', None)

    title = info.get('title', None)
    if not text and not title:
        raise KeyError(
            'fulltext, description or title is missing in post data')
        return

    if title:
        text = f"{title}. {text}"
    return text


def predict_keywords(info: dict, num_keywords: int = 5) -> List[str]:
    """returns the a list of keywords for the info"""
    text = info2text(info)
    return KEYWORD_PRED.predict(text, num_keywords=num_keywords)


def predict_language(info: dict) -> str:
    """ An info comes in as a json (dict) in the following format, use title and
    description for prediction.
    {
            "title": "Achieving 100k connections per second with Elixir",
            "url": ,
            "description": "",
            "poster": "",
            "tags": [ ],
            "creators": [ ],
            "language": "en"
    }

    Returns
    -------
    str of the language acronym, en or cn
    """
    if not LAN_PRED.initialized:
        LAN_PRED.init()

    fulltext = info.get('fulltext', None)
    if fulltext:
        text = fulltext
    else:
        text = info.get('description', None)

    title = info.get('title', None)
    if not text and not title:
        raise KeyError(
            'fulltext, description or title is missing in post data')
        return

    if title:
        text = f"{title}. {text}"

    # text = f"{info['title']}. {info['description']}"
    lan = LAN_PRED.predict(text)
    # predicted = LAN_MODEL.predict([text])[0]

    # for lan, enc in LAN_ENCODING.items():
    #     if enc == predicted:
    #         return lan

    return lan


async def predict_lan_by_url(info: dict, entity_tags: bool = True) -> str:
    """ An info comes in as a json (dict), use the url sent in to extract text
     for prediction.

    Returns
    -------
    str of the language 
    """
    from .dataset import extractor

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

    return predict_language(info)


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

    fulltext = info.get('fulltext', None)
    if fulltext:
        text = fulltext
    else:
        text = info.get('description', None)

    title = info.get('title', None)
    if not text and not title:
        raise KeyError(
            'fulltext, description or title is missing in post data')
        return

    if title:
        text = f"{title}. {text}"

    # predicted = TAGS_MODEL.predict([text])
    predicted = TAG_PRED.predict(text, entity_tags=entity_tags)
    # inverse transform tags
    return predicted


async def predict_tags_by_url(info: dict, entity_tags: bool = True) -> Tuple[List[str], dict]:
    """ An info comes in as a json (dict), use the url sent in to extract text
     for prediction.

    Returns
    -------
    List of str of the tagID
    """
    from .dataset import extractor

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

    return await predict_tags(info, entity_tags=entity_tags), info


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
async def pred_lan(info: Info, by_url: bool = False):
    """ Accept POST request with data in application/json. The data body should
    contain either `description`, `fulltext` or `url`. When intend to predict by
    `url`, the requesting url should include parameter `by_url=[True, true, 1,
    on, yes]`.
    """
    # info = request.get_json()
    try:
        if by_url:
            lan_pred = await predict_lan_by_url(info.dict())
        else:
            lan_pred = predict_language(info.dict())
    except KeyError as e:
        raise HTTPException(status_code=400,
                            detail=f"Data key missing: {e}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f'Value error: {e}')
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
    valid_req, msg = check_valid_request(info.dict(), by_url, only_model)
    if not valid_req:
        raise HTTPException(status_code=400, detail=f'Value error: {msg}')

    if only_model:
        entity_tags = False
    else:
        entity_tags = True

    try:
        if by_url:
            tags_pred, _ = await predict_tags_by_url(info.dict(), entity_tags=entity_tags)
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


class InfoExtracted(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    fulltext: Optional[str] = None
    url: Optional[str] = None
    creators: Optional[List[str]] = []
    tags: List[str]
    keywords: List[str]
    language: str


@app.post('/detection', response_model=InfoExtracted)
async def info_detection(info: Info, by_url: bool = False, only_model: bool = False,
                         num_keywords: int = 5):
    """ Accept POST request with data in application/json. The data body should
    contain either `description`, `fulltext` or `url`. When intend to predict by
    `url`, the requesting url should include parameter `by_url=[True, true, 1,
    on, yes]`.
    """
    valid_req, msg = check_valid_request(info.dict(), by_url, only_model)
    if not valid_req:
        raise HTTPException(status_code=400, detail=f'Value error: {msg}')

    if only_model:
        entity_tags = False
    else:
        entity_tags = True

    try:
        if by_url:
            tags_pred, info_ext = await predict_tags_by_url(info.dict(), entity_tags=entity_tags)
        else:
            tags_pred = await predict_tags(info.dict(), entity_tags=entity_tags)
            info_ext = info.dict()

        lan = predict_language(info_ext)

        keywords = predict_keywords(info_ext, num_keywords=num_keywords)
    except KeyError as e:
        raise HTTPException(status_code=400,
                            detail=f"Data key missing: {e}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f'Value error: {e}')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Unknown error: {e}')

    # if not only_model:
    #     tags_pred = append_map_tags(tags_pred, info.dict())

    resp = InfoExtracted(tags=tags_pred, language=lan, keywords=keywords)
    # resp.tags = tags_pred
    # resp.language = lan
    if info_ext.get('fulltext'):
        resp.fulltext = info_ext.get('fulltext')
    if info_ext.get('description'):
        resp.description = info_ext.get('description')
    if info_ext.get('url'):
        resp.url = info_ext.get('url')
    if info_ext.get('title'):
        resp.title = info_ext.get('title')
    if info_ext.get('creators'):
        resp.creators = info_ext.get('creators')

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


if __name__ == '__main__':
    import uvicorn

    uvicorn.run('ml.webapp:app', host="127.0.0.1", port=5000,
                reload=True, log_level="debug")
