# %%
"""
Fetch infos of articles from LinkedInfo.co or other places.
"""
import os
import time
from datetime import datetime
import json
import logging
from dataclasses import dataclass
import random
from collections import Counter
import re


import requests
import html2text
from urllib.parse import urlparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize.treebank import TreebankWordTokenizer, TreebankWordDetokenizer
import pysnooper


nltk.download('punkt')

logger = logging.getLogger('info')
# logger.setLevel(logging.In)
handler = logging.FileHandler(filename='info.log')
handler.setLevel(logging.INFO)
logger.addHandler(handler)
consoleHandler = logging.StreamHandler()
consoleHandler.setLevel(logging.DEBUG)
logger.addHandler(consoleHandler)

# logging.basicConfig(level=logging.INFO)

RAND_STATE = 20200122
DATA_DIR = 'data'
INFOS_CACHE = 'infos_0_3790.json'
INFOS_FULLTEXT_CACHE = 'infos_0_3790_fulltext.json'
# UNTAGGED_INFOS_FULLTEXT_CACHE = 'untagged_infos_fulltext.json'
UNTAGGED_INFOS_CACHE = 'untagged_infos.json'


def caching_untagged_infos(data_home='data'):
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    infos_home = os.path.join(data_home, 'untagged_infos')

    if not os.path.exists(infos_home):
        os.makedirs(infos_home)

    cache = fetch_untagged_infos(fulltext=True)

    filename = f'{infos_home}/untagged_infos_{timestamp}.json'
    with open(filename, 'w') as f:
        json.dump(cache, f)


def fetch_untagged_infos(data_home='data', fulltext=False,
                         force_download=True):
    data_home = data_home
    cache_path = os.path.join(data_home, 'cache')
    infos_home = os.path.join(data_home, 'untagged_infos')
    infos_cache = os.path.join(infos_home, UNTAGGED_INFOS_CACHE)
    cache = None

    if os.path.exists(infos_cache) and not force_download:
        with open(infos_cache, 'r') as f:
            cache = json.load(f)

    if cache is None:
        logger.info("Calling API to retrieve infos.")
        cache = _retrieve_untagged_infos(target_dir=infos_home,
                                         cache_path=cache_path)

    if fulltext:
        cache_path_fulltext = os.path.join(cache_path, 'fulltext')
        target_path_fulltext = os.path.join(data_home, 'fulltext')
        if not os.path.exists(cache_path_fulltext):
            os.makedirs(cache_path_fulltext)
        if not os.path.exists(target_path_fulltext):
            os.makedirs(target_path_fulltext)
        for info in cache['content']:
            info['fulltext'] = _retrieve_info_fulltext(info,
                                                       target_dir=target_path_fulltext,
                                                       cache_path=cache_path_fulltext)

    return cache


def fetch_infos(data_home='data', from_batch_cache: str = None, fulltext=True,
                save_cache: bool = True, force_download=False,
                force_extract=True,
                random_state=42, remove=(), download_if_missing=True,
                total_size=None, allow_infos_cache: bool = True,
                *args, **kwargs):
    """Load the infos from linkedinfo.co or local cache.
    Parameters
    ----------
    data_home : optional, default: 'data'
        Specify a download and cache folder for the datasets. If None,
        all scikit-learn data is stored in './data' subfolders.

    from_batch_cache: 'fulltext','info', None, optional
        Read from aggregated all infos batch cache file. Download or reload from
        small cache files.

    fulltext : optional, False by default
        If True, it will fectch the full text of each info.

    random_state : int, RandomState instance or None (default)
        Determines random number generation for dataset shuffling. Pass an int
        for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    remove : tuple
        May contain any subset of ('headers', 'footers', 'quotes'). Each of
        these are kinds of text that will be detected and removed from the
        newsgroup posts, preventing classifiers from overfitting on
        metadata.

        'headers' removes newsgroup headers, 'footers' removes blocks at the
        ends of posts that look like signatures, and 'quotes' removes lines
        that appear to be quoting another post.

        'headers' follows an exact standard; the other filters are not always
        correct.

    download_if_missing : optional, True by default
        If False, raise an IOError if the data is not locally available
        instead of trying to download the data from the source site.

    Returns
    -------
    Dict: all infos w/ or w/o fulltext
    """
    data_home = data_home
    cache_path = os.path.join(data_home, 'cache')
    infos_home = os.path.join(data_home, 'infos')
    cache = None

    if from_batch_cache == 'fulltext':
        infos_cache = os.path.join(infos_home, INFOS_FULLTEXT_CACHE)
        if os.path.exists(infos_cache):
            with open(infos_cache, 'r') as f:
                cache = json.load(f)
                return cache
        else:
            return cache

    if from_batch_cache == 'info':
        infos_cache = os.path.join(infos_home, INFOS_CACHE)
        if os.path.exists(infos_cache):
            with open(infos_cache, 'r') as f:
                cache = json.load(f)
                return cache
        else:
            return cache

    logger.info("Calling API to retrieve infos.")
    cache = _retrieve_infos(target_dir=infos_home,
                            cache_path=cache_path, fragment_size=10,
                            total_size=total_size
                            )
    if save_cache:
        filename = f'infos_0_{len(cache["content"])}.json'
        infos_cache = os.path.join(infos_home, filename)
        logger.info('Saving info cache without fulltext to file {infos_cache}')
        with open(infos_cache, 'w') as f:
            json.dump(cache, f)

    if fulltext:
        logger.info("Retriving fulltext")
        cache_path_fulltext = os.path.join(cache_path, 'fulltext')
        target_path_fulltext = os.path.join(data_home, 'fulltext')

        if not os.path.exists(cache_path_fulltext):
            os.makedirs(cache_path_fulltext)
        if not os.path.exists(target_path_fulltext):
            os.makedirs(target_path_fulltext)

        for info in cache['content']:
            info['fulltext'] = _retrieve_info_fulltext(info,
                                                       target_dir=target_path_fulltext,
                                                       cache_path=cache_path_fulltext,
                                                       force_download=False,
                                                       force_extract=True)
        if save_cache:
            filename = f'infos_0_{len(cache["content"])}_fulltext.json'
            infos_cache = os.path.join(infos_home, filename)
            logger.info(
                'Saving info cache with fulltext to file {infos_cache}')
            with open(infos_cache, 'w') as f:
                json.dump(cache, f)

    return cache


# TODO: Deprecated
def fetch_infos_dep(data_home='data', subset='train', fulltext=False,
                    random_state=42, remove=(), download_if_missing=True,
                    total_size=None, allow_infos_cache: bool = True,
                    allow_full_cache: bool = True, *args, **kwargs):
    """Load the infos from linkedinfo.co or local cache.
    Parameters
    ----------
    data_home : optional, default: 'data'
        Specify a download and cache folder for the datasets. If None,
        all scikit-learn data is stored in './data' subfolders.

    subset : 'train' or 'test', 'all', optional
        Select the dataset to load: 'train' for the training set, 'test'
        for the test set, 'all' for both, with shuffled ordering.

    fulltext : optional, False by default
        If True, it will fectch the full text of each info.

    random_state : int, RandomState instance or None (default)
        Determines random number generation for dataset shuffling. Pass an int
        for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    remove : tuple
        May contain any subset of ('headers', 'footers', 'quotes'). Each of
        these are kinds of text that will be detected and removed from the
        newsgroup posts, preventing classifiers from overfitting on
        metadata.

        'headers' removes newsgroup headers, 'footers' removes blocks at the
        ends of posts that look like signatures, and 'quotes' removes lines
        that appear to be quoting another post.

        'headers' follows an exact standard; the other filters are not always
        correct.

    download_if_missing : optional, True by default
        If False, raise an IOError if the data is not locally available
        instead of trying to download the data from the source site.

    Returns
    -------
    bunch : Bunch object with the following attribute:
        - bunch.data: list, length [n_samples]
        - bunch.target: array, shape [n_samples]
        - bunch.filenames: list, length [n_samples]
        - bunch.DESCR: a description of the dataset.
        - bunch.target_names: a list of categories of the returned data,
          length [n_classes]. This depends on the `categories` parameter.
    """
    data_home = data_home
    cache_path = os.path.join(data_home, 'cache')
    infos_home = os.path.join(data_home, 'infos')
    infos_cache = os.path.join(infos_home, INFOS_CACHE)
    cache = None

    if allow_infos_cache and os.path.exists(infos_cache):
        with open(infos_cache, 'r') as f:
            cache = json.load(f)

    if cache is None:
        if download_if_missing:
            logger.info("Calling API to retrieve infos.")
            cache = _retrieve_infos(target_dir=infos_home,
                                    cache_path=cache_path, total_size=total_size)
        else:
            raise FileNotFoundError(
                'Infos dataset not found, set download_if_missing to True to '
                'enable data download.')

    if fulltext:
        if allow_full_cache:
            infos_cache = os.path.join(infos_home, INFOS_FULLTEXT_CACHE)
            if os.path.exists(infos_cache):
                with open(infos_cache, 'r') as f:
                    cache = json.load(f)
                    return cache
        cache_path_fulltext = os.path.join(cache_path, 'fulltext')
        target_path_fulltext = os.path.join(data_home, 'fulltext')
        if not os.path.exists(cache_path_fulltext):
            os.makedirs(cache_path_fulltext)
        if not os.path.exists(target_path_fulltext):
            os.makedirs(target_path_fulltext)
        for info in cache['content']:
            info['fulltext'] = _retrieve_info_fulltext(info,
                                                       target_dir=target_path_fulltext,
                                                       cache_path=cache_path_fulltext)
        with open(infos_cache, 'w') as f:
            json.dump(cache, f)

    return cache


def _retrieve_untagged_infos(target_dir, cache_path):
    size = 20
    infos_cache = os.path.join('data/infos', INFOS_CACHE)
    cache = None

    if os.path.exists(infos_cache):
        with open(infos_cache, 'r') as f:
            cache = json.load(f)

    offset = random.randint(0, len(cache['content']) - size)
    return {'content': cache['content'][offset:offset + size]}


# @pysnooper.snoop()
def _retrieve_infos(target_dir, cache_path, fragment_size=10, total_size=None):
    """Call API to retrieve infos data. Retrieve a fragment of infos in multiple
    API calls, caches each fragment in cache_path. Combine all caches into one
    file to target_dir.

    Cache file naming: infos_{offset}_{offset+fragment_size}.json
    Target file naming: infos_{offset}_{size}.json

    Note: fragment_size support only 10 for now due to the restriction of
    linkedinfo.co API
    """
    offset = 0
    ret_size = fragment_size
    cache_files = []

    infos_url = 'https://linkedinfo.co/infos'
    headers = {
        'Accept': 'application/json',
    }

    if not os.path.exists(cache_path):
        os.makedirs(cache_path)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    while(ret_size == fragment_size):
        cache_filename = f'infos_{offset}_{offset+fragment_size}.json'
        cache_pathfile = os.path.join(cache_path, cache_filename)
        # check cache
        if not os.path.exists(cache_pathfile):
            # call api
            params = {'offset': offset, 'quantify': fragment_size}
            res = requests.get(infos_url, headers=headers, params=params)
            if res.status_code != 200:
                raise ConnectionError('Get infos not succeed!')
            # store cache
            infos_new = res.json()
            ret_size = len(infos_new['content'])
            cache_filename = f'infos_{offset}_{offset+ret_size}.json'
            cache_pathfile = os.path.join(cache_path, cache_filename)
            with open(cache_pathfile, 'w') as f:
                json.dump(infos_new, f)
        # push cache file name
        cache_files.append(cache_pathfile)

        if total_size:
            if total_size <= offset + ret_size:
                break

        offset += fragment_size

        time.sleep(0.1)

    # load all caches and combine to target_dir
    allinfos = {'content': []}
    for cf in cache_files:
        with open(cf, 'r') as f:
            infos = json.load(f)
            allinfos['content'].extend(infos['content'])
            # logger.info(len(allinfos['content']))

    size = len(allinfos['content'])
    target_file = os.path.join(target_dir, f'infos_0_{size}.json')
    with open(target_file, 'w') as f:
        json.dump(allinfos, f)

    return allinfos


def extract_bs4(source: str) -> str:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(source, 'html.parser')
    return soup.body.get_text()


def extract_html2text(source: str) -> str:
    h = html2text.HTML2Text()
    h.ignore_links = True
    h.ignore_images = True
    h.escape_all = False
    h.ignore_anchors = True
    h.ignore_emphasis = True
    h.ignore_tables = True
    h.mark_code = True

    return h.handle(source)


def extract_text_from_html(source: str, method=extract_html2text) -> str:
    return method(source)


def retrieve_infoqcn_fulltext(referer_url: str) -> str:
    infoqcn_url = 'https://www.infoq.cn'
    detail_url = f'{infoqcn_url}/public/v1/article/getDetail'
    key = referer_url.split('/')[-1]

    headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        'Host': 'www.infoq.cn',
        # 'Cookie': 'SERVERID=1fa1f330efedec1559b3abbcb6e30f50|1566995276|1566995276',
        'Referer': referer_url,
    }

    body = {'uuid': key}
    # logger.debug(body)
    # logger.debug(referer_url)
    res = requests.post(detail_url, headers=headers, json=body)
    if res.status_code != 200:
        raise ConnectionError('Get infos not succeed!')
    article = res.json()
    # logger.debug(article)
    content = article['data'].get('content', '')
    # text = html2text.html2text(content)
    text = extract_text_from_html(content)
    # logger.debug(f'extract infoq text  {referer_url},  {text[:10]}')
    return text


fulltext_spec_dict = {
    'www.infoq.cn': retrieve_infoqcn_fulltext,
}


# TODO: make it asynchronous
def _retrieve_info_fulltext(info, target_dir='data/fulltext',
                            cache_path='data/cache/fulltext',
                            fallback_threshold=100, force_download=False,
                            force_extract=True):
    """Retrieve fulltext of an info by its url. The original html doc is stored
    in cache_path named as info.key. The extracted text doc will be stored in
    target_dir named as info.key.

    Some of the webpage may lazy load the fulltext or the page not exists
    anymore, then test if the length of the retrieved text is less than the
    fallback_threshold. If it's less than the fallback_threshold, return the
    short description of the info.

    Can make a list of hosts that lazy load the fulltext, then try to utilize
    their APIs to retrieve the fulltext.

    If force_download is True or cache not exists, then force_extract is True
    despite of the passing value.

    Cache file naming: {key}.html
    Target file naming: {key}.txt

    Returns
    -------
    str : str of fulltext of the info
    """
    txt = info['description']
    cache_filename = f'{info["key"]}.html'
    cache = os.path.join(cache_path, cache_filename)
    target_filename = f'{info["key"]}.txt'
    target = os.path.join(target_dir, target_filename)

    logger.debug(f'to retrieve fulltext of {info["url"]}')
    if force_download or not os.path.exists(cache):
        force_extract = True
        # download and store
        try:
            res = requests.get(info['url'])
            logger.debug(
                f'encoding: {res.encoding}, key: {info["key"]}, url: {info["url"]}')
            if res.status_code != 200:
                logger.info(f'Failed to retrieve html from {info["url"]}')
                tmp = ''
        except Exception as e:
            logger.error(e)
            logger.info(f'Failed to retrieve html from {info["url"]}')
            tmp = ''
        else:
            res.encoding = 'utf-8'
            tmp = res.text
        with open(cache, 'w') as f:
            # if res.encoding not in ('utf-8', 'UTF-8'):
            #     logger.debug(f'write encoding: {res.encoding} to utf-8, key: {info["key"]}')
            #     f.write(tmp.decode(res.encoding).encode('utf-8'))
            # else:
            f.write(tmp)

    if force_extract:
        # extract from cache or API, and store to target
        urlobj = urlparse(info['url'])
        if urlobj.netloc in fulltext_spec_dict.keys():
            logger.debug(
                f'to extract special url: {info["key"]}, url: {info["url"]}')
            tmp = fulltext_spec_dict[urlobj.netloc](info['url'])
        else:
            with open(cache, 'r') as f:
                # tmp = html2text.html2text(f.read())
                tmp = extract_text_from_html(f.read())
        with open(target, 'w') as f:
            f.write(tmp)

    if os.path.exists(target):
        # get fulltext
        with open(target, 'r') as f:
            tmp = f.read()

        # test if the fulltext is ok
        if len(tmp) >= fallback_threshold:
            txt = tmp
        else:
            logger.debug(f'Short text from {info["url"]}')

    return txt


def extract_info_from_url(infourl: str) -> dict:
    pass


if __name__ == '__main__':
    # logging.info('start')
    # df = fetch_infos(fulltext=True)
    # ds = df_tags()
    # infos = fetch_untagged_infos()
    # caching_untagged_infos()

    cache_path = 'data/cache/fulltext'
    cache_filename = '3df4551ab3c513422ecb39b00fc80443.html'
    cache = os.path.join(cache_path, cache_filename)
    with open(cache, 'r') as f:
        tmp = extract_text_from_html(f.read(), method=extract_bs4)
        print(tmp)
    # pass

    # %%
    #
