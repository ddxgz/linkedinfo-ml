# %%
"""
To fetch data from linkedinfo.co, and prepare the data for training. Function
params ref to scikit-learn fetch_dataset.
"""
import os
import time
import json
import logging
from dataclasses import dataclass


import requests
import html2text
from urllib.parse import urlparse
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder, OneHotEncoder


logger = logging.getLogger('dataset')
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler(filename='dataset.log')
handler.setLevel(logging.INFO)
logger.addHandler(handler)
consoleHandler = logging.StreamHandler()
consoleHandler.setLevel(logging.DEBUG)
logger.addHandler(consoleHandler)

# logging.basicConfig(level=logging.INFO)

INFOS_CACHE = 'infos_0_3353.json'

LAN_ENCODING = {
    'en': 0,
    'cn': 1,
}


@dataclass
class Dataset:
    data: pd.DataFrame
    target: pd.DataFrame
    target_names: pd.DataFrame


def df_tags(*args, **kwargs):
    """
    All the data relate to identify tags of an info.

    Text data: title, description / full text
    scikit-learn MultiLabelBinarizer encoding: tags, creators(not used currently)

    Returns
    -------
    pandas.DataFrame: df with the following attribute:
        - df.data:
        - df.target: encoding of tagsID
        - df.target_names: tagsID
    """
    cache = fetch_infos(fulltext=True, *args, **kwargs)

    data_lst = []
    tags_lst = []
    for info in cache['content']:
        # logger.info(info['title'])
        data_lst.append({'title': info['title'],
                         'description': info['description'],
                         'fulltext': info['fulltext']})
        tags_lst.append([tag['tagID'] for tag in info['tags']])

    df_data = pd.DataFrame(data_lst)
    df_tags = pd.DataFrame(tags_lst)
    # df_tags.fillna(value=pd.np.nan, inplace=True)
    # print(df_tags)
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(tags_lst)

    ds = Dataset(df_data, Y, mlb.classes_)

    return ds


def df_lan(*args, **kwargs):
    """
    Only the title and short description of an info are used for training and
     prediction.

    Returns
    -------
    pandas.DataFrame: df with the following attribute:
        - df.data: title + description
        - df.target: language encoding
        - df.target_names: language in str
    """
    cache = fetch_infos(*args, **kwargs)

    row_lst = []
    for info in cache['content']:
        # logger.info(info['title'])
        rec = {}
        rec['data'] = f"{info['title']}. {info['description']}"
        rec['target_names'] = info['language']
        rec['target'] = LAN_ENCODING[info['language']]
        row_lst.append(rec)

    df = pd.DataFrame(row_lst)

    # if subset in ('train', 'test'):
    #     data = cache[subset]
    # elif subset == 'all':
    #     data_lst = list()

    return df


def fetch_infos(data_home='data', subset='train', fulltext=False,
                random_state=42, remove=(), download_if_missing=True,
                total_size=None):
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

    if os.path.exists(infos_cache):
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


def _retrieve_infos(target_dir, cache_path, fragment_size=10, total_size=None):
    """Call API to retrieve infos data. Retrieve a fragment of infos in multiple
    API calls, caches each fragment in cache_path. Combine all caches into one
    file to target_dir.

    Cache file naming: infos_{offset}_{offset+fragment_size}.json
    Target file naming: infos_{offset}_{size}.json
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
            with open(cache_pathfile, 'w') as f:
                json.dump(infos_new, f)
            ret_size = len(infos_new['content'])
        # push cache file name
        cache_files.append(cache_pathfile)

        if total_size:
            if total_size <= offset + fragment_size:
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
    text = html2text.html2text(content)
    # logger.debug(f'extract infoq text  {referer_url},  {text[:10]}')
    return text


fulltext_spec_dict = {
    'www.infoq.cn': retrieve_infoqcn_fulltext,
}


# TODO: make it asynchronous
def _retrieve_info_fulltext(info, target_dir='data/fulltext',
                            cache_path='data/cache/fulltext',
                            fallback_threshold=100, force_download=False,
                            force_extract=False):
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
                tmp = html2text.html2text(f.read())
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
            logger.info(f'Short text from {info["url"]}')

    return txt


if __name__ == '__main__':
    # logging.info('start')
    df = fetch_infos(fulltext=True)
    # ds = df_tags()

    # pass


# %%
#
