# %%
"""
To fetch data from linkedinfo.co, and prepare the data for training. Function
params ref to scikit-learn fetch_dataset.
"""
import os
import time
import json
import logging


import requests
import pandas as pd


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


INFOS_CACHE = 'infos_0_3353.json'


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
    lan_encoding = {
        'en': 0,
        'cn': 1,
    }
    for info in cache['content']:
        # logger.info(info['title'])
        rec = {}
        rec['data'] = f"{info['title']}. {info['description']}"
        rec['target_names'] = info['language']
        rec['target'] = lan_encoding[info['language']]
        row_lst.append(rec)

    df = pd.DataFrame(row_lst)

    # if subset in ('train', 'test'):
    #     data = cache[subset]
    # elif subset == 'all':
    #     data_lst = list()

    return df


def fetch_infos(data_home='data', subset='train', random_state=42, remove=(),
                download_if_missing=True, total_size=None):
    """Load the infos from linkedinfo.co or local cache.
    Parameters
    ----------
    data_home : optional, default: 'data'
        Specify a download and cache folder for the datasets. If None,
        all scikit-learn data is stored in './data' subfolders.

    subset : 'train' or 'test', 'all', optional
        Select the dataset to load: 'train' for the training set, 'test'
        for the test set, 'all' for both, with shuffled ordering.

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


if __name__ == '__main__':
    df = fetch_infos()


# %%
