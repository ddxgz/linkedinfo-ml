# %%
"""
To fetch data from linkedinfo.co, and prepare the data for training. Function
params ref to scikit-learn fetch_dataset.
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
from functools import partial
from urllib.parse import urlparse
import pickle

import requests
import numpy as np
import pandas as pd
from torch.utils.data import Dataset as TorchDataset
from sklearn.preprocessing import MultiLabelBinarizer
# import nltk
# from nltk.tokenize.treebank import TreebankWordTokenizer,
# TreebankWordDetokenizer
# import spacy
# import pysnooper
from typing import List, Callable, Union, Tuple

from . import extractor


# nltk.download('punkt')
# nlp = spacy.load('en_core_web_sm')


logger = logging.getLogger('dataset')
# logger.setLevel(logging.In)
handler = logging.FileHandler(filename='dataset.log')
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

LAN_ENCODING = {
    'en': 0,
    'cn': 1,
}


class LinkedInfoTag(TorchDataset):

    def __init__(self, features, targets, data_transforms: Tuple = tuple([None, None])):
        self.features = features
        self.targets = targets
        self.target_transforms = data_transforms[0]
        self.text_transforms = data_transforms[1]

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, index):
        if self.target_transforms:
            target = self.target_transforms(self.targets[index])
        else:
            target = self.targets[index]
        if self.text_transforms:
            feature = self.text_transforms(self.features[index])
        else:
            feature = self.features[index]
        return feature, target

    @property
    def num_labels(self):
        return self.targets.shape[1]


@dataclass
class Dataset:
    data: pd.DataFrame
    target: np.ndarray
    target_names: np.ndarray
    target_decoded: List[List[str]]
    # target: pd.DataFrame
    # target_names: pd.DataFrame
    # target_decoded: pd.DataFrame
    mlb: MultiLabelBinarizer

    def __post_init__(self):
        self.dump_dir = 'data/linkedinfo/processed'

    def get_train_test(self, test_size: Union[float, int] = 0.3):
        """ Default split method `multilearn_iterative_train_test_split` from
        `mltb.mltb.experiment`.

        Returns:
            train_features(pd.DataFrame)
            test_features(pd.DataFrame)
            train_targets(np.Array)
            test_targets(np.Array)
        """
        from mltb.mltb.experiment import multilearn_iterative_train_test_split

        (self.train_features, self.test_features, self.train_targets,
         self.test_targets) = multilearn_iterative_train_test_split(
            self.data, self.target, test_size=test_size, cols=self.data.columns)
        # return multilearn_iterative_train_test_split(
        #     self.data, self.target, test_size=test_size,
        #     cols=self.data.columns)
        return (self.train_features, self.test_features, self.train_targets,
                self.test_targets)

    def dump(self, version: str = None):
        """Dump the splited train test features and targets to files. 
        `get_train_test()` should be called before dump.
        """
        import joblib

        if not version:
            from datetime import datetime

            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            version = f'ver_{timestamp}'

        output_dir = os.path.join(self.dump_dir, version)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        train_csv = os.path.join(
            output_dir, 'train.csv')
        self.train_features.to_csv(train_csv, index=False)

        test_csv = os.path.join(
            output_dir, 'test.csv')
        self.test_features.to_csv(test_csv, index=False)

        train_targets_csv = os.path.join(
            output_dir, 'train_targets.npy')
        np.save(train_targets_csv, self.train_targets)

        test_targets_csv = os.path.join(
            output_dir, 'test_targets.npy')
        np.save(test_targets_csv, self.test_targets)

        dump_target_mlb = os.path.join(
            output_dir, 'mlb.joblib.gz')
        m = joblib.dump(self.mlb, dump_target_mlb, compress=3)

    def dump_fasttext(self, fname, only_title=True, preprocess=None,
                      split_test=False, shuffle=False):
        labels = map(
            lambda x: [f'__label__{i}' for i in x], self.target_decoded)
        labels = [' '.join(i) for i in labels]
        label = pd.Series(labels).astype('string')

        lines = label.str.cat(self.data['title'], sep=' ')

        if shuffle:
            lines = lines.sample(frac=1)

        if split_test:
            if isinstance(split_test, float):
                size = int(lines.shape[0] * split_test)
            else:
                size = int(lines.shape[0] * 0.7)
            lines_tail = lines[size:]
            lines = lines[:size]
            with open(fname + '.test', 'w') as f:
                for i in lines:
                    f.writelines(i)
                    f.writelines('\n')

        with open(fname + '.train', 'w') as f:
            for i in lines:
                f.writelines(i)
                f.writelines('\n')


@dataclass
class DataappSet:
    data: pd.DataFrame
    tags: List[str]
    tags_per_info: List[List[str]]
    creators_per_info: List[List[str]]

    def __post_init__(self):
        self.target_decoded = self.tags_per_info
        self.save()

    def save(self, filename='data/pickle/dataappset.pkl') -> str:
        if not os.path.exists('data/pickle'):
            os.makedirs('data/pickle')
        with open(filename, 'wb')as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        return filename


def plot_tag_dist(targets, target_names=None):
    import plotly.express as px

    targ = pd.DataFrame(targets)
    cnts = []
    for col in targ.columns:
        cnts.append(targ[col].value_counts()[1])

    dfc = pd.DataFrame(cnts, columns=['count'], index=target_names)

    fig_Y = px.bar(dfc, x=dfc.index, y='count',
                   text='count',
                   labels={'count': 'Number of infos',
                           'x': 'Tags'})
    fig_Y.update_traces(texttemplate='%{text}')
    fig_Y.show()


def clean_text(text: str):
    text = text.strip()
    text = text.replace('\\n', '')
    text = text.replace('\\', '')
    # text = text.replace('\t', '')
    # text = re.sub('\[(.*?)\]','',text) #removes [this one]
    text = re.sub('(http:\/\/www\.|https:\/\/www\.|http:\/\/|https:\/\/)?[a-z0-9]+([\-\.]{1}[a-z0-9]+)*\.[a-z]{2,5}(:[0-9]{1,5})?(\/.*)?\s',
                  ' __url__ ', text)  # remove urls
    text = text.replace("&amp;", "and").replace(
        "&gt;", ">").replace("&lt;", "<")
    # text = re.sub('\'','',text)
    # text = re.sub(r'\d+', ' __number__ ', text) #replaces numbers
    # text = re.sub('\W', ' ', text)
    # text = re.sub(' +', ' ', text)
    text = text.replace('\r', '')
    text = text.replace('\t', '')
    text = text.replace('\n', '')
    return text


# def text_token_cat(rec):
#     d = TreebankWordDetokenizer()
#     toks = nltk.word_tokenize(rec)
#     return d.detokenize(toks)


# TODO
def remove_code_sec(text):
    return text


def filter_tags(df_data, tags_list, threshold: int = 0):
    new_tglst = [item for subl in tags_list for item in subl]
    c = Counter(new_tglst)

    tags_rm = []
    records_rm = []
    new_tags_list = []
    for t in c.most_common()[::-1]:
        if t[1] > threshold:
            break
        tags_rm.append(t[0])

    logger.debug(f'tags to remove: {tags_rm}')

    # print(len(tags_list))
    for i, tags in enumerate(tags_list):
        for tag_rm in tags_rm:
            if tag_rm in tags:
                # remove tag in list
                tags.remove(tag_rm)
        if len(tags) == 0:
            # remove info record
            records_rm.append(i)
        else:
            new_tags_list.append(tags)

    logger.debug(f'records to remove: {records_rm}')
    df_data = df_data.drop(records_rm)
    # print(tags_list)
    # print(len(new_tags_list))

    return df_data, new_tags_list


def augmented_ds(col: str = 'description', level: int = 0, test_ratio: float = 0.3, *args, **kwargs):
    from sklearn.model_selection import train_test_split
    import spacy

    nlp = spacy.load('en_core_web_sm')

    kwargs.pop('aug_level')

    ds = ds_info_tags(aug_level=0, *args, **kwargs)

    if 'random_state' in kwargs.keys():
        random_state = kwargs.pop('random_state')
    else:
        random_state = RAND_STATE

    train_features, test_features, train_labels, test_labels = train_test_split(
        ds.data, ds.target, test_size=test_ratio, random_state=random_state)

    len_test = test_features.shape[0]
    len_ori = train_features.shape[0]

    train_features = pd.concat([train_features] * (int(level) + 1),
                               ignore_index=True)
    train_labels = np.concatenate([train_labels] * (int(level) + 1),
                                  axis=0)

    def text_random_crop(rec):
        # sents = nltk.word_tokenize(rec)

        # sents = nltk.sent_tokenize(rec)
        sents = nlp(rec)
        size = len(sents)
        chop_size = size // 10
        chop_offset = random.randint(0, chop_size)
        sents_chop = sents[chop_offset:size - chop_offset - 1]

        # return rec['fulltext'][100:]
        # return ' '.join(sents_chop)
        return sents_chop.text

    train_features.iloc[len_ori:][col] = train_features.iloc[len_ori:][col].apply(
        text_random_crop)

    features = pd.concat([train_features, test_features], ignore_index=True)
    labels = np.append(train_labels, test_labels, axis=0)

    return features, labels, len_test, ds.mlb


# Deprecated, use text_augment in mltb.nlp instead
def augmented_samples(features, labels, col: str = 'description', level: int = 0,
                      oversample_weight: int = None, crop_ratio: float = 0.1,
                      aug_method: Callable = None, *args, **kwargs):
    """Used to augment the text col of the data set, the augmented copies will
    be randomly transformed a little

    Parameters
    ----------
    level : how many copies to append to the dataset. 0 means no append.

    crop_ratio : How much ratio of the text to be raondomly cropped from head or
    tail. It actually crops out about 1/ratio of the text.
    """
    from mltb.mltb import nlp as mnlp

    if 'random_state' in kwargs.keys():
        random_state = kwargs.pop('random_state')
    else:
        random_state = RAND_STATE

    len_ori = features.shape[0]

    features = pd.concat([features] * (int(level) + 1), ignore_index=True)
    labels = np.concatenate([labels] * (int(level) + 1), axis=0)

    if aug_method is not None:
        text_aug_method = aug_method
    else:
        text_aug_method = partial(
            mnlp.text_random_crop, crop_by='word', crop_ratio=crop_ratio)
    # features.iloc[:len_ori][col] = features.iloc[:len_ori][col].apply(
    #     text_token_cat)
    # features.iloc[len_ori:][col] = features.iloc[len_ori:][col].apply(
    #     text_random_crop)
    # see
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#indexing-view-versus-copy
    features[col].iloc[len_ori:] = features[col].iloc[len_ori:].apply(
        text_aug_method)

    return features, labels


def tag_terms(ds):
    tag_terms = []
    for tag in ds.target_names:
        if '-' in tag:
            tag_terms.extend(tag.split('-'))
        elif '.' in tag:
            terms = tag.split('.')
            comb = ''.join(terms)
            terms.append(comb)
            tag_terms.extend(terms)
        else:
            tag_terms.append(tag)
    if '' in tag_terms:
        tag_terms.remove('')
    return tag_terms


def ds_info_tags(from_batch_cache: str = 'fulltext',
                 tag_type: str = 'tagID', content_length_threshold: int = 100,
                 lan: str = None, filter_tags_threshold: int = None,
                 concate_title: bool = False,
                 partial_len: bool = None, remove_code: bool = True, *args, **kwargs):
    """
    All the data relate to identify tags of an info.

    Text data: title, description / full text
    scikit-learn MultiLabelBinarizer encoding: tags, creators(not used currently)

    Parameters
    ----------
    from_batch_cache: 'fulltext','info', None, optional
        Read from aggregated all infos batch cache file. Download or reload from
        small cache files.

    tag_type : optional, label or tagID, default: 'tagID'
        used to indicate which is used for tag encoding, should have no influence
        on the results.

    lan: optional, select from cn, en or None. None == both

    filter_tags_threshold : filter tags that appear at least times

    partial_len : optional, used to limit the length of fulltext to include.
        If not None, the title of an info will be put in the beginning.

    remove_code: Not completed. optional, set to remove code sections in the fulltext. The
        current impl removes only code sections that have marked as code sections

    Returns
    -------
    pandas.DataFrame: df with the following attribute:
        - df.data:
        - df.target: encoding of tagsID
        - df.target_names: tagsID
        - df.target_decoded: the list of lists contains tagsID for each info
    """
    if tag_type not in ['label', 'tagID']:
        logger.warning(
            'tag_type should be either label or tagID, use default: "tagID"')
        tag_type = 'tagID'

    cache = extractor.fetch_infos(from_batch_cache=from_batch_cache,
                                  fulltext=True, *args, **kwargs)

    require_fulltext = False if from_batch_cache == 'info' else True

    require_partial_text = False
    if partial_len is not None and partial_len > 0:
        require_partial_text = True

    data_lst = []
    tags_lst = []
    for info in cache['content']:
        # logger.info(info['title'])
        if lan:
            if info['language'] != lan:
                continue
        if require_fulltext:
            if len(info['fulltext']) < content_length_threshold:
                continue
        if len(info['description']) < content_length_threshold:
            continue
        if require_fulltext:
            if remove_code:
                info['fulltext'] = remove_code_sec(info['fulltext'])

        # info['description'] = text_token_cat(info['description'])
        info['description'] = clean_text(info['description'])
        if require_fulltext or require_partial_text:
            info['fulltext'] = clean_text(info['fulltext'])

        # TODO make partial_len based on tokens
        if require_partial_text:
            if partial_len is not None and partial_len < len(info['fulltext']):
                info['partial_text'] = info['fulltext'][:partial_len]
            else:
                info['partial_text'] = info['fulltext']

        if concate_title:
            info['description'] = info['title'] + '. ' + info['description']
            if require_fulltext:
                info['fulltext'] = info['title'] + '. ' + info['fulltext']
            if require_partial_text:
                info['partial_text'] = info['title'] + \
                    '. ' + info['partial_text']

        if require_fulltext:
            data_lst.append({'title': info['title'],
                             'language': info['language'],
                             'description': info['description'],
                             'fulltext': info['fulltext']})
        elif require_partial_text:
            data_lst.append({'title': info['title'],
                             'language': info['language'],
                             'partial_text': info['partial_text']})
        else:
            data_lst.append({'title': info['title'],
                             'description': info['description'],
                             'language': info['language']})
        tags_lst.append([tag[tag_type] for tag in info['tags']])

    df_data = pd.DataFrame(data_lst)

    if filter_tags_threshold is not None and filter_tags_threshold > 0:
        df_data, tags_lst = filter_tags(
            df_data, tags_lst, threshold=filter_tags_threshold)

    df_tags = pd.DataFrame(tags_lst)

    # df_tags.fillna(value=pd.np.nan, inplace=True)
    # print(df_tags)
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(tags_lst)

    df_data.reset_index(inplace=True)
    df_tags.reset_index(inplace=True)

    ds = Dataset(df_data, Y, mlb.classes_, tags_lst, mlb)

    return ds


def load_dataapp_set(filename='data/pickle/dataappset.pkl'):
    if not os.path.exists(filename):
        return ds_dataapp()
    with open(filename, 'rb')as f:
        tmp = pickle.load(f)
    return tmp


def ds_dataapp(tag_type: str = 'tagID'):
    infos = extractor.fetch_infos(from_batch_cache='info', fulltext=False)

    tag_type = 'label'
    data_lst = []
    tags_lst = []
    creators_lst = []

    for info in infos['content']:
        u = urlparse(info['url'])
        data_lst.append({'title': info['title'],
                         'language': info['language'],
                         'url': info['url'],
                         'host': u.hostname,
                         'domain': u.netloc,
                         })

        tags_lst.append([tag[tag_type] for tag in info['tags']])
        if not info['creators']:
            # print(info["title"])
            continue
        creators_lst.append([creator['label'] for creator in info['creators']])
    df_data = pd.DataFrame(data_lst)

    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(tags_lst)
    # df_tags = pd.DataFrame(tags_lst)
    # df_creators = pd.DataFrame(creators_lst)
    return DataappSet(df_data, mlb.classes_, tags_lst, creators_lst)


def tags(tag_type: str = 'label'):
    infos = extractor.fetch_infos(from_batch_cache='info', fulltext=False)

    tags_lst = []

    for info in infos['content']:

        tags_lst.extend([tag[tag_type] for tag in info['tags']])

    return tags_lst


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


# TODO: Deprecated
def df_tags(tag_type: str = 'tagID', content_length_threshold: int = 100, lan: str = None,
            partial_len: bool = None, remove_code: bool = True, *args, **kwargs):
    """
    All the data relate to identify tags of an info.

    Text data: title, description / full text
    scikit-learn MultiLabelBinarizer encoding: tags, creators(not used currently)

    Parameters
    ----------
    tag_type : optional, label or tagID, default: 'tagID'
        used to indicate which is used for tag encoding, should have no influence
        on the results.

    lan: optional, select from cn, en or None. None == both

    partial_len : optional, used to limit the length of fulltext to include.
        If not None, the title of an info will be put in the beginning.

    remove_code: optional, set to remove code sections in the fulltext. The
        current impl removes only code sections that have marked as code sections

    Returns
    -------
    pandas.DataFrame: df with the following attribute:
        - df.data:
        - df.target: encoding of tagsID
        - df.target_names: tagsID
        - df.target_decoded: the list of lists contains tagsID for each info
    """
    if tag_type not in ['label', 'tagID']:
        logger.warning(
            'tag_type should be either label or tagID, use default: "tagID"')
        tag_type = 'tagID'

    cache = extractor.fetch_infos(fulltext=True, *args, **kwargs)

    data_lst = []
    tags_lst = []
    for info in cache['content']:
        # logger.info(info['title'])
        if lan:
            if info['language'] != lan:
                continue
        if len(info['fulltext']) < content_length_threshold:
            continue
        if len(info['description']) < content_length_threshold:
            continue
        if remove_code:
            info['fulltext'] = remove_code_sec(info['fulltext'])
        info['fulltext'] = clean_text(info['fulltext'])
        if partial_len is not None and partial_len > 0:
            info['partial_text'] = info['title'] + '. '
            if partial_len < len(info['fulltext']):
                info['partial_text'] += info['fulltext'][:partial_len]
            else:
                info['partial_text'] += info['fulltext']

        data_lst.append({'title': info['title'],
                         'description': info['description'],
                         'language': info['language'],
                         'fulltext': info['fulltext'],
                         'partial_text': info['partial_text']})
        tags_lst.append([tag[tag_type] for tag in info['tags']])

    df_data = pd.DataFrame(data_lst)
    df_tags = pd.DataFrame(tags_lst)
    # df_tags.fillna(value=pd.np.nan, inplace=True)
    # print(df_tags)
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(tags_lst)

    ds = Dataset(df_data, Y, mlb.classes_, tags_lst, mlb)

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
    cache = extractor.fetch_infos(*args, **kwargs)

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
