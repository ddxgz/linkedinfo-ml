import os
from datetime import datetime
import functools
import json

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn import metrics
import torch
import nltk
from typing import List, Callable

from mltb.model_utils import download_once_pretrained_transformers, get_tokenizer_model


def bert_tokenize(tokenizer, descs: pd.DataFrame, col_text: str = 'description'):

    max_length = descs[col_text].apply(
        lambda x: len(nltk.word_tokenize(x))).max()
    if max_length > 512:
        max_length = 512
    encoded = descs[col_text].apply(
        (lambda x: tokenizer.encode_plus(x, add_special_tokens=True,
                                         pad_to_max_length=True,
                                         return_attention_mask=True,
                                         max_length=max_length,
                                         return_tensors='pt')))

    input_ids = torch.cat(tuple(encoded.apply(lambda x: x['input_ids'])))
    attention_mask = torch.cat(
        tuple(encoded.apply(lambda x: x['attention_mask'])))

    return input_ids, attention_mask


def bert_transform(train_features, test_features, col_text: str,
                   model_name: str = "google/bert_uncased_L-4_H-256_A-4",
                   batch_size: int = 128):

    tokenizer, model = get_tokenizer_model(model_name)

    input_ids, attention_mask = bert_tokenize(
        tokenizer, train_features, col_text=col_text)
    input_ids_test, attention_mask_test = bert_tokenize(
        tokenizer, test_features, col_text=col_text)

    # train_features= torch.Tensor(train_features)
    # train_labels= torch.Tensor(train_labels)
    # test_features= torch.Tensor(test_features)
    # test_labels= torch.Tensor(test_labels)

    # train_set = torch.utils.data.TensorDataset(
    #     input_ids, attention_mask, train_labels)
    # test_set = torch.utils.data.TensorDataset(
    #     input_ids_test, attention_mask_test, test_labels)
    train_set = torch.utils.data.TensorDataset(
        input_ids, attention_mask)
    test_set = torch.utils.data.TensorDataset(
        input_ids_test, attention_mask_test)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)

    train_features = []

    for batch in train_loader:

        with torch.no_grad():
            last_hidden_states = model(batch[0], attention_mask=batch[1])
            features_batch = last_hidden_states[0][:, 0, :].numpy()
            train_features.extend(features_batch)

    train_features = np.array(train_features)

    test_features = []
    for batch in test_loader:

        with torch.no_grad():
            last_hidden_states = model(batch[0], attention_mask=batch[1])
            features_batch = last_hidden_states[0][:, 0, :].numpy()
            test_features.extend(features_batch)

    test_features = np.array(test_features)

    return train_features, test_features


class BertTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, col_text, model_name, batch_size):
        self.col_text = col_text
        self.model_name = model_name
        self.batch_size = batch_size

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        tokenizer, model = get_tokenizer_model(self.model_name)

        input_ids, attention_mask = bert_tokenize(
            tokenizer, X, col_text=self.col_text)

        train_set = torch.utils.data.TensorDataset(
            input_ids, attention_mask)

        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=self.batch_size)

        train_features = []

        for batch in train_loader:

            with torch.no_grad():
                last_hidden_states = model(batch[0], attention_mask=batch[1])
                features_batch = last_hidden_states[0][:, 0, :].numpy()
                train_features.extend(features_batch)

        train_features = np.array(train_features)

        return train_features


def cat_cols(df: pd.DataFrame) -> List[str]:
    cols: List[str] = []

    cols.append('ProductCD')

    cols_card = [c for c in df.columns if 'card' in c]
    cols.extend(cols_card)

    cols_addr = ['addr1', 'addr2']
    cols.extend(cols_addr)

    cols_emaildomain = [c for c in df if 'email' in c]
    cols.extend(cols_emaildomain)

    cols_M = [c for c in df if c.startswith('M')]
    cols.extend(cols_M)

    cols.extend(['DeviceType', 'DeviceInfo'])

    cols_id = [c for c in df if c.startswith('id')]
    cols.extend(cols_id)

    return cols


def num_cols(df: pd.DataFrame, target_col='isFraud') -> List[str]:
    cols_cat = cat_cols(df)
    cats = df[cols_cat]
    cols_num = list(set(df.columns) - set(cols_cat))

    if target_col in cols_num:
        cols_num.remove(target_col)

    return cols_num


def missing_ratio_col(df):
    df_na = (df.isna().sum() / len(df)) * 100
    if isinstance(df, pd.DataFrame):
        df_na = df_na.drop(
            df_na[df_na == 0].index).sort_values(ascending=False)
        missing_data = pd.DataFrame({'Missing Ratio %': df_na})
    else:
        missing_data = pd.DataFrame({'Missing Ratio %': df_na}, index=[0])

    return missing_data


class NumColsNaMedianFiller(TransformerMixin, BaseEstimator):

    def fit(self, X, y=None):
        return self

    def transform(self, df):
        cols_cat = cat_cols(df)
        cols_num = list(set(df.columns) - set(cols_cat))

        for col in cols_num:
            median = df[col].median()
            df[col].fillna(median, inplace=True)

        return df


class NumColsNegFiller(TransformerMixin, BaseEstimator):

    def fit(self, X, y=None):
        return self

    def transform(self, df):
        cols_num = num_cols(df)

        for col in cols_num:
            df[col].fillna(-999, inplace=True)

        return df


class NumColsRatioDropper(TransformerMixin, BaseEstimator):
    def __init__(self, ratio: float = 0.5):
        self.ratio = ratio

    def fit(self, X, y=None):
        return self

    def transform(self, df):
        # print(X[self.attribute_names].columns)

        cols_cat = cat_cols(df)
        cats = df[cols_cat]
        # nums = df.drop(columns=cols_cat)
        # cols_num = df[~df[cols_cat]].columns
        cols_num = list(set(df.columns) - set(cols_cat))
        nums = df[cols_num]

        ratio = self.ratio * 100
        missings = missing_ratio_col(nums)
        # print(missings)
        inds = missings[missings['Missing Ratio %'] > ratio].index
        df = df.drop(columns=inds)
        return df


class ColsDropper(TransformerMixin, BaseEstimator):
    def __init__(self, cols: List[str]):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, df):
        return df.drop(columns=self.cols)


class DataFrameSelector(TransformerMixin, BaseEstimator):
    def __init__(self, col_names):
        self.attribute_names = col_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        print(X[self.attribute_names].columns)

        return X[self.attribute_names].values


class DummyEncoder(TransformerMixin, BaseEstimator):

    def fit(self, X, y=None):
        return self

    def transform(self, df):
        cols_cat = cat_cols(df)

        cats = df[cols_cat]
        noncats = df.drop(columns=cols_cat)

        cats = cats.astype('category')
        cats_enc = pd.get_dummies(cats, prefix=cols_cat, dummy_na=True)

        return noncats.join(cats_enc)


# Label encoding is OK when we're using tree models
class MyLabelEncoder(TransformerMixin, BaseEstimator):

    def fit(self, X, y=None):
        return self

    def transform(self, df):
        cols_cat = cat_cols(df)

        for col in cols_cat:
            df[col] = df[col].astype('category').cat.add_categories(
                'missing').fillna('missing')
            le = preprocessing.LabelEncoder()
            # TODO add test set together to encoding
            # le.fit(df[col].astype(str).values)
            df[col] = le.fit_transform(df[col].astype(str).values)
        return df


class FrequencyEncoder(TransformerMixin, BaseEstimator):
    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, df):
        for col in self.cols:
            vc = df[col].value_counts(dropna=True, normalize=True).to_dict()
            vc[-1] = -1
            nm = col + '_FE'
            df[nm] = df[col].map(vc)
            df[nm] = df[nm].astype('float32')
        return df


class CombineEncoder(TransformerMixin, BaseEstimator):
    def __init__(self, cols_pairs: List[List[str]]):
        self.cols_pairs = cols_pairs

    def fit(self, X, y=None):
        return self

    def transform(self, df):
        for pair in self.cols_pairs:
            col1 = pair[0]
            col2 = pair[1]
            nm = col1 + '_' + col2
            df[nm] = df[col1].astype(str) + '_' + df[col2].astype(str)
            df[nm] = df[nm].astype('category')
            # print(nm, ', ', end='')
        return df


class AggregateEncoder(TransformerMixin, BaseEstimator):
    def __init__(self, main_cols: List[str], uids: List[str], aggr_types: List[str],
                 fill_na: bool = True, use_na: bool = False):
        self.main_cols = main_cols
        self.uids = uids
        self.aggr_types = aggr_types
        self.use_na = use_na
        self.fill_na = fill_na

    def fit(self, X, y=None):
        return self

    def transform(self, df):
        for col in self.main_cols:
            for uid in self.uids:
                for aggr_type in self.aggr_types:
                    col_new = f'{col}_{uid}_{aggr_type}'
                    tmp = df.groupby([uid])[col].agg([aggr_type]).reset_index().rename(
                        columns={aggr_type: col_new})
                    tmp.index = list(tmp[uid])
                    tmp = tmp[col_new].to_dict()
                    df[col_new] = df[uid].map(tmp).astype('float32')
                    if self.fill_na:
                        df[col_new].fillna(-1, inplace=True)
        return df
