import gc
import os
from datetime import datetime
import functools
import json
import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn import metrics
from typing import List, Callable

from . import utils


RAND_STATE = 20200219
DATA_DIR = 'data'
EXP_DIR = 'exp'
TEST_DIR = 'test'

LOGGER = logging.getLogger(__name__)
utils.setup_logging(log_level=logging.DEBUG, logger=LOGGER)


def multilearn_iterative_train_test_split(features, labels, cols, test_size=0.3):
    from skmultilearn.model_selection import iterative_train_test_split

    train_features, train_labels, test_features, test_labels = iterative_train_test_split(
        np.array(features), labels, test_size=test_size)
    # print(type(train_features))
    train_features = pd.DataFrame(train_features, columns=cols)
    test_features = pd.DataFrame(test_features, columns=cols)
    return train_features, test_features, train_labels, test_labels


def data_split_v1(X: pd.DataFrame, Y: pd.Series):
    # Y = df['isFraud']
    # X = df.drop(columns=['isFraud'])

    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=0.25, shuffle=False, random_state=RAND_STATE)
    return X_train, X_val, Y_train, Y_val


def data_split_oversample_v1(X: pd.DataFrame, Y: pd.Series):
    from imblearn.over_sampling import RandomOverSampler

    # Y = df['isFraud']
    # X = df.drop(columns=['isFraud'])

    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=0.25, shuffle=False, random_state=RAND_STATE)

    ros = RandomOverSampler(random_state=RAND_STATE)
    X_train, Y_train = ros.fit_resample(X_train, Y_train)

    return X_train, X_val, Y_train, Y_val


class Experiment:
    def __init__(self, transform_pipe: Pipeline = None,
                 data_split: Callable = None, model=None, model_class=None,
                 model_param: dict = None, scorer: Callable = None):
        # self.df_nrows = df_nrows
        if transform_pipe is None:
            self.pipe = Pipeline()
        else:
            self.pipe = transform_pipe

        if data_split is None:
            self.data_split = data_split_v1
        else:
            self.data_split = data_split

        if model_class:
            self.model = model_class(**model_param)
        else:
            self.model = model

        self.model_param = model_param

        if scorer is None:
            self.scorer = metrics.roc_auc_score
        else:
            self.scorer = scorer

    def transform(self, X):
        return self.pipe.fit_transform(X)

    def run(self, X: pd.DataFrame, Y: pd.DataFrame,
            save_exp: bool = True) -> float:
        # self.df = load_df(nrows=self.df_nrows)
        # self.y = self.df['isFraud']
        # self.X = self.df.drop(columns=['isFraud'])

        self.df_nrows = len(Y)

        X = self.transform(X)
        LOGGER.info('transform done')

        # X = self.X
        # Y = self.y

        X_train, X_val, Y_train, Y_val = self.data_split(X, Y)
        LOGGER.info('data_split done')

        # print(np.where(np.isnan(X_train)))
        self.model.fit(X_train, Y_train)
        LOGGER.info('model.fit done')

        Y_pred = self.model.predict(X_val)
        LOGGER.info('model.predict done')
        # self.last_roc_auc = metrics.roc_auc_score(Y_val, Y_pred)
        self.score = self.scorer(Y_val, Y_pred)
        LOGGER.info(f'score: {self.score}')

        if save_exp:
            self.save_result()
            LOGGER.info('result saved')

        return self.score

    def save_result(self, feature_importance: bool = False):
        save_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        result = {}
        # result['roc_auc'] = self.last_roc_auc
        result['score'] = self.score
        result['score_metric'] = self.scorer.__name__
        # result['transform'] = self.pipe.get_params(deep=False)
        result['transform'] = list(self.pipe.named_steps.keys())
        result['model'] = self.model.__class__.__name__
        result['model_param'] = self.model_param
        result['data_split'] = self.data_split.__name__
        result['num_sample_rows'] = self.df_nrows
        result['save_time'] = save_time
        # if feature_importance:
        #     if hasattr(self.model, 'feature_importances_'):
        #         result['feature_importances_'] = dict(
        #             zip(self.X.columns, self.model.feature_importances_.tolist()))
        #     if hasattr(self.model, 'feature_importance'):
        #         result['feature_importances_'] = dict(
        #             zip(self.df.columns, self.model.feature_importance.tolist()))

        import pprint
        # pp = pprint.PrettyPrinter(indent=4)
        pprint.pprint(result, indent=4)

        if not os.path.exists(EXP_DIR):
            os.makedirs(EXP_DIR)
        with open(f'{EXP_DIR}/exp_{save_time}_{self.score:.4f}.json', 'w') as f:
            json.dump(result, f, indent=4)
