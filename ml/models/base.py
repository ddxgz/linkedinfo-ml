import os
from abc import ABC, abstractmethod

import fasttext
import joblib
from typing import List, Tuple, Optional


PRETRAINED_BERT_WEIGHTS = "./data/models/bert_mini_finetuned_tagthr_20/"

MODEL_FILE = 'data/models/tags_textbased_pred_9.joblib.gz'
MLB_FILE = 'data/models/tags_textbased_pred_9_mlb.joblib.gz'
FAST_TEXT_MODEL_FILE = 'data/models/fasttext_thr10_v1_2.bin'


def model_file(model_type: str) -> str:
    if model_type == 'fasttext':
        return FAST_TEXT_MODEL_FILE
    if model_type == 'tag_clf':
        return MODEL_FILE
    return 'not matched model type'


def singleton(cls, *args, **kwargs):
    instances = {}

    def _singleton(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return _singleton


class BasePredictModel(ABC):
    @abstractmethod
    def predict(self, text):
        raise NotImplementedError(
            'users must define `predict` to use this base class')

    @abstractmethod
    def _load_model(self, modelfile: str = None):
        raise NotImplementedError(
            'users must define `_load_model` to use this base class')


class PredictModel(BasePredictModel):
    def __init__(self, modelfile: str = None):
        self.model = self._load_model(modelfile)
        super().__init__()

    @abstractmethod
    def predict(self, text):
        raise NotImplementedError(
            'users must define `predict` to use this base class')

    def _load_model(self, modelfile: str = None):
        if modelfile is None:
            modelfile = model_file('tag_clf')

        if os.path.exists(modelfile):
            model = joblib.load(modelfile)
            return model
        else:
            cwd = os.getcwd()
            raise FileNotFoundError(f'Model file not exists! The model should be'
                                    'place under ./data/models/. CWD: {cwd}')


class FastTextModel(BasePredictModel):
    def __init__(self, modelfile: str = None):
        self.model = self._load_model(modelfile)
        super().__init__()

    @abstractmethod
    def predict(self, text):
        raise NotImplementedError(
            'users must define `predict` to use this base class')

    def _load_model(self, modelfile: str = None):
        if modelfile is None:
            modelfile = model_file('fasttext')

        if os.path.exists(modelfile):
            model = fasttext.load_model(modelfile)
            return model
        else:
            raise FileNotFoundError(f'fastText Model file not exists! The model should be'
                                    'place under ./data/models/. CWD: {cwd}')
