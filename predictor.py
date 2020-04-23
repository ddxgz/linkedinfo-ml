import os
from abc import ABC, abstractmethod

import pandas as pd
# import requests
import joblib

import dataset

# PRETRAINED_BERT_WEIGHTS = "./data/models/google/"
# PRETRAINED_BERT_WEIGHTS = "./data/models/google/bert_uncased_L-2_H-128_A-2/"
# PRETRAINED_BERT_WEIGHTS = "google/bert_uncased_L-2_H-128_A-2"
# PRETRAINED_BERT_WEIGHTS = download_once_pretrained_transformers(
#     "google/bert_uncased_L-4_H-256_A-4")
PRETRAINED_BERT_WEIGHTS = "./data/models/bert_mini_finetuned_tagthr_20/"

MODEL_FILE = 'data/models/tags_textbased_pred_8.joblib.gz'
MLB_FILE = 'data/models/tags_textbased_pred_8_mlb.joblib.gz'


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


@singleton
class TagPredictor(object):
    def __init__(self, init=False):
        self.model = None
        self.matcher = None
        self.initialized = False
        if init:
            self.init()

    def init(self):
        self.model = TagsTextModelV3(modelfile=MODEL_FILE)
        self.tag_list = dataset.get_tags_list()
        self.tags_map = dataset.get_tags_map()

        self.initialized = True

    def predict(self, text):
        return self.model.predict(text)


def get_tag_predictor(init=False) -> TagPredictor:
    return TagPredictor()
