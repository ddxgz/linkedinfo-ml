import os
from abc import ABC, abstractmethod

import pandas as pd
# import requests
import fasttext
import joblib
from typing import List, Tuple, Optional

from .base import singleton, PredictModel, FastTextModel


# PRETRAINED_BERT_WEIGHTS = "./data/models/google/"
# PRETRAINED_BERT_WEIGHTS = "./data/models/google/bert_uncased_L-2_H-128_A-2/"
# PRETRAINED_BERT_WEIGHTS = "google/bert_uncased_L-2_H-128_A-2"
# PRETRAINED_BERT_WEIGHTS = download_once_pretrained_transformers(
#     "google/bert_uncased_L-4_H-256_A-4")
PRETRAINED_BERT_WEIGHTS = "./data/models/bert_mini_finetuned_tagthr_20/"

MODEL_FILE = 'data/models/tags_textbased_pred_9.joblib.gz'
MLB_FILE = 'data/models/tags_textbased_pred_9_mlb.joblib.gz'


@singleton
class TagsTestModel(PredictModel):
    def __init__(self):
        pass

    def predict(self, text):
        # return [[]]
        return [['test-tags', 'python']]


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

        from transformers import AutoTokenizer, AutoModel
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
        import torch

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
class TagsFasttextModel(FastTextModel):
    def __init__(self, modelfile: str):
        super().__init__(modelfile)

    def predict(self, text, k: int = 2) -> List[str]:
        pred = self.model.predict(text, k=k)
        tags = [tag.lstrip('__label__') for tag in pred[0]]
        return tags


def append_map_tags(predictor, tags: List[str], text: str) -> List[str]:
    # if not TAG_PRED.initialized:
    #     TAG_PRED.init()

    map_tags: List[str] = []

    # print(text)
    # toks = nltk.word_tokenize(text)
    toks = text.split(' ,.!?')
    if len(toks) > 20:
        toks = toks[:20]

    # print(toks)

    # d = TreebankWordDetokenizer()
    # text = d.detokenize(toks).lower()

    # print(text)
    # for tag in TAGS_LIST:
    for tag in predictor.tag_list:
        if tag['label'] in toks:
            map_tags.append(tag['tagID'])
        if len(tag['label']) > 9:
            if tag['label'] in text:
                map_tags.append(tag['tagID'])

    # for k, v in TAGS_MAP.items():
    for k, v in predictor.tags_map.items():
        if k in text:
            map_tags.append(v)

    # print(tags)
    # print(map_tags)
    if map_tags:
        tags = list(tags) + map_tags

    return list(set(tags))


# TODO add NER/Matcher based Predictor class as a component instead of function
@singleton
class TagPredictor(object):
    def __init__(self, init: bool = False, test_model: bool = False):
        self.model: PredictModel
        self.matcher = None
        self.test_model = test_model
        self.initialized = False
        if init:
            self.init()

    def init(self):
        import spacy
        from spacy.matcher import PhraseMatcher

        from .. import dataset

        if self.test_model:
            print('loading test model...')
            self.model = TagsTestModel()
            self.tag_list = [{'tagID': 'python', 'label': 'Python'},
                             {'tagID': 'machine-learning', 'label': 'Machine Learning'}]
            self.tags_map = {"ML": "machine-learning"}
            # self.tag_list = dataset.get_tags_list()
            # self.tags_map = dataset.get_tags_map()

            from spacy.language import Language
            self.nlp = Language()
        else:
            self.model = TagsTextModelV3(modelfile=MODEL_FILE)
            self.tag_list = dataset.get_tags_list()
            self.tags_map = dataset.get_tags_map()

            self.nlp = spacy.load('en_core_web_sm')

        for pair in self.tag_list:
            self.tags_map[pair['label']] = pair['tagID']

        matcher = PhraseMatcher(self.nlp.vocab, max_length=3)

        # for tag in self.tag_list:
        #     matcher.add(tag['label'], None, self.nlp(tag['tagID']))
        #     matcher.add(tag['label'], None, self.nlp(tag['label']))
        for k, v in self.tags_map.items():
            matcher.add(v, None, self.nlp(k.lower()))
            matcher.add(v, None, self.nlp(k))
            matcher.add(v, None, self.nlp(v))

        # print(len(matcher))
        self.matcher = matcher

        self.initialized = True

    def predict(self, text, entity_tags: bool = False) -> List[str]:
        tags = self.model.predict([text])[0]
        if entity_tags:
            # tags = append_map_tags(self, tags, text)
            tags = self._append_map_tags(tags, text)
        return tags

    def _append_map_tags(self, tags, text, min_label_len: int = 9) -> List[str]:
        tags = list(tags)
        ent_tags = self.matcher(self.nlp(text))
        for ent in ent_tags:
            label = self.nlp.vocab.strings[ent[0]]
            if len(label) < min_label_len:
                continue
            tag = self.tags_map.get(label)
            if tag:
                tags.append(tag)
            elif label in self.tags_map.values():
                tags.append(label)

        return list(set(tags))
