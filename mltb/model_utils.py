import os

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, BertConfig

import dataset


def get_tokenizer_model(model_name: str = "google/bert_uncased_L-2_H-128_A-2",
                        config: BertConfig = None):
    model_name = download_once_pretrained_transformers(model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    return tokenizer, model


def save_pretrained_bert(model_name: str) -> str:

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    model_path = f'./data/models/{model_name}/'

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    tokenizer.save_pretrained(model_path)
    model.save_pretrained(model_path)

    return model_path


def download_once_pretrained_transformers(
        model_name: str = "google/bert_uncased_L-2_H-128_A-2") -> str:
    model_path = f'./data/models/{model_name}/'

    if not os.path.exists(model_path):
        return save_pretrained_bert(model_name)

    return model_path
