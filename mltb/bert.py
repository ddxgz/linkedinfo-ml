import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, TransformerMixin
import nltk
from transformers import (BertPreTrainedModel,
                          DistilBertModel, DistilBertTokenizer, AutoTokenizer, AutoModel, BertModel,
                          BertForSequenceClassification, AdamW, BertModel, BertConfig)

import dataset


nltk.download('punkt')


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


class BertForSequenceMultiLabelClassification(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForSequenceMultiLabelClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        logtis = torch.sigmoid(logits)

        # add hidden states and attention if they are here
        outputs = (logits,) + outputs[2:]

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                labels = torch.max(labels, 1)[1]
                loss = loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1))
                # loss = loss_fct(
                #     logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
