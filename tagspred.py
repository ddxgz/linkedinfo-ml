# %%
import os
import datetime
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.base import TransformerMixin
from transformers import DistilBertModel, DistilBertTokenizer, AutoTokenizer, AutoModel
import torch
import nltk
import joblib
import matplotlib.pyplot as plt

import dataset
from mltb.model_utils import download_once_pretrained_transformers, get_tokenizer_model

# # logging.basicConfig(level=logging.INFO)
# handler = logging.FileHandler(filename='experiment.log')
# handler.setLevel(logging.INFO)
# logger = logging.getLogger(__name__)
# logger.addHandler(handler)


def model_search():
    ds = dataset.df_tags(content_length_threshold=100)

    # TODO: remove infos with very short text / description

    # Split the dataset in training and test set:
    X_train, X_test, Y_train, Y_test = train_test_split(
        ds.data, ds.target, test_size=0.5, random_state=42)

    # Build vectorizer classifier pipeline
    clf = Pipeline([
        ('vect', TfidfVectorizer(use_idf=True, max_df=0.8)),
        ('clf', OneVsRestClassifier(LinearSVC(penalty='l1', dual=False))),
    ])

    # grid search parameters
    C_OPTIONS = [1, 10, 100, 1000]

    parameters = {
        'vect__ngram_range': [(1, 2), (1, 3), (1, 4)],
        'vect__max_df': [1, 0.9, 0.8, 0.7],
        # 'vect__min_df': [1, 0.9, 0.8, 0.7, 0],
        # 'vect__use_idf': [True, False],
        'clf__estimator__penalty': ['l1', 'l2'],
        # 'clf__alpha': [0.001, 0.0001, 0.00001],
        'clf__estimator__C': C_OPTIONS,
    }
    gs_clf = GridSearchCV(clf, parameters, cv=5, n_jobs=-1)
    gs_clf.fit(X_train.fulltext, Y_train)
    # y_score = gs_clf.decision_function(X_test.fulltext)
    # pred_test = gs_clf.predict(X_test.fulltext)

    # Predict the outcome on the testing set in a variable named y_predicted
    Y_predicted = gs_clf.predict(X_test.fulltext)

    print(metrics.classification_report(Y_test, Y_predicted))

    # # Plot the confusion matrix
    # cm = metrics.confusion_matrix(Y_test, Y_predicted)
    # print(cm)

    print(gs_clf.best_params_)
    print(gs_clf.best_score_)

    # %%
    cols = [
        'mean_test_score',
        'mean_fit_time',
        'param_vect__ngram_range',
    ]
    df_result = pd.DataFrame(gs_clf.cv_results_)
    df_result = df_result.sort_values(by='rank_test_score')
    df_result = df_result[cols]

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    df_result.to_html(
        f'data/results/gridcv_results_{timestamp}_linearSVC.html')


def model_persist(filename='tags_textbased_pred_1', datahome='data/models'):
    ds = dataset.df_tags(content_length_threshold=100)

    X, y = ds.data.fulltext, ds.target

    clf = Pipeline([
        ('vect', TfidfVectorizer(use_idf=True,
                                 max_df=0.8, ngram_range=[1, 4])),
        ('clf', OneVsRestClassifier(LinearSVC(penalty='l1', C=10, dual=False))),
    ])

    clf.fit(X, y)

    if not os.path.exists(datahome):
        os.makedirs(datahome)

    dump_target = os.path.join(datahome, f'{filename}.joblib.gz')
    m = joblib.dump(clf, dump_target, compress=3)


def model_persist_v2(filename='tags_textbased_pred_3', datahome='data/models'):
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ds = dataset.df_tags(content_length_threshold=100, lan='en',
    #                      partial_len=1000)

    ds = dataset.ds_info_tags(from_batch_cache='fulltext', content_length_threshold=100, lan='en',
                              filter_tags_threshold=2, partial_len=3000, total_size=None)
    # %%
    from transformers import DistilBertModel, DistilBertTokenizer, AutoTokenizer, AutoModel, BertConfig

    # PRETRAINED_BERT_WEIGHTS = 'distilbert-base-uncased'
    # PRETRAINED_BERT_WEIGHTS = "google/bert_uncased_L-2_H-128_A-2"
    PRETRAINED_BERT_WEIGHTS = download_once_pretrained_transformers(
        "google/bert_uncased_L-4_H-256_A-4")
    # PRETRAINED_BERT_WEIGHTS = "google/bert_uncased_L-4_H-256_A-4"
    # tokenizer = DistilBertTokenizer.from_pretrained(PRETRAINED_BERT_WEIGHTS)
    # model = DistilBertModel.from_pretrained(PRETRAINED_BERT_WEIGHTS)
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_BERT_WEIGHTS)
    config = BertConfig()
    # config = BertConfig(max_position_embeddings=2048)
    model = AutoModel.from_pretrained(PRETRAINED_BERT_WEIGHTS)

    # col_text = 'partial_text'
    col_text = 'fulltext'
    tokenized = ds.data[col_text].apply(
        (lambda x: tokenizer.encode(x, add_special_tokens=True,
                                    max_length=256)))

    # %%
    max_len = 0
    for i in tokenized.values:
        if len(i) > max_len:
            max_len = len(i)

    padded = np.array([i + [0] * (max_len - len(i)) for i in tokenized.values])
    # %%
    attention_mask = np.where(padded != 0, 1, 0)
    attention_mask.shape
    # %%
    input_ids = torch.tensor(padded)
    attention_mask = torch.tensor(attention_mask)
    features = []

    with torch.no_grad():
        last_hidden_states = model(input_ids, attention_mask=attention_mask)
        features = last_hidden_states[0][:, 0, :].numpy()

    from sklearn.svm import SVC, LinearSVC
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn import metrics

    # Build vectorizer classifier pipeline
    clf = OneVsRestClassifier(LinearSVC(penalty='l2', C=0.1, dual=True))

    clf.fit(features, ds.target)

    if not os.path.exists(datahome):
        os.makedirs(datahome)

    dump_target = os.path.join(datahome, f'{filename}.joblib.gz')
    m = joblib.dump(clf, dump_target, compress=3)

    dump_target_mlb = os.path.join(datahome, f'{filename}_mlb.joblib.gz')
    m = joblib.dump(ds.mlb, dump_target_mlb, compress=3)


def feature_transform(model_name: str, descs: pd.DataFrame, col_text: str = 'description'):
    tokenizer, model = get_tokenizer_model(model_name)

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

    features = []
    with torch.no_grad():
        last_hidden_states = model(input_ids, attention_mask=attention_mask)
        features = last_hidden_states[0][:, 0, :].numpy()

    return features


def model_persist_v3(filename='tags_textbased_pred_4', datahome='data/models'):
    descs, labels, len_test, mlb = dataset.augmented_ds(
        col='description', level=1, test_ratio=1,
        from_batch_cache='fulltext',
        aug_level=0, lan='en',
        concate_title=True,
        filter_tags_threshold=0, partial_len=3000)

    model_name = "google/bert_uncased_L-4_H-256_A-4"

    features = feature_transform(model_name, descs, col_text='description')

    from sklearn.svm import SVC, LinearSVC
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn import metrics

    # Build vectorizer classifier pipeline
    clf = OneVsRestClassifier(LinearSVC(penalty='l2', C=1, dual=True))

    clf.fit(features, labels)

    if not os.path.exists(datahome):
        os.makedirs(datahome)

    dump_target = os.path.join(datahome, f'{filename}.joblib.gz')
    m = joblib.dump(clf, dump_target, compress=3)

    dump_target_mlb = os.path.join(datahome, f'{filename}_mlb.joblib.gz')
    m = joblib.dump(mlb, dump_target_mlb, compress=3)


def model_persist_v4(filename='tags_textbased_pred_5', datahome='data/models'):

    from mltb.experiment import multilearn_iterative_train_test_split

    COL_TEXT = 'description'
    ds = dataset.ds_info_tags(from_batch_cache='fulltext', lan='en',
                              concate_title=True,
                              filter_tags_threshold=4, partial_len=3000)

    # train_features, test_features, train_labels, test_labels = multilearn_iterative_train_test_split(
    #     ds.data, ds.target, test_size=0.001, cols=ds.data.columns)
    features, labels = dataset.augmented_samples(
        ds.data, ds.target, level=3, crop_ratio=0.2)

    from sklearn.svm import SVC, LinearSVC
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn import metrics

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline
    from sklearn import metrics

    clf = Pipeline([
        ('vect', TfidfVectorizer(use_idf=True, max_df=0.8)),
        ('clf', OneVsRestClassifier(LinearSVC(penalty='l2', C=1, dual=True))),
    ])

    clf.fit(features[COL_TEXT], labels)

    if not os.path.exists(datahome):
        os.makedirs(datahome)

    dump_target = os.path.join(datahome, f'{filename}.joblib.gz')
    m = joblib.dump(clf, dump_target, compress=3)

    dump_target_mlb = os.path.join(datahome, f'{filename}_mlb.joblib.gz')
    m = joblib.dump(ds.mlb, dump_target_mlb, compress=3)


if __name__ == '__main__':
    # model_search()
    model_persist_v4()
    # save_pretrained()
