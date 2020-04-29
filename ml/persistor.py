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
import joblib
# import matplotlib.pyplot as plt

from ml import dataset
import mltb
from mltb.mltb.nlp.bert import bert_transform, download_once_pretrained_transformers, get_tokenizer_model

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

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline

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


def model_persist_v5(filename='tags_textbased_pred_6', datahome='data/models'):

    from mltb.experiment import multilearn_iterative_train_test_split

    col_text = 'description'
    ds = dataset.ds_info_tags(from_batch_cache='info', lan='en',
                              concate_title=True,
                              filter_tags_threshold=20)

    # train_features, test_features, train_labels, test_labels = multilearn_iterative_train_test_split(
    #     ds.data, ds.target, test_size=0.001, cols=ds.data.columns)
    features, labels = dataset.augmented_samples(
        ds.data, ds.target, level=3, crop_ratio=0.2)

    batch_size = 128
    model_name = "./data/models/bert_mini_finetuned_tagthr_20/"

    # train_features, test_features = bert_transform(
    #     train_features, test_features, col_text, model_name, batch_size)

    from mltb.bert import BertForSequenceClassificationTransformer

    clf = Pipeline([
        ('bert_tran', BertForSequenceClassificationTransformer(
            col_text, model_name, batch_size)),
        ('clf', OneVsRestClassifier(LinearSVC(penalty='l2', C=0.1, dual=True))),
    ])

    # Build vectorizer classifier pipeline
    # clf = OneVsRestClassifier(LinearSVC(penalty='l2', C=1, dual=True))

    clf.fit(features, labels)

    if not os.path.exists(datahome):
        os.makedirs(datahome)

    dump_target = os.path.join(datahome, f'{filename}.joblib.gz')
    m = joblib.dump(clf, dump_target, compress=3)

    dump_target_mlb = os.path.join(datahome, f'{filename}_mlb.joblib.gz')
    m = joblib.dump(ds.mlb, dump_target_mlb, compress=3)


class Persistor(object):
    def __init__(self):
        self.datahome = 'data/models'

    def load(self, param):
        self.ds = dataset.ds_info_tags(**param)

    # def run(self):
    #     # X, y = self.preprocess()
    #     # model = self.fit(X, y)
    #     # self.persist(model)

    def preprocess(self):
        def word_substitution(text, aug_src='wordnet'):
            # import nlpaug.flow as naf
            import nlpaug.augmenter.word as naw

            aug = naw.SynonymAug(aug_src=aug_src)
            augmented_text = aug.augment(text)
            return augmented_text

        self.train_features, self.train_labels = dataset.augmented_samples(
            self.ds.data, self.ds.target, level=2, crop_ratio=0.1,
            )

    def fit(self, model):
        self.model = model
        self.model.fit(self.train_features, self.train_labels)

    def persist(self, filename='tags_textbased_pred_9'):
        dump_target = os.path.join(self.datahome, f'{filename}.joblib.gz')
        m = joblib.dump(self.model, dump_target, compress=3)

        dump_target_mlb = os.path.join(
            self.datahome, f'{filename}_mlb.joblib.gz')
        m = joblib.dump(self.ds.mlb, dump_target_mlb, compress=3)


def model_persist_v6():
    # from mltb.bert import BertForSequenceClassificationTransformer
    per = Persistor()
    col_text = 'description'
    ds_param = dict(from_batch_cache='info', lan='en',
                    concate_title=True,
                    filter_tags_threshold=20)
    per.load(ds_param)
    per.preprocess()

    batch_size = 128
    model_name = "./data/models/bert_mini_finetuned_tagthr_20/"
    clf = Pipeline([
        ('bert_tran', mltb.mltb.nlp.bert.BertForSequenceClassificationTransformer(
            col_text, model_name, batch_size)),
        ('clf', OneVsRestClassifier(LinearSVC(penalty='l2', C=0.1, dual=True))),
    ])

    per.fit(clf)
    per.persist()


if __name__ == '__main__':
    # model_search()
    model_persist_v6()
    # save_pretrained()
