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
import joblib
import matplotlib.pyplot as plt

import dataset

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


def model_persist_v2(filename='tags_textbased_pred_2', datahome='data/models'):
    import numpy as np
    import torch
    from transformers import DistilBertModel, DistilBertTokenizer, AutoTokenizer, AutoModel

    import dataset

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    ds = dataset.df_tags(content_length_threshold=100, lan='en',
                         partial_len=1000)

    # %%
    from transformers import DistilBertModel, DistilBertTokenizer, AutoTokenizer, AutoModel, BertConfig

    # PRETRAINED_BERT_WEIGHTS = 'distilbert-base-uncased'
    PRETRAINED_BERT_WEIGHTS = "google/bert_uncased_L-2_H-128_A-2"
    # PRETRAINED_BERT_WEIGHTS = "google/bert_uncased_L-4_H-256_A-4"
    # tokenizer = DistilBertTokenizer.from_pretrained(PRETRAINED_BERT_WEIGHTS)
    # model = DistilBertModel.from_pretrained(PRETRAINED_BERT_WEIGHTS)
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_BERT_WEIGHTS)
    config = BertConfig()
    # config = BertConfig(max_position_embeddings=2048)
    model = AutoModel.from_pretrained(PRETRAINED_BERT_WEIGHTS)

    col_text = 'partial_text'
    tokenized = ds.data[col_text].apply(
        (lambda x: tokenizer.encode(x, add_special_tokens=True,
                                    max_length=128)))

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
    clf = OneVsRestClassifier(LinearSVC(penalty='l1', C=1, dual=False))

    clf.fit(features, ds.target)

    if not os.path.exists(datahome):
        os.makedirs(datahome)

    dump_target = os.path.join(datahome, f'{filename}.joblib.gz')
    m = joblib.dump(clf, dump_target, compress=3)


def save_pretrained():
    from transformers import DistilBertModel, DistilBertTokenizer, AutoTokenizer, AutoModel, BertConfig

    # PRETRAINED_BERT_WEIGHTS = 'distilbert-base-uncased'
    PRETRAINED_BERT_WEIGHTS = "google/bert_uncased_L-2_H-128_A-2"
    # PRETRAINED_BERT_WEIGHTS = "google/bert_uncased_L-4_H-256_A-4"
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_BERT_WEIGHTS)
    model = AutoModel.from_pretrained(PRETRAINED_BERT_WEIGHTS)

    model_path = f'./data/models/{PRETRAINED_BERT_WEIGHTS}/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    tokenizer.save_pretrained(model_path)
    model.save_pretrained(model_path)


if __name__ == '__main__':
    # model_search()
    # model_persist_v2()
    save_pretrained()
