import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import torch
import nltk
from transformers import DistilBertModel, DistilBertTokenizer, AutoTokenizer, AutoModel

from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.observers import MongoObserver

import dataset
from mltb.model_utils import download_once_pretrained_transformers, get_tokenizer_model
from mltb.transformers import bert_tokenize, bert_transform
from mltb.experiment import multilearn_iterative_train_test_split

exp_name = 'linkedinfo-ml-bert-nn'
ex = Experiment(exp_name)
ex.observers.append(MongoObserver(
    url='mongodb://mongo_user:mongo_password@127.0.0.1:27017/the_database?authSource=admin', db_name='sacred'))
ex.observers.append(FileStorageObserver('sacred_exp/{exp_name}'))


@ex.config
def params():
    col_text = 'description'
    ds_param = dict(from_batch_cache='fulltext', lan='en',
                    concate_title=True,
                    filter_tags_threshold=4, partial_len=30)
    test_size = 0.3
    aug_param = dict(level=3, crop_ratio=0.2)

    bert_param = dict(
        batch_size=128, model_name="google/bert_uncased_L-4_H-256_A-4")

    train_param = dict(dropout=0.5, lr=0.005,
                       batch_size=512, epochs=4, threshold=0.5)


@ex.capture
def log_step(step, _log):
    _log.info(step)


@ex.automain
def run(_run, _log, col_text, ds_param, test_size, aug_param, bert_param, train_param):
    ds = dataset.ds_info_tags(**ds_param)
    log_step('loaded ds')

    train_features, test_features, train_labels, test_labels = multilearn_iterative_train_test_split(
        ds.data, ds.target, test_size=test_size, cols=ds.data.columns)
    log_step('splited ds')

    train_features, train_labels = dataset.augmented_samples(
        train_features, train_labels, **aug_param)
    log_step('augmented ds')

    train_features, test_features = bert_transform(
        train_features, test_features, col_text, **bert_param)
    log_step('bert transformed ds')

    n_samples, n_features = train_features.shape
    n_classes = train_labels.shape[1]

    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            # self.fc = torch.nn.Linear(n_features, 64)
            self.clf = torch.nn.Linear(n_features, n_classes)
            self.dropout = torch.nn.Dropout(train_param['dropout'])

        def forward(self, x):
            x = self.dropout(x)
            # x = F.relu(self.fc(x))
            x = self.clf(x)
            # x = F.sigmoid(self.clf(x))
            # lossfn=BCEWithLogitsLoss()
            # x=lossfn(x.view(-1, n_classes),)
            return x

    clf = Model()

    # loss_fn = torch.nn.MultiLabelSoftMarginLoss()
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(clf.parameters(), lr=train_param['lr'])

    train_features = torch.Tensor(train_features)
    train_labels = torch.Tensor(train_labels)
    test_features = torch.Tensor(test_features)
    test_labels = torch.Tensor(test_labels)

    log_step('start training')
    clf.train()
    for e in range(train_param['epochs']):
        Y_pred = clf(train_features)
        # clf.zero_grad()
        # print((Y_pred[0]))

        loss = loss_fn(Y_pred, train_labels)

        # if e % 10 == 9:
        # print(e, loss.item())
        _run.log_scalar('train.loss', loss.item())

        loss.backward()
        optimizer.step()

    clf.eval()
    Y_predicted = clf(test_features)
    loss = loss_fn(Y_predicted, test_labels)
    _run.log_scalar('val.loss', loss.item())
    # print(Y_predicted)
    # Y_predicted = torch.nn.Sigmoid()(Y_predicted)
    log_step(Y_predicted)
    # print(test_labels)

    from sklearn import metrics
    from sklearn.metrics import fbeta_score
    from scipy.optimize import fmin_l_bfgs_b, basinhopping

    def best_f1_score(true_labels, predictions):
        fbeta = 0
        thr_bst = 0
        for thr in range(0, 6):
            Y_predicted = (predictions > (thr * 0.1))

            f = metrics.average_precision_score(
                true_labels, Y_predicted, average='micro')
            if f > fbeta:
                fbeta = f
                thr_bst = thr * 0.1

        return fbeta, thr

    _run.log_scalar('best f2', best_f1_score(test_labels, Y_predicted))
    Y_predicted = (Y_predicted > train_param['threshold'])
    _run.log_scalar('avg micro prec', metrics.average_precision_score(
        test_labels, Y_predicted.tolist(), average='micro'))
    _run.log_scalar('avg micro recall', metrics.recall_score(
        test_labels, Y_predicted.tolist(), average='micro'))
