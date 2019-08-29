# %%
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


ds = dataset.df_tags()

# @dataclass
# class Dataset:
#     data: pd.DataFrame
#     target: pd.DataFrame
#     target_names: pd.DataFrame


# cache = dataset.fetch_infos(fulltext=True)

# data_lst = []
# tags_lst = []
# for info in cache['content']:
#     # logger.info(info['title'])
#     data_lst.append({'title': info['title'],
#                      'description': info['description'],
#                      'fulltext': info['fulltext']})
#     tags_lst.append([tag['tagID'] for tag in info['tags']])

# df_data = pd.DataFrame(data_lst)
# df_tags = pd.DataFrame(tags_lst)
# # df_tags.fillna(value=pd.np.nan, inplace=True)
# # print(df_tags)
# mlb = MultiLabelBinarizer()
# Y = mlb.fit_transform(tags_lst)
# # print(mlb.inverse_transform(Y))

# ds = Dataset(df_data, Y, mlb.classes_)

# Split the dataset in training and test set:
X_train, X_test, Y_train, Y_test = train_test_split(
    ds.data, ds.target, test_size=0.5, random_state=42)

# Build vectorizer classifier pipeline
# clf = Pipeline([
#     ('vect', TfidfVectorizer()),
#     ('clf', OneVsRestClassifier(LogisticRegression())),
#     # ('clf', OneVsRestClassifier(SVC(random_state=0))),
# ])

# vectorizer = TfidfVectorizer(max_df=0.8).fit(ds.data.description)

# X_train = vectorizer.transform(X_train.description)
# X_test = vectorizer.transform(X_test.description)
# clf = OneVsRestClassifier(LinearSVC())


class DenseTransformer(TransformerMixin):

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X.todense()


# Build vectorizer classifier pipeline
clf = Pipeline([
    ('vect', TfidfVectorizer(use_idf=True, max_df=0.8)),
    # ('clf', Perceptron()),
    ('clf', OneVsRestClassifier(LinearSVC(penalty='l1', dual=False))),
    # ('to_dense', DenseTransformer()),
    # ('clf', GaussianNB()),
])

# grid search parameters
# C_OPTIONS = [1, 10, 100, 1000]
C_OPTIONS = [10]

parameters = {
    'vect__ngram_range': [(1, 4)],
    # 'vect__ngram_range': [(1, 2), (1, 3), (1, 4)],
    # 'vect__max_df': [1, 0.9, 0.8, 0.7],
    # 'vect__min_df': [1, 0.9, 0.8, 0.7, 0],
    # 'vect__analyzer': ['char', 'word'],
    # 'vect__use_idf': [True, False],
    # 'clf__estimator__penalty': ['l1', 'l2'],
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
df_result.to_html('data/results/gridcv_results_20190829_gauNB.html')

# clf.fit(X_train.fulltext, Y_train)
# clf.fit(X_train, Y_train)
# y_score = clf.decision_function(X_test)
# pred_test = clf.predict(X_test)

# mlb = MultiLabelBinarizer()
# mlb.fit(ds.target_decoded)
# # Y = mlb.fit_transform(tags_lst)
# # # print(mlb.inverse_transform(Y))

# for (rec, pred) in zip(Y_test, Y_predicted):
#     print(
#         f'tags: {mlb.inverse_transform([rec])}\npred: {mlb.inverse_transform([pred])}')

# # # mlb = MultiLabelBinarizer()
# # # mlb.fit(ds.target)
# # # pred_tags =
# for (t, p) in zip(Y_test, pred_test):
#     # print(t)
#     if np.array_equal(t, p):
#         print(t)

# from sklearn.metrics import precision_recall_curve
# from sklearn.metrics import average_precision_score
# from inspect import signature

# step_kwargs = ({'step': 'post'}
#                if 'step' in signature(plt.fill_between).parameters
#                else {})

# # For each class
# precision = dict()
# recall = dict()
# average_precision = dict()
# for i in range(len(ds.target_names)):
#     precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i],
#                                                         y_score[:, i])
#     average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])

# # A "micro-average": quantifying score on all classes jointly
# precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.ravel(),
#                                                                 y_score.ravel())
# average_precision["micro"] = average_precision_score(Y_test, y_score,
#                                                      average="micro")
# print('Average precision score, micro-averaged over all classes: {0:0.2f}'
#       .format(average_precision["micro"]))

# plt.figure()
# plt.step(recall['micro'], precision['micro'], color='b', alpha=0.2,
#          where='post')
# plt.fill_between(recall["micro"], precision["micro"], alpha=0.2, color='b',
#                  **step_kwargs)

# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.ylim([0.0, 1.05])
# plt.xlim([0.0, 1.0])
# plt.title(
#     'Average precision score, micro-averaged over all classes: AP={0:0.2f}'
#     .format(average_precision["micro"]))
