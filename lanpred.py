

# %%
import os
import sys
import pickle
import logging

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
import joblib

from dataset import df_lan

# logging.basicConfig(level=logging.INFO)
handler = logging.FileHandler(filename='experiment.log')
handler.setLevel(logging.INFO)
logger = logging.getLogger(__name__)
logger.addHandler(handler)


# %%
def model_search():
    dataset = df_lan()

    # Split the dataset in training and test set:
    docs_train, docs_test, y_train, y_test = train_test_split(
        dataset.data, dataset.target, test_size=0.5)

    # Build vectorizer classifier pipeline
    clf = Pipeline([
        ('vect', TfidfVectorizer()),
        # ('clf', Perceptron()),
        ('clf', SVC()),
    ])

    # grid search parameters
    parameters = {
        'vect__ngram_range': [(1, 2), (1, 3), (1, 4)],
        'vect__analyzer': ['char', 'word'],
        'vect__use_idf': [True, False],
        # 'clf__penalty': ['l1', 'l2'],
        # 'clf__alpha': [0.001, 0.0001, 0.00001],
    }
    gs_clf = GridSearchCV(clf, parameters, cv=5, n_jobs=-1)

    gs_clf.fit(docs_train, y_train)

    # Predict the outcome on the testing set in a variable named y_predicted
    y_predicted = gs_clf.predict(docs_test)

    # Print the classification report
    print(metrics.classification_report(y_test, y_predicted))

    # Plot the confusion matrix
    cm = metrics.confusion_matrix(y_test, y_predicted)
    print(cm)

    print(gs_clf.best_params_)
    print(gs_clf.best_score_)
    logger.info(f'{"-"*78}\n'
                f'score: {gs_clf.best_score_} \n'
                f'estimator: {gs_clf.best_estimator_} \n'
                f'best_params: {gs_clf.best_params_}')


def model_persist(filename='lan_pred_1', datahome='data/models'):
    dataset = df_lan()
    X, y = dataset.data, dataset.target
    clf = Pipeline(memory=None,
                   steps=[('vect',
                           TfidfVectorizer(analyzer='char', binary=False,
                                           decode_error='strict',
                                           encoding='utf-8', input='content',
                                           lowercase=True, max_df=1.0, max_features=None,
                                           min_df=1, ngram_range=(1, 4), norm='l2',
                                           preprocessor=None, smooth_idf=True,
                                           stop_words=None, strip_accents=None,
                                           sublinear_tf=False,
                                           token_pattern='(?u)\\b\\w\\w+\\b',
                                           tokenizer=None, use_idf=True,
                                           vocabulary=None)),
                          ('clf',
                           Perceptron(alpha=0.0001, class_weight=None,
                                      early_stopping=False, eta0=1.0, fit_intercept=True,
                                      max_iter=1000, n_iter_no_change=5, n_jobs=None,
                                      penalty='l2', random_state=0, shuffle=True,
                                      tol=0.001, validation_fraction=0.1, verbose=0,
                                      warm_start=False))],
                   verbose=False)

    clf.fit(X, y)

    if not os.path.exists(datahome):
        os.makedirs(datahome)

    dump_target = os.path.join(datahome, f'{filename}.joblib.gz')
    m = joblib.dump(clf, dump_target, compress=3)
    # clf2 = joblib.load(dump_target)
    # print(clf2.predict(X[:30]))


if __name__ == '__main__':
    # model_search()
    model_persist()
