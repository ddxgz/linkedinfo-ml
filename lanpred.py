

import sys

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Perceptron
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics

from dataset import df_lan


dataset = df_lan()

# Split the dataset in training and test set:
docs_train, docs_test, y_train, y_test = train_test_split(
    dataset.data, dataset.target, test_size=0.5)


# Build vectorizer classifier pipeline
clf = Pipeline([
    ('vect', TfidfVectorizer()),
    ('clf', Perceptron()),
])

# grid search parameters
parameters = {
    'vect__ngram_range': [(1, 2), (1, 3), (1, 4)],
    'vect__analyzer': ['char', 'word'],
    'vect__use_idf': [True, False],
    'clf__penalty': ['l1', 'l2'],
    'clf__alpha': [0.001, 0.0001, 0.00001],
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
