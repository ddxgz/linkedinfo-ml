# %% [markdown]
# Go back to [pcx.linkedinfo.co](https://pcx.linkedinfo.co)

# %% [markdown]
# # Multi-label classification to predict topic tags of technical articles from LinkedInfo.co

# This code snippet is to predict topic tags based on the text of an article.
# Each article could have 1 or more tags (usually have at least 1 tag), and the
# tags are not mutually exclusive. So this is a multi-label classification
# problem. It's different from multi-class classification, the classes in
# multi-class classification are mutually exclusive, i.e., each item belongs to
# 1 and  only 1 class.

# In this snippet, we will use `OneVsRestClassifier` (the One-Vs-the-Rest) in
# scikit-learn to process the multi-label classification. The article data will
# be retrieved from [LinkedInfo.co](https://linkedinfo.co) via Web API.
# 1. Preprocessing data and explore the method
# 2. Search for best model include SVM and Neural Networks (Will be updated)
# 3. Training
# 4. Testing on new untagged_infos

# The methods in this snippet should give credits to
# [Working With Text Data - scikit-learn](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html)

# ## Preprocessing data and explore the method
# `dataset.df_tags` fetches the data set from [LinkedInfo.co](https://linkedinfo.co).
# It calls Web API of LinkedInfo.co to retrieve the article list, and then download
# and extract the full text of each article based on an article's url. The tags
# of each article are encoded using `MultiLabelBinarizer` in scikit-learn.
# The implementation of the code could be found in [dataset.py](https://github.com/ddxgz/linkedinfo-ml-models/blob/master/dataset.py).
# We've set the parameter of `content_length_threshold` to 100 to screen out the
# articles with less than 100 for the description or full text.
# %%
import dataset

ds = dataset.df_tags(content_length_threshold=100)
# %%[markdown]
# The dataset contains 3353 articles by the time retrieved the data.
# The dataset re returned as an object with the following attribute:
# > - ds.data: pandas.DataFrame with cols of title, description, fulltext
# > - ds.target: encoding of tagsID
# > - ds.target_names: tagsID
# > - ds.target_decoded: the list of lists contains tagsID for each info
#
# %%
ds.data.head()
ds.target[:5]
ds.target_names[:5]
ds.target_decoded[:5]

# %%[markdown]
# The following snippet is the actual process of getting the above dataset, by
# reading from file.

# %%
import json
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

# infos = dataset.fetch_infos(fulltext=True)
infos_file = 'data/infos/infos_0_3353_fulltext.json'
with open(infos_file, 'r') as f:
    infos = json.load(f)

content_length_threshold = 100

data_lst = []
tags_lst = []
for info in infos['content']:
    if len(info['fulltext']) < content_length_threshold:
        continue
    if len(info['description']) < content_length_threshold:
        continue
    data_lst.append({'title': info['title'],
                     'description': info['description'],
                     'fulltext': info['fulltext']})
    tags_lst.append([tag['tagID'] for tag in info['tags']])

df_data = pd.DataFrame(data_lst)
df_tags = pd.DataFrame(tags_lst)

# fit and transform the binarizer
mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(tags_lst)
Y.shape

# %%[markdown]
#
# %%[markdown]

# Now we've transformed the target (tags) but we cannot directly perform the
# algorithms on the text data, so we have to process and transform them into
#  vectors. In order to do this, we will use `TfidfVectorizer` to preprocess,
# tokenize, filter stop words and transform the text data. The `TfidfVectorizer`
# implements the [*tf-idf*](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
# (Term Frequency-Inverse Deocument Frequency) to reflect
# how important a word is to to a docuemnt in a collection of documents.

# %%
from sklearn.feature_extraction.text import TfidfVectorizer

# Use the default parameters for now, use_idf=True in default
vectorizer = TfidfVectorizer()
# Use the short descriptions for now for faster processing
X = vectorizer.fit_transform(df_data.description)
X.shape

# %%[markdown]
# As mentioned in the beginning, this is a multi-label classification problem,
# we will use `OneVsRestClassifier` to tackle our problem. And firstly we will
# use the SVM (Support Vector Machines) with linear kernel, implemented as
# `LinearSVC` in scikit-learn, to do the classification.
# %%
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

# Use default parameters, and train and test with small set of samples.
clf = OneVsRestClassifier(LinearSVC())

from sklearn.utils import resample

X_sample, Y_sample = resample(
    X, Y, n_samples=1000, replace=False, random_state=7)

# X_sample_test, Y_sample_test = resample(
#     X, Y, n_samples=10, replace=False, random_state=1)

X_sample_train, X_sample_test, Y_sample_train, Y_sample_test = train_test_split(
    X_sample, Y_sample, test_size=0.01, random_state=42)

clf.fit(X_sample, Y_sample)
Y_sample_pred = clf.predict(X_sample_test)

# Inverse transform the vectors back to tags
pred_transformed = mlb.inverse_transform(Y_sample_pred)
test_transformed = mlb.inverse_transform(Y_sample_test)

for (t, p) in zip(test_transformed, pred_transformed):
    print(f'tags: {t} predicted as: {p}')

# %%[markdown]
# Though not very satisfied, this classifier predicted right a few tags. Next
# we'll try to search for the best parameters for the classifier and train with
# fulltext of articles.

# ## Search for best model parameters for SVM with linear kernel
# For the estimators `TfidfVectorizer` and `LinearSVC`,
# they both have many parameters could be tuned for better performance. We'll
# the `GridSearchCV` to search for the best parameters with the help of `Pipeline`.

# %%
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV


# Split the dataset into training and test set, and use fulltext of articles:
X_train, X_test, Y_train, Y_test = train_test_split(
    df_data.fulltext, Y, test_size=0.5, random_state=42)

# Build vectorizer classifier pipeline
clf = Pipeline([
    ('vect', TfidfVectorizer()),
    ('clf', OneVsRestClassifier(LinearSVC())),
])

# Grid search parameters, I minimized the parameter set based on previous
# experience to accelerate the processing speed.
# And the combination of penalty='l1' and loss='squared_hinge' are not supported when dual=True
parameters = {
    'vect__ngram_range': [(1, 2), (1, 3), (1, 4)],
    'vect__max_df': [1, 0.9, 0.8, 0.7],
    'vect__min_df': [1, 0.9, 0.8, 0.7, 0],
    'vect__use_idf': [True, False],
    'clf__estimator__penalty': ['l1', 'l2'],
    'clf__estimator__C': [1, 10, 100, 1000],
    'clf__estimator__dual': [False],
}

gs_clf = GridSearchCV(clf, parameters, cv=5, n_jobs=-1)
gs_clf.fit(X_train, Y_train)

# %%
import datetime
from sklearn import metrics


# Predict the outcome on the testing set in a variable named y_predicted
Y_predicted = gs_clf.predict(X_test)

print(metrics.classification_report(Y_test, Y_predicted))
print(gs_clf.best_params_)
print(gs_clf.best_score_)

# Export some of the result cols
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

# %%[markdown]
# Based on the grid search results, we found the following parameters combined
# with the default parameters have the best performance. Now let's see how it
# will perform.

# %%
X_train, X_test, Y_train, Y_test = train_test_split(
    df_data, Y, test_size=0.2, random_state=42)

clf = Pipeline([
    ('vect', TfidfVectorizer(use_idf=True,
                             max_df=0.8, ngram_range=[1, 4])),
    ('clf', OneVsRestClassifier(LinearSVC(penalty='l1', C=10, dual=False))),
])

clf.fit(X_train.fulltext, Y_train)


# %%
Y_pred = clf.predict(X_test.fulltext)

# Inverse transform the vectors back to tags
pred_transformed = mlb.inverse_transform(Y_pred)
test_transformed = mlb.inverse_transform(Y_test)

#%%
for (title, t, p) in zip(X_test.title, test_transformed, pred_transformed):
    print(f'info title: {title} \n'
          f'tags: {t} predicted as: {p}')

# %%[markdown]
# Now we are
# %%
# Use pretrained model to make the experiment faster.
import joblib

clf = joblib.load('data/models/tags_textbased_pred_1.joblib.gz')

# %%
# tags = predict_tags_feedinfos()
infos = dataset.fetch_untagged_infos(fulltext=True)

content_length_threshold = 20
data_lst = []
for info in infos['content']:
    if len(info['fulltext']) < content_length_threshold:
        continue
    if len(info['description']) < content_length_threshold:
        continue
    data_lst.append({'title': info['title'],
                     'description': info['description'],
                     'fulltext': info['fulltext']})

df_data = pd.DataFrame(data_lst)

untagged_predicted = clf.predict(df_data.fulltext)

# %%
# pre-trained model was trained on dataset with content_length_threshold 100
ds = dataset.df_tags(content_length_threshold=100)
mlb = MultiLabelBinarizer()
mlb.fit(ds.target_decoded)
# %%
predicted_tags = mlb.inverse_transform(untagged_predicted)

# %%

for (title, tags) in zip(df_data.title, predicted_tags):
    print(f'{title}\n{tags}')
