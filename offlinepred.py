# %%
from typing import List

from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import joblib

# from webapp import TagsTextModel
import dataset


def predict_tags(infos: List[dict]) -> List[List[str]]:
    """ An info comes in as a json (dict), use 
    description or fulltext (if presence) for prediction.

    Returns
    -------
    str of the tags acronym
    """
    if 'fulltext' in info.keys():
        text = info['fulltext']
    else:
        text = info['description']

    predicted = TAGS_MODEL.predict([text])
    # inverse transform tags
    return predicted

    return []


def predict_tags_feedinfos():
    """
    1) fetch feedinfos
    2) fetch infos fulltext 
    3) preprocessing data
    4) predict_tags 
    5) send back predictions
    """
    infos = dataset.fetch_untagged_infos(fulltext=True)

    # pre-trained model was trained on dataset with content_length_threshold 100
    content_length_threshold = 100
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

    model = joblib.load('data/models/tags_textbased_pred_1.joblib.gz')
    predicted = model.predict(df_data.fulltext)

    # pre-trained model was trained on dataset with content_length_threshold 100
    ds = dataset.df_tags(content_length_threshold=100)
    mlb = MultiLabelBinarizer()
    mlb.fit(ds.target_decoded)
    predicted_tags = mlb.inverse_transform(predicted)
    # for (title, tags) in zip(df_data.title, predicted_tags):
    #     print(f'{title}\n{tags}')
    return predicted_tags


if __name__ == '__main__':
    tags = predict_tags_feedinfos()
