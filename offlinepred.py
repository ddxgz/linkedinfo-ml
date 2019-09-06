from typing import List

from webapp import TagsTextModel
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
    3) predict_tags 
    4) send back predictions
    """
    infos = dataset.fetch_untagged_infos(fulltext=True)


if __name__ == '__main__':
    predict_tags_feedinfos()
