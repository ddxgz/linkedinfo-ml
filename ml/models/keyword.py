from gensim.summarization import keywords
from typing import List

from .base import singleton


@singleton
class KeywordPredictor(object):

    def predict(self, text, num_keywords) -> List[str]:
        try:
            raw_keywords = keywords(
                text, scores=True, lemmatize=True, words=num_keywords)
        # REVIEW: need further wordaround
        except IndexError as e:
            # num_keywords -= 2
            num_keywords = 1
            try:
                raw_keywords = keywords(
                    text, scores=True, lemmatize=True, words=num_keywords)
            except IndexError:
                return []

        words = [pair[0] for pair in raw_keywords]
        return words
