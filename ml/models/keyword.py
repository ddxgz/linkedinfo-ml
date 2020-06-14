from gensim.summarization import keywords
from typing import List

from .base import singleton


@singleton
class KeywordPredictor(object):

    def predict(self, text, num_keywords) -> List[str]:
        raw_keywords = keywords(
            text, scores=True, lemmatize=True, words=num_keywords)

        words = [pair[0] for pair in raw_keywords]
        return words
