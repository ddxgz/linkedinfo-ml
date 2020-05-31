import cld3

from .base import singleton, PredictModel


@singleton
class LanModel(PredictModel):

    def predict(self, text):
        return self.model.predict(text)


class LanPredictor(object):
    def __init__(self, init: bool = False):
        self.initialized = False
        if init:
            self.init()

    def init(self):
        self.initialized = True

    def predict(self, text) -> str:
        pred = cld3.get_language(text)
        # cld3.get_frequent_languages(text, 2)
        return pred.language
