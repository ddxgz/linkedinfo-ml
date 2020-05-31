import os
from abc import ABC, abstractmethod

import pandas as pd
# import requests
import joblib
from typing import List, Tuple, Optional

from .language import LanPredictor
from .tag import TagPredictor


def get_tag_predictor(init=False, test_model=False) -> TagPredictor:
    return TagPredictor(init=init, test_model=test_model)


def get_lan_predictor(init=False) -> LanPredictor:
    return LanPredictor(init=init)
