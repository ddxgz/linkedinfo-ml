import unittest
from unittest import mock
import tempfile

import numpy as np
import pandas as pd
import fasttext

from ml.models import tag
from ml import dataset
from ml.persistor import save_model_url


class Tester(unittest.TestCase):
    def setUp(self):
        # col_text = 'description'
        # ds_param = dict(from_batch_cache='info', lan='en',
        #                 concate_title=True,
        #                 filter_tags_threshold=30)
        # self.ds = dataset.ds_info_tags(**ds_param)

        self.tempd = tempfile.TemporaryDirectory()
        # self.fasttestfile = self.tempd.name + '/linkedinfo'
        # self.ds.dump_fasttext(self.fasttestfile, split_test=0.3, shuffle=True)

    def tearDown(self):
        self.tempd.cleanup()

    def test_save_model_url(self):
        save_model_url("SK_MODEL_KEY", "dump_target")
        save_model_url("SK_MLB_KEY", "dump_target_mlb")


if __name__ == '__main__':
    unittest.main()
