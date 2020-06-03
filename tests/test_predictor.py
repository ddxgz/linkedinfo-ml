import unittest
from unittest import mock
import tempfile

import numpy as np
import pandas as pd
import fasttext

from ml.models import tag
from ml import dataset


class Tester(unittest.TestCase):
    def setUp(self):
        col_text = 'description'
        ds_param = dict(from_batch_cache='info', lan='en',
                        concate_title=True,
                        filter_tags_threshold=30)
        self.ds = dataset.ds_info_tags(**ds_param)

        self.tempd = tempfile.TemporaryDirectory()
        self.fasttestfile = self.tempd.name + '/linkedinfo'
        self.ds.dump_fasttext(self.fasttestfile, split_test=0.3, shuffle=True)

    def tearDown(self):
        self.tempd.cleanup()

    def test_fast_text(self):
        modelfile = self.tempd.name + '/model.bin'
        train_model = fasttext.train_supervised(
            input=self.fasttestfile + '.train', lr=1, epoch=1)
        train_model.save_model(modelfile)
        model = tag.TagsFasttextModel(modelfile)

        k = 2
        pred = model.predict(self.ds.data['title'][0], top_n=k)
        self.assertEqual(k, len(pred))

        k = 5
        pred = model.predict(self.ds.data['title'][0], k=k, threshold=0.2)
        print(pred)
        self.assertGreaterEqual(k, len(pred))


if __name__ == '__main__':
    unittest.main()
