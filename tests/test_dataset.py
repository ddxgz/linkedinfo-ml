import sys
import os
import unittest
from unittest import mock
import tempfile

import numpy as np
import pandas as pd

from ml import dataset


class Tester(unittest.TestCase):
    # def tearDown(self):
    #     os.remove('tests/tmp')

    def test_dataset_dump(self):
        col_text = 'description'
        ds_param = dict(from_batch_cache='info', lan='en',
                        concate_title=True,
                        filter_tags_threshold=20)
        ds = dataset.ds_info_tags(**ds_param)
        ds.get_train_test(0.3)

        with tempfile.TemporaryDirectory() as tmpdirname:
            ver_name = 'test'
            ds.dump_dir = tmpdirname
            ds.dump(version=ver_name)

            output_dir = os.path.join(ds.dump_dir, ver_name)
            print(output_dir)

            self.assertTrue(os.path.exists(output_dir))

            train_features_path = os.path.join(output_dir, 'train.csv')
            train_features_loaded = pd.read_csv(train_features_path)
            self.assertEqual(train_features_loaded.shape,
                             ds.train_features.shape)

            train_targets_path = os.path.join(output_dir, 'train_targets.npy')
            train_targets_loaded = np.load(train_targets_path)
            self.assertEqual(train_targets_loaded.shape,
                             ds.train_targets.shape)


if __name__ == '__main__':
    unittest.main()
