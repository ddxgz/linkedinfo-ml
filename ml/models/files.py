import os
import zipfile
import json

from google.cloud import storage

PRETRAINED_BERT_WEIGHTS = "data/models/bert_mini_finetuned_tagthr_20/"

MODEL_FILE = 'data/models/tags_textbased_pred_10.joblib.gz'
MLB_FILE = 'data/models/tags_textbased_pred_10_mlb.joblib.gz'
FAST_TEXT_MODEL_FILE = 'data/models/fasttext_thr10_v1_2.bin'


SK_MODEL_KEY = 'sk_clf'
SK_MLB_KEY = 'sk_mlb'
FT_MODEL_KEY = 'fasttext'
BERT_PRETRAINED = 'bert_pretrained'
DS_DATA_APP = 'ds_dataapp'


ALL_MODELS = {
    SK_MODEL_KEY: MODEL_FILE,
    SK_MLB_KEY: MLB_FILE,
    FT_MODEL_KEY: FAST_TEXT_MODEL_FILE,
    BERT_PRETRAINED: PRETRAINED_BERT_WEIGHTS,
    DS_DATA_APP: 'data/pickle/dataappset.pkl',
}


def download_model_bin(key, model_file):
    client = storage.Client()
    bucket_name = 'tag-models'
    # bucket = client.create_bucket(bucket_name)
    bucket = client.bucket(bucket_name)

    src = model_file.split('/')[-1]
    blob = bucket.blob(src)
    dest = f'data/models/{model_file.split("/")[-1]}'
    # print(dest)
    if not os.path.exists('data/models/'):
        os.makedirs('data/models')
    if not os.path.exists('data/pickle/'):
        os.makedirs('data/pickle/')
    blob.download_to_filename(dest)
    return dest


def download_models(location='gcloud'):
    location_file = 'model_location.json'

    if os.path.exists(location_file):
        with open(location_file, 'r') as f:
            model_location = json.load(f)

        local = model_location.get(location, None)
        if local is None:
            return

        for k, v in local.items():
            if v[-1] == '/':
                zipname = f'{v.rstrip("/")}.zip'
                v = zipname

            dest_file = download_model_bin(k, v)

            if dest_file.endswith('.zip'):
                with zipfile.ZipFile(dest_file, 'r') as zipObj:
                    zipObj.extractall(dest_file.rstrip('.zip') + '/')


def model_file(model_type: str) -> str:
    if not os.path.exists('data/models/'):
        download_models()

    return ALL_MODELS.get(model_type, '')


def init_model_files():
    if not os.path.exists('data/models/') or not os.path.exists('data/pickle/'):
        download_models()
