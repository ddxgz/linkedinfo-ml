import os
import zipfile
import json

from google.cloud import storage

from ml.models.files import ALL_MODELS


def save_model_url(key: str, model_file: str, location: str = 'local'):
    """model file path should be abolute path"""
    location_file = 'model_location.json'

    if os.path.exists(location_file):
        with open(location_file, 'r') as f:
            model_location = json.load(f)
    else:
        model_location = {}

    loc = model_location.get(location, {})
    loc[key] = model_file
    model_location[location] = loc

    with open(location_file, 'w') as f:
        json.dump(model_location, f)


def local_models_to_json():
    for k, v in ALL_MODELS.items():
        save_model_url(k, v, location='local')


def upload_model_bin(key, model_file: str):
    client = storage.Client()
    bucket_name = 'data-science-258408-models'
    # bucket_name = 'tag-models'
    # bucket = client.create_bucket(bucket_name)
    bucket = client.bucket(bucket_name)

    dest = model_file.split('/')[-1]
    blob = bucket.blob(dest)

    blob.upload_from_filename(model_file)

    save_model_url(key, f'{bucket.name}/{dest}', location='gcloud')


def upload_models():
    location_file = 'model_location.json'

    if os.path.exists(location_file):
        with open(location_file, 'r') as f:
            model_location = json.load(f)

        local = model_location.get('local', None)
        if local is None:
            return

        for k, v in local.items():
            if os.path.isdir(v):
                zipname = f'{v.rstrip("/")}.zip'
                with zipfile.ZipFile(zipname, 'w') as zipObj:
                    for folder, subfolders, files in os.walk(v):
                        for fname in files:
                            filePath = os.path.join(folder, fname)
                            zipObj.write(filePath, os.path.basename(filePath))
                v = zipname
            upload_model_bin(k, v)


def fake_upload():
    print('fake uplaod')
