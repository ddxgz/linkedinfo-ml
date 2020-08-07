import os
from pathlib import Path

from dotenv import load_dotenv


ENV_PREFIX = 'LINKEDINFO_'

# load_dotenv does not override existing System environment variables.
load_dotenv()

env = os.getenv(f'{ENV_PREFIX}ENV', default='prod')

test_model = False
test_model_env = os.getenv(f'{ENV_PREFIX}TEST_MODEL')
if env != 'prod' and test_model_env is not None and test_model_env not in ('False', 'false', '0'):
    test_model = True

log_level = os.getenv(f'{ENV_PREFIX}LOG_LEVEL', default='error')

model_path = os.getenv(f'{ENV_PREFIX}MODEL_PATH', default='data/models/')
if not os.path.exists(model_path):
    os.makedirs(model_path)

# print(test_model, env, model_path)
