import os
from pathlib import Path

from dotenv import load_dotenv

# load_dotenv does not override existing System environment variables.
load_dotenv()

env = os.getenv('LINKEDINFO_ENV')
if env is None:
    env = 'prod'

test_model = False
test_model_env = os.getenv('LINKEDINFO_TEST_MODEL')
if test_model_env is not None and test_model_env not in ('False', 'false', '0'):
    test_model = True

log_level = os.getenv('LINKEDINFO_LOG_LEVEL')
if log_level is None:
    log_level = 'error'

# print(test_model)