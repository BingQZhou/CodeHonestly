import sys
import json
import shutil
import subprocess
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, 'src')
#from etl import download_apk
from model import run

TEST_DATA_PARAMS = 'config/test-data-params.json'
TRAIN_DATA_PARAMS = 'config/data-params.json'
TEST_PROJECT_PARAMS = 'config/test-project-params.json'


def load_params(fp):
    with open(fp) as fh:
        param = json.load(fh)
    return param


def main(targets):

    if 'clean' in targets:
        shutil.rmtree('result', ignore_errors=True)

    if 'test' in targets:
        # Run and then generate a fake similarity comparison according to function call(pycode_similar)
        cfg = load_params(TEST_PROJECT_PARAMS)
        run(**cfg)
if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)
