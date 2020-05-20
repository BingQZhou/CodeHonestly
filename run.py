import sys
import json
import shutil
import subprocess
import warnings

sys.path.insert(0, './src')
#import py_file
from API_comparison import *
warnings.filterwarnings("ignore")

sys.path.insert(0, 'src')
#from etl import download_apk
from model import run

TEST_DATA_PARAMS = 'config/test-data-params.json'
TRAIN_DATA_PARAMS = 'config/data-params.json'
TEST_PROJECT_PARAMS = "config/test-project-params.json"



def load_params(fp):
    with open(fp) as fh:
        param = json.load(fh)
    return param



def Sort(sub_li): 
  
    return(sorted(sub_li, key = lambda x: x[2], reverse=True))  

def main(targets):

    if 'clean' in targets:
        shutil.rmtree('result', ignore_errors=True)

    if 'test-project' in targets:
        # Run and then generate a fake similarity comparison according to function call(pycode_similar)
        #cfg = load_params(TEST_PROJECT_PARAMS)
        #run(**cfg)
        cfg = load_params(TEST_PROJECT_PARAMS)
        output_file = cfg['output_file']
        data_dir = cfg['data_dir']
        output_dir = cfg['output_dir'] 
        thre = cfg['threshold']
        output_type = cfg['output_type'] # save or print out or [save and print out]
        output_mode = cfg['output_mode'] # simple or complex



        onlyfiles = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]
        all_py = [f for f in onlyfiles if f[-3:] == '.py']
        #print(all_py)
        #thre = 0.3
        fit_result = []
        max_ = 0
        max_info = ''
        file_1 = ''
        file_2 = ''

        comb = combinations(all_py, 2) 
        list_ = list(comb)
        for i in range(len(list_)):
            #print(list_)
            f1 = list_[i][0]
            f2 = list_[i][1]
            score_ , str_ = run_files(join(data_dir,f1), join(data_dir,f2), output_mode)
            fit_result = fit_result + [[list_[i][0], list_[i][1], score_, str_]]

        string_ = Sort(fit_result)

        top_k = []
        first_k = len(string_)*thre
        for i in range(len(string_)):
            if i <= first_k:
                top_k = top_k + [string_[i]]
            else:
                break
        if 'save' in output_type:
            os.mkdir(output_dir) 
            f = open(join(output_dir, output_file), "w")
            for i in top_k:
                for j in i:
                    f.write(str(j) + '\n')
            f.close()
            print("Result saved to folder: Result")
        if 'print' in output_type:
            for i in top_k:
                for j in i:
                    print(j)
        



if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)
