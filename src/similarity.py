import sys, logging
from itertools import combinations
from API_comparison import *

def Sort(sub_li):
    return sorted(sub_li, key = lambda x: x[2], reverse=True)

def process(processed1, processed2):
    threshold = 1

    res = run_files(processed1, processed2, 'complex')
    res['pairs'] = sorted(res['pairs'], key=lambda x: x[2])[:int(len(res['pairs']) * threshold)]
    # fit_result = [['Code 1', 'Code 2', score_, str_]]

    # string_ = Sort(fit_result)

    return res
