import sys, logging
from itertools import combinations
from API_comparison import *

def Sort(sub_li):
    return sorted(sub_li, key = lambda x: x[2], reverse=True)

def process(processed1, processed2):
    threshold = 1

    score_, str_ = run_files(processed1, processed2, 'complex')
    fit_result = [['Code 1', 'Code 2', score_, str_]]

    string_ = Sort(fit_result)

    first_k = int(len(string_) * threshold)
    top_k = string_[:first_k]

    return top_k
