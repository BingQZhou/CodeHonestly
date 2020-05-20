from pre_process import *




def insert_cost(a):
    return 1
def remove_cost(a):
    return 1
def update_cost(a, b):
    #need to revise here
    a_type = get_node_type(a.label)
    b_type = get_node_type(b.label)
    #two nodes are completed the same, including the node type and variable/arguments' names
    if a == b:
        return 0
    #though two nodes' types are the same, the name are different
    elif a_type == b_type:
        return 0.6
    #the positions of two nodes are the same, but the represenations of nodes are different
    return 0.8


def get_smaller(ls):
    data = {}
    N = ls.keys()
    for l in N:
        length = []
        APIs = []
        subs = []
        for i in ls[l]:
            try:
                N, API, sub = get_info(i)
                length.append(N)
                APIs.append(API)
                subs.append(sub)
                
            except:
                continue
        result = [length, APIs, subs]
        data[l] = result
    return data
        
def get_info(sub):
    s = "%d:%s" % (len(sub.children), sub.label)
    s = '\n'.join([s]+[str(c) for c in sub.children])
    API = re.search('func (.+)', s).group(1).split(' ')[-1].lower()
    N = len(s.split('\n'))
    return N, API, sub

def get_label(sub):
    s = "%d:%s" % (len(sub.children), sub.label)
    s = '\n'.join([s]+[str(c) for c in sub.children])
    s = s.split('\n')

    s = [re.search('\d:(.+)', i).group(1) for i in s]
    return s

def get_node_type(node):
    ntype = re.search('(\w+)', node).group(1)
    return ntype

def get_sim_matrix(data_1, data_2):
    methods_1 = data_1.keys()
    methods_2 = data_2.keys()
    ls = []
    temp = {}
    new = {}
    for i in methods_1:
        info_1 = data_1[i]
        temp[i] = []
        new[i] = {}
        for n in methods_2:
            info_2 = data_2[n]
            N1 = len(info_1[0])
            N2 = len(info_2[0])
            matrix = np.full((N1, N2), np.nan)
            new[i][n] = []
            for k in range(N1):
                API_1 = info_1[1][k]
                max_sim = -1
                for j in range(N2):
                    API_2 = info_2[1][j]
                    if API_1 == API_2:
                        dist = zss.distance(info_1[2][k], info_2[2][j], Node.get_children, insert_cost, remove_cost, update_cost).round(3)
                        max_len = max(info_1[0][k], info_2[0][j])
                        sim = (max_len - dist)/max_len
                        matrix[k, j] = sim
                        if sim > max_sim:
                            max_sim = sim
#                         print(info_1[1][k], info_2[1][j], sim, dist, info_1[0][k], info_2[0][j])
                    else:
                        continue
                if max_sim == -1:
                    detail = (info_1[0][k], 0, info_1[1][k], info_2[1][j])
                    matrix[k, N2-1] = 0
                else:
                    detail = (max_len, max_sim, info_1[1][k], info_2[1][j])
                new[i][n].append(detail)
                        
            temp[i].append(matrix)
        
    return temp, new

def get_score(matrix):
    if len(matrix) == 0 or len(matrix[0]) == 0:
        return 0
    mth_1_len = len(matrix)
    mth_2_len = len(matrix[0])
    sum_ = 0
    
    for i in matrix:
        temp_ = [j for j in i if j > 0]
        if len(temp_) != 0:
            
            temp_ = max(temp_)
            sum_ = sum_ + temp_
        else:
            sum_ = sum_ + 0
    return sum_ / mth_1_len

def find_peak(mat):
    x = 0
    y = 0
    max_ = -1000
    for i in range(len(mat)):
        for j in range(len(mat[0])):
            if mat[i][j]>max_:
                x = i
                y = j
                max_ = mat[i][j]
    return max_, x, y


def run_files (file1, file2, type_):
    with open(file1) as f_1:
        full_lines_1 = ''
        for i in f_1.readlines():
            full_lines_1+=i
    # for ur own reference
    
    with open(file2) as f_2:
        full_lines_2 = ''
        for i in f_2.readlines():
            full_lines_2+=i
    data_1 = json.loads(process(full_lines_1))
    data_2 = json.loads(process(full_lines_2))


    body_2 = get_body(data_2)
    result_2 = create_func_dict(body_2[1:])

    body_1 = get_body(data_1)
    result_1 = create_func_dict(body_1[1:])
    #print(result_1)

    data_1 = get_smaller(result_1)
    data_2 = get_smaller(result_2)
    #print(data_1)

    temp, new = get_sim_matrix(data_1, data_2)
    
    methods_1 = data_1.keys()
    methods_2 = data_2.keys()
    
    if min(len(methods_1), len(methods_2)) == len(methods_1):
        matrix = np.full((len(methods_1), len(methods_2)), np.nan)
    else: 
        matrix = np.full((len(methods_2), len(methods_1)), np.nan)


    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            score = get_score(temp[list(data_1.keys())[i]][j])
            matrix[i][j] = score

    #print(matrix)


    # TODO: consider different len of method - do it when building the matrix
    final_pair = []
    for method_A in range(len(matrix)):
        #print(matrix)
        curr_peak, x, y = find_peak(matrix)
        #print(curr_peak)
        for i in range(len( matrix)):
            if i == x:
                #print(i, method)
                for j in range(len(matrix[i])):
                    matrix[i][j] = -1000
            matrix[i][y] = -1000
        final_pair = final_pair + [[curr_peak, (x, y)]]
        #print('-----***********************************-----')
        #print([curr_peak, (x, y)])
    #print('----------------------------------------------')
    #print(final_pair)
    str_ = 'Similarity result: ' + '\n '
    #print('Similiarity result: ')
    if type_ == 'complex':
        for i in final_pair:
            str_ = str_ + 'Method: ' + str(list(data_1.keys())[i[1][0]]) +  ' ------ ' + \
                  str(list(data_2.keys())[i[1][1]]) + ' with similarity: ' + str(i[0]) + '\n '
            #print('Mythod: ',list(data_1.keys())[i[1][0]], ' ------ ', 
             #     list(data_2.keys())[i[1][1]], ' with similiarity: ', i[0])
    score_list = []
    all_nodes = 0
    for i in range (len(final_pair)):
        #str_  = str_  + str(final_pair[i][0]) + str(len(data_1[list(data_1.keys())[i]][0])) + '\n '
        #print(final_pair[i][0], len(data_1[list(data_1.keys())[i]][0]))
        score_list = score_list + [final_pair[i][0]*(len(data_1[list(data_1.keys())[i]][0]))]
        all_nodes += len(data_1[list(data_1.keys())[i]][0])
    score_ = sum(score_list)/all_nodes
    str_ = str_ + 'Overall Similarity Score: ' + str(score_) + '\n '
    #print('Overall Similiarity Score: ', score_ )
    return score_, str_