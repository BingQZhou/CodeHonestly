from zss import Node
import json

def get_body(treedict, parent=None):
    name = iter(treedict.keys())
    body = ''
    for i in name:
        if i == 'graph':
            body = get_body(treedict[i])
        if i == 'body':
            body = treedict[i]
    # now we should have independent trees in body variable
    if not isinstance(body, list):
        # try to fix the problem by wrapping it to a list......
        return list(body)
    return body
def convert_body(body,parent_node = None, root_node = None):
    body = seperate_dict(body)
    if isinstance(body, dict):
        if parent_node == None:
            parent_node = Node(body['_PyType'])
            new_parent = parent_node
            root_node = parent_node
        for j in body:
            if j!= '_PyType':
                # still have a kid, then recursion needed
                if isinstance(body[j],dict):
                    if '_PyType' in body[j].keys():
                        if 'attr' in body[j].keys():
                            node_content = j+' '+body[j]['_PyType']+' '+body[j]['attr']
                            new_parent = Node(node_content)
                            parent_node.addkid(new_parent)
                            new_parent = convert_body(body[j],parent_node = new_parent,root_node = root_node)
                        else:
                            call_call_func_name = ''
                            if j == 'func':
                                call_call_func_name =' '+body[j]['id']
                            node_content = j+' '+body[j]['_PyType']+call_call_func_name
                            new_parent = Node(node_content)
                            parent_node.addkid(new_parent)
                            new_parent = convert_body(body[j],parent_node = new_parent,root_node = root_node)
                    else:
                        # case when it's a dict but not with PyType
                        if 'udv' in json.dumps(body[j]):
                            node_content = j+' '+'udv'
                        else:
                            node_content = j+' '+json.dumps(body[j])
                        parent_node = parent_node.addkid(Node(node_content))

                elif isinstance(body[j],list) or isinstance(body[j],str):
                    if body[j]:
                        node_content = j+' '+body[j]
                    else:
                        node_content = j+' '+''
                    parent_node = parent_node.addkid(Node(node_content))
    return root_node
def seperate_dict(dic):
    dic = dic.copy()
    for i in dic.copy():
        if isinstance(dic[i],list):
            if dic[i]:
                count= 0
                for j in dic[i]:
                    count+=1
                    dic[i+str(count)] = j
            del dic[i]
    return dic
def create_func_dict(body):
    func_dict={}
    for i in body:
        func_nodes_list = []
        func_name = ''
        body_i = get_body(i)
        if i['_PyType'] == 'FunctionDef':
            func_name = i['name']
            if body_i and isinstance(body_i,list):
                for j in body_i:
                    func_nodes_list.append(convert_body(j))
        else:
            #print('Not all codes are wrapped into functions, please do so.')
            continue
        func_dict[func_name] = func_nodes_list
    return func_dict
