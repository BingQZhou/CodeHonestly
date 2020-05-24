from __future__ import print_function
from _ast import AST
import ast, json, pprint, sys, re, numpy as np, zss, os
from zss import simple_distance, Node
from os import listdir
from os.path import isfile, join
from itertools import combinations

def process(code):
    node = ast.parse(code)
    jsoned = ast2json(node)
    as_obj = json.loads(jsoned)
    # manipulation
    cpy = dict(as_obj)
    imports, cpy = preprocess_import_and_call_statements(cpy)
    remove_unnecessary(cpy)
    # remove_insignificant_variable(cpy)
    unchain(cpy, cpy['body'])
    remove_excessive_call(cpy)
    return json.dumps({
        'imports': imports,
        'graph': cpy
    })

unnecessary_set = frozenset(['ctx', 'kwargs', 'starargs'])
def remove_unnecessary(cpy):
    queue = cpy['body'][:]
    while len(queue):
        polled = queue.pop(0)
        for key in polled.copy():
            if key in unnecessary_set:
                del polled[key]
        queue.extend([polled[key] for key in polled.keys() if isinstance(polled[key], dict)])
        for key in polled.keys():
            if isinstance(polled[key], list):
                queue.extend(polled[key])
# a = default tree
# a['body'] = uchain_body_calls['body']
def unchain_body_calls(body):
    cpy = list(body)
    to_delete = []
    for i, ele in enumerate(body):
        did_unchain = unchain(ele, cpy)
        if did_unchain:
            to_delete.append(i)
    for x in to_delete[::-1]:
        cpy.pop(x)
    return cpy
def unchain(node, body):
    if should_unchain(node):
        unchain_helper(node, body)
        return True
    for key, item in node.items():
        if isinstance(item, list) and key == 'body':
            node['body'] = unchain_body_calls(item)
    return False

def should_unchain(node):
    if not isinstance(node, dict):
        return False
    if node['_PyType'] == 'Call':
        return True
    else:
        return any([should_unchain(item) for key, item in node.items() if isinstance(item, dict)])
def unchain_helper(node, body):
    stack = [node]
    call_stack = []

    while len(stack):
        popped = stack.pop(-1)
        if isinstance(popped, dict):
            if popped['_PyType'] == 'Call':
                call_stack.append(popped)

            stack.extend([item for key, item in popped.items() if isinstance(item, dict)])
            for key, item in popped.items():
                if isinstance(item, list):
                    stack.extend(item)
    for i in range(0, len(call_stack) - 1):
        call_stack[i] = remove_subtree(call_stack[i])
    body.extend(call_stack[::-1])
def remove_subtree(tree, flag=False):
    if not isinstance(tree, dict):
        return tree
    output = {}

    for key, item in tree.items():
        if tree['_PyType'] == 'Call':
            if not flag:
                output[key] = remove_subtree(item, True)
        else:
            output[key] = remove_subtree(item, flag)

            # Modified here. Added try
            try:
                if len(output[key]) == 0 and key == 'value':
                    output[key] = {
                        'id': None,
                        'type': 'udv'
                    }
            except:
                #print(output[key])
                pass
                #print('output[key] has a problem, ignoring it')
    return output
def preprocess_import_and_call_statements(graph):
    built_ins = frozenset(['abs', 'all', 'any', 'ascii', 'bin', 'bool', 'breakpoint', 'bytearray', 'bytes', 'callable', 'chr', 'classmethod', 'compile', 'complex', 'delattr', 'dict', 'dir', 'divmod', 'enumerate', 'eval', 'exec', 'filter', 'float', 'format', 'frozenset', 'getattr', 'globals', 'hasattr', 'hash', 'help', 'hex', 'id', 'input', 'int', 'isinstance', 'issubclass', 'iter', 'len', 'list', 'locals', 'map', 'max', 'memoryview', 'min', 'next', 'object', 'oct', 'open', 'ord', 'pow', 'print', 'property', 'range', 'repr', 'reversed', 'round', 'set', 'setattr', 'slice', 'sorted', 'staticmethod', 'str', 'sum', 'super', 'tuple', 'type', 'vars', 'zip', '__import__'])
    udfs = set()
    # bfs through graph
    queue = graph['body'][:]
    imports = {}
    while len(queue):
        polled = queue.pop(0)

        if polled['_PyType'] in ['Import', 'ImportFrom']:
            for name in polled['names']:
                if name['asname'] is not None:
                    imports[name['asname']] = name['name']
                else:
                    imports[name['name']] = name['name']
        if polled['_PyType'] == 'FunctionDef':
            udfs.add(polled['name'])

        queue.extend([polled[key] for key in polled.keys() if isinstance(polled[key], dict)])
        for key in polled.keys():
            if isinstance(polled[key], list):
                queue.extend(polled[key])
    queue = graph['body'][:]
    while len(queue):
        polled = queue.pop(0)

        if isinstance(polled, dict):
            if polled['_PyType'] == 'Name':
                if polled['id'] in imports:
                    polled['id'] = imports[polled['id']]
                elif polled['id'] in udfs:
                    polled['id'] = polled['id']
                    polled['type'] = 'udf'
                elif polled['id'] in built_ins:
                    polled['id'] = polled['id']
                    polled['type'] = 'built_in'
                elif polled['id'] not in built_ins and polled['id'] not in udfs:
                    polled['type'] = 'udv'

        queue.extend([polled[key] for key in polled.keys() if isinstance(polled[key], dict) and key != 'id'])
        for key in polled.keys():
            if isinstance(polled[key], list):
                queue.extend(polled[key])

    return imports, graph

def remove_insignificant_variable(cpy):
    queue = cpy['body'][:]
    # add package name to pakcage_list
    package_list = ["F"]
    while len(queue):
        polled = queue.pop(0)

        if 'value' in polled.keys():
            if 'Name' in polled['value']["_PyType"]:
                if polled['value']['id'] not in package_list:
                    print("Deleting "+str(polled['value']['id']))
                    polled['value']['id'] = 'dummy_value'

        queue.extend([polled[key] for key in polled.keys() if isinstance(polled[key], dict)])
        for key in polled.keys():
            if isinstance(polled[key], list):
                queue.extend(polled[key])

def remove_excessive_call(cpy):
    #Todo
    pass
def ast2json( node ):

    if not isinstance( node, AST ):
        raise TypeError( 'expected AST, got %r' % node.__class__.__name__ )
    def _format( node ):
        if isinstance( node, AST ):
            fields = [ ( '_PyType', _format( node.__class__.__name__ ) ) ]
            fields += [ ( a, _format( b ) ) for a, b in iter_fields( node ) ]

            return '{ %s }' % ', '.join( ( '"%s": %s' % field for field in fields ) )

        if isinstance( node, list ):
            return '[ %s ]' % ', '.join( [ _format( x ) for x in node ] )

        return json.dumps( node )
    return _format( node )

def iter_fields( node ):

    for field in node._fields:
        try:
            yield field, getattr( node, field )
        except AttributeError:
            pass


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
