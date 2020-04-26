from flask import Flask, request
from flask import render_template
from _ast import AST
import ast
import json

application = Flask(__name__)

@application.route('/')
def index():
    return render_template('index.html')

@application.route('/ast2json', methods=['POST'])
def process():
    node = ast.parse(request.form.get('pysrc'))
    jsoned = ast2json(node)
    as_obj = json.loads(jsoned)
    # manipulation
    cpy = dict(as_obj)
    imports, cpy = preprocess_import_and_call_statements(cpy)
    if request.form.get('ctx') == 'false':
        remove_unnecessary(cpy)
    # remove_insignificant_variable(cpy)
    if request.form.get('normalize') == 'true':
        unchain_call_parent(cpy)
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
        for key in polled.keys():
            if key in unnecessary_set:
                del polled[key]
        queue.extend([polled[key] for key in polled.keys() if isinstance(polled[key], dict)])
        for key in polled.keys():
            if isinstance(polled[key], list):
                queue.extend(polled[key])
def unchain_call_parent(cpy, flag=False):
    calls = unchain_call(cpy, flag=flag)
    cpy['body'] = calls
    return cpy

def unchain_call(cpy, prev=None, flag=False):
    calls = []
    if isinstance(cpy, dict) and cpy['_PyType'] in ['If'] and not flag:
        cpy = unchain_call_parent(cpy, True)
        return [cpy]
    for key, item in cpy.items():
        if isinstance(item, list):
            for ele in item:
                if should_unchain(ele) or cpy['_PyType'] == 'Assign':
                    calls.extend(unchain_call(ele, prev=cpy['_PyType']))
                else:
                    calls.append(ele)
        elif isinstance(item, dict):
            if should_unchain(item) or cpy['_PyType'] == 'Assign':
                if item['_PyType'] == 'Call' and (cpy['_PyType'] in ['Call', 'Expr'] or (cpy['_PyType'] == 'Attribute' and prev == 'Call')):
                    calls.extend(unchain_helper(item))
                else:
                    calls.extend(unchain_call(item, prev=cpy['_PyType']))
            else:
                calls.append(item)
    return calls
def should_unchain(node):
    if not isinstance(node, dict):
        return False
    if node['_PyType'] == 'Call':
        return True
    else:
        return any([should_unchain(item) for key, item in node.items() if isinstance(item, dict)] + [any((should_unchain(ele) for ele in item)) for key, item in node.items() if isinstance(item, list)])
def unchain_helper(node):
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
    return call_stack[::-1]
def preprocess_import_and_call_statements(graph):
    built_ins = frozenset(['abs', 'all', 'any', 'ascii', 'bin', 'bool', 'breakpoint', 'bytearray', 'bytes', 'callable', 'chr', 'classmethod', 'compile', 'complex', 'delattr', 'dict', 'dir', 'divmod', 'enumerate', 'eval', 'exec', 'filter', 'float', 'format', 'frozenset', 'getattr', 'globals', 'hasattr', 'hash', 'help', 'hex', 'id', 'input', 'int', 'isinstance', 'issubclass', 'iter', 'len', 'list', 'locals', 'map', 'max', 'memoryview', 'min', 'next', 'object', 'oct', 'open', 'ord', 'pow', 'print', 'property', 'range', 'repr', 'reversed', 'round', 'set', 'setattr', 'slice', 'sorted', 'staticmethod', 'str', 'sum', 'super', 'tuple', 'type', 'vars', 'zip', '__import__'])
    udfs = set()
    # bfs through graph
    queue = graph['body'][:]
    imports = {}
    while len(queue):
        polled = queue.pop(0)

        if polled['_PyType'] == 'Import':
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

# This is a simplified preprocessing function used to demonstrate validity in local
def run_local(path):
    f = open(path,'r')
    node = ast.parse(f.read())
    jsoned = ast2json(node)
    as_obj = json.loads(jsoned)
    # manipulation
    cpy = dict(as_obj)
    imports, cpy = preprocess_import_and_call_statements(cpy)
    # remove_insignificant_variable(cpy)
    remove_excessive_call(cpy)
    parsed_json = json.dumps({
        'imports': imports,
        'graph': cpy
    })
    with open('../ast_in_json.json', 'w') as f:
        json.dump(parsed_json,f, indent=2, sort_keys=True)
    return

if __name__ == '__main__':
    application.run(debug=True, host='0.0.0.0')
