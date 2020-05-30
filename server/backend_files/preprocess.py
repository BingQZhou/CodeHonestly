import ast, json
from _ast import AST

def process(src, normalize=True, ctx=False):
    node = ast.parse(src)
    jsoned = ast2json(node)
    as_obj = json.loads(jsoned)
    # manipulation
    cpy = dict(as_obj)
    imports, cpy = preprocess_import_and_call_statements(cpy)
    if not ctx:
        remove_unnecessary(cpy)
    # remove_insignificant_variable(cpy)
    if normalize:
        unchain(cpy, cpy['body'])
    remove_excessive_call(cpy)
    return {
        'imports': imports,
        'graph': cpy
    }

unnecessary_set = frozenset(['ctx', 'kwargs', 'starargs'])
def remove_unnecessary(cpy):
    queue = cpy['body'][:]
    while len(queue):
        polled = queue.pop(0)
        if isinstance(polled, dict):
            for key in unnecessary_set:
                if key in polled:
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
        if isinstance(item, list) and key in ['body', 'args']:
            node[key] = unchain_body_calls(item)
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
    # need to handle lists too like nested calls in args
    # handle below instead of returning just tree of not dict check for list
    if isinstance(tree, list):
        output = [remove_subtree(element, flag) for element in tree]
        empty_idxs = [i for i, item in enumerate(tree) if len(item) == 0]
        for idx in empty_idxs[::-1]:
            output.pop(idx)
        return output if len(output) else None
    if not isinstance(tree, dict):
        return tree
    output = {}

    for key, item in tree.items():
        if tree['_PyType'] == 'Call':
            if not flag:
                output[key] = remove_subtree(item, True)
                if output[key] is None:
                    del output[key]
        else:
            output[key] = remove_subtree(item, flag)
            if isinstance(output[key], dict) and len(output[key]) == 0 and key == 'value':
                output[key] = {
                    'id': None,
                    'type': 'udv'
                }
            if output[key] is None:
                del output[key]
    return output
def preprocess_import_and_call_statements(graph):
    built_ins = frozenset(['abs', 'all', 'any', 'ascii', 'bin', 'bool', 'breakpoint', 'bytearray', 'bytes', 'callable', 'chr', 'classmethod', 'compile', 'complex', 'delattr', 'dict', 'dir', 'divmod', 'enumerate', 'eval', 'exec', 'filter', 'float', 'format', 'frozenset', 'getattr', 'globals', 'hasattr', 'hash', 'help', 'hex', 'id', 'input', 'int', 'isinstance', 'issubclass', 'iter', 'len', 'list', 'locals', 'map', 'max', 'memoryview', 'min', 'next', 'object', 'oct', 'open', 'ord', 'pow', 'print', 'property', 'range', 'repr', 'reversed', 'round', 'set', 'setattr', 'slice', 'sorted', 'staticmethod', 'str', 'sum', 'super', 'tuple', 'type', 'vars', 'zip', '__import__'])
    udfs = set()
    # bfs through graph
    queue = graph['body'][:]
    imports = {}
    while len(queue):
        polled = queue.pop(0)

        if isinstance(polled, dict):
            if polled['_PyType'] == 'Import':
                for name in polled['names']:
                    if name['asname'] is not None:
                        imports[name['asname']] = name['name']
                    else:
                        imports[name['name']] = name['name']
            elif polled['_PyType'] == 'ImportFrom':
                for name in polled['names']:
                    if name['asname'] is not None:
                        imports[name['asname']] = polled['module'] + '.' + name['name']
                    else:
                        imports[name['name']] = polled['module'] + '.' + name['name']
            elif polled['_PyType'] == 'FunctionDef':
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
