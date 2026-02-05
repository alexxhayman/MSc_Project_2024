"""Helper to safely import functions from scripts with unguarded top-level code.

The project scripts contain top-level code (file reads, API calls) interspersed
with function definitions. Standard import would execute that code and fail.
This helper uses AST parsing to extract and execute only function/import nodes.
"""

import ast
import types


def import_functions_from_script(script_path, function_names):
    """Extract specific functions from a script without running top-level code.

    Parses the script's AST and executes only:
    - import statements (so function dependencies are available)
    - function definitions listed in function_names

    Returns a dict mapping function_name -> function_object.
    """
    with open(script_path, 'r') as f:
        source = f.read()

    tree = ast.parse(source)

    # Build a new module with only imports and requested function defs
    keep_nodes = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            keep_nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in function_names:
            keep_nodes.append(node)

    new_tree = ast.Module(body=keep_nodes, type_ignores=[])
    ast.fix_missing_locations(new_tree)

    code = compile(new_tree, script_path, 'exec')
    namespace = {}
    exec(code, namespace)

    return {name: namespace[name] for name in function_names if name in namespace}
