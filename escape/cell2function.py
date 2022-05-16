# -*- coding: utf-8 -*-

import ast
import sys
import types
PY3 = sys.version_info[0] >= 3

try:
    import builtins  # Python 3
except ImportError:
    import __builtin__ as builtins  # Python 2

class NameScanner(ast.NodeVisitor):
    """Check an AST for names defined in it, and names read before they are defined."""
    def __init__(self, user_ns):
        super(NameScanner, self).__init__()
        self.user_ns = user_ns
        self.read_before_defined = []
        self.defined_in_scopes = [list(builtins.__dict__), []]
    
    def visible(self, name):
        return any(name in defined for defined in self.defined_in_scopes)

    def is_module(self, name):
        """If 'name' is a module in user_ns, we don't want it to be a parameter."""
        return isinstance(self.user_ns.get(name), types.ModuleType)

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Store):
            self.defined_in_scopes[-1].append(node.id)
        elif isinstance(node.ctx, ast.Load) and not self.visible(node.id) \
                and not self.is_module(node.id):
            self.read_before_defined.append(node.id)
    
    def visit_Assign(self, node):
        """Reimplemented to ensure that we visit value before targets.
        So 'a = a + 1' counts as reading then writing a."""
        self.visit(node.value)
        for target in node.targets:
            self.visit(target)
    
    def visit_AugAssign(self, node):
        """Treat 'a += 1' as both reading and writing a."""
        if isinstance(node.target, ast.Name):
            if not self.visible(node.target.id):
                self.read_before_defined.append(node.target.id)
            self.defined_in_scopes[-1].append(node.target.id)
            self.visit(node.value)
        else:
            # Augmented assignment on attribute/subscript/slice
            self.generic_visit(node)
    
    def visit_FunctionDef(self, node):
        pn = ParameterNames()
        pn.visit(node.args)
        self.defined_in_scopes.append(pn.paramnames)
        self.generic_visit(node)
        self.defined_in_scopes.pop()
        self.defined_in_scopes[-1].append(node.name)
        # XXX: For now, assume functions aren't declaring any global/nonlocal
        # variables

class ParameterNames(ast.NodeVisitor):
    def __init__(self):
        super(ParameterNames, self).__init__()
        self.paramnames = []
    
    # Parameters are stored differently in Python 2 and 3
    if PY3:
        def visit_arg(self, node):
            self.paramnames.append(node.arg)
    else:
        def visit_Name(self, node):
            if isinstance(node.ctx, ast.Param):
                self.paramnames.append(node.id)

# Copied from IPython.utils.data
def uniq_stable(elems):
    """uniq_stable(elems) -> list

    Return from an iterable, a list of all the unique elements in the input,
    but maintaining the order in which they first appear.

    Note: All elements in the input must be hashable for this routine
    to work, as it internally uses a set for efficiency reasons.
    """
    seen = set()
    return [x for x in elems if x not in seen and not seen.add(x)]

def makefunction(name, cell, user_ns):
    namescanner = NameScanner(user_ns=user_ns)
    namescanner.visit(ast.parse(cell))
    args = ", ".join(uniq_stable(namescanner.read_before_defined))
    out = ["def {name}({args}):".format(name=name, args=args)]
    out.extend("    " + l for l in cell.splitlines())
    outnames = ", ".join(uniq_stable(namescanner.defined_in_scopes[1]))
    out.append("    return " + outnames)
    out.append("")
    out.append("{outnames} = {name}({args})".format(outnames=outnames,
                                   name=name, args=args))
    return "\n".join(out)

def load_ipython_extension(ip):
    def cell2function(line, cell):
        func_name = line.strip().split()[0] if line.strip() else 'unnamed'
        ip.set_next_input(makefunction(func_name, cell, ip.user_ns))
    
    ip.register_magic_function(cell2function, magic_kind='cell')