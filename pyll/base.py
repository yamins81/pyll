# file is called AST to not collide with std lib module 'ast'
#
# It provides types to build ASTs in a simple lambda-notation style
#

import copy
import logging; logger = logging.getLogger(__name__)
import time
import itertools

from StringIO import StringIO
from collections import deque

import networkx as nx

# TODO: move things depending on numpy (among others too) to a library file
import numpy as np
np_versions = map(int, np.__version__.split('.')[:2])


DEFAULT_MAX_PROGRAM_LEN = 100000


class PyllImportError(ImportError):
    """A pyll symbol was not defined in the scope """


class MissingArgument(object):
    """Object to represent a missing argument to a function application
    """


class SymbolTable(object):
    """
    An object whose methods generally allocate Apply nodes.

    _impls is a dictionary containing implementations for those nodes.

    >>> self.add(a, b)          # -- creates a new 'add' Apply node
    >>> self._impl['add'](a, b) # -- this computes a + b
    """

    def __init__(self):
        # -- list and dict are special because they are Python builtins
        self._impls = {
                'list': list,
                'dict': dict,
                'range': range,
                'len': len,
                'int': int,
                'float': float,
                'map': map,
                'max': max,
                'min': min,
                'getattr': getattr,
                }

    def _new_apply(self, name, args, kwargs, o_len, pure, 
                            learn_args=(), fit_class=None, fit_kwargs=None,
                            _label=None, compiler=None, compiler_data=None
                            ):
        pos_args = [as_apply(a) for a in args]
        named_args = [(k, as_apply(v)) for (k, v) in kwargs.items()]
        named_args.sort()
        return Apply(name,
                pos_args=pos_args,
                named_args=named_args,
                o_len=o_len,
                pure=pure,
                learn_args=learn_args,
                fit_class=fit_class,
                fit_kwargs=fit_kwargs,
                _label=_label,
                compiler=compiler,
                compiler_data=compiler_data)

    # ----

    def dict(self, *args, **kwargs):
        # XXX: figure out len
        return self._new_apply('dict', args, kwargs, o_len=None,
                pure=True)

    def int(self, arg):
        return self._new_apply('int', [as_apply(arg)], {}, o_len=None,
                pure=True)

    def float(self, arg):
        return self._new_apply('float', [as_apply(arg)], {}, o_len=None,
                pure=True)

    def len(self, obj):
        return self._new_apply('len', [obj], {}, o_len=None,
                pure=True)

    def list(self, init):
        return self._new_apply('list', [as_apply(init)], {}, o_len=None,
                pure=True)

    def map(self, fn, seq, pure=False):
        """
        pure - True is assertion that fn does not modify seq[i]
        """
        return self._new_apply('map', [as_apply(fn), as_apply(seq)], {},
                               o_len=seq.o_len,
                               pure=pure
                               )

    def range(self, *args):
        return self._new_apply('range', args, {}, o_len=None, pure=True)

    def max(self, *args):
        """ return max of args """
        return self._new_apply('max', map(as_apply, args), {},
                o_len=None, pure=True)

    def min(self, *args):
        """ return min of args """
        return self._new_apply('min', map(as_apply, args), {},
                o_len=None, pure=True)

    def getattr(self, obj, attr, *args):
        return self._new_apply('getattr',
                [as_apply(obj), as_apply(attr)] + map(as_apply, args),
                {},
                o_len=None,
                pure=True)

    # ----

    def define(self, f, o_len=None, pure=False, learn_args=(), fit_class=None, compiler=None):
        """Decorator for adding python functions to self
        """
        name = f.__name__
        if hasattr(self, name):
            raise ValueError('Cannot override existing symbol', name)

        entry = SymbolTableEntry(self, name, o_len, pure, learn_args, fit_class, compiler=compiler)
        setattr(self, name, entry)
        self._impls[name] = f
        return f

    def define_pure(self, f):
        return self.define(f, o_len=None, pure=True)

    def define_info(self, o_len=None, pure=False, learn_args=(), fit_class=None, compiler=None):
        def wrapper(f):
            return self.define(f, o_len=o_len, pure=pure, learn_args=learn_args, fit_class=fit_class, compiler=compiler)
        return wrapper

    def inject(self, *args, **kwargs):
        """
        Add symbols from self into a dictionary and return the dict.

        This is used for import-like syntax: see `import_`.
        """
        rval = {}
        for k in args:
            try:
                rval[k] = getattr(self, k)
            except AttributeError:
                raise PyllImportError(k)
        for k, origk in kwargs.items():
            try:
                rval[k] = getattr(self, origk)
            except AttributeError:
                raise PyllImportError(origk)
        return rval

    def import_(self, _globals, *args, **kwargs):
        _globals.update(self.inject(*args, **kwargs))


class SymbolTableEntry(object):
    """A functools.partial-like class for adding symbol table entries.
    """
    def __init__(self, symbol_table, apply_name, o_len, pure, 
                 learn_args, fit_class, compiler):
        self.symbol_table = symbol_table
        self.apply_name = apply_name
        self.o_len = o_len
        self.pure = pure
        self.learn_args = learn_args
        self.fit_class = fit_class
        self.compiler = compiler

    def __call__(self, *args, **kwargs):
        _label = kwargs.pop('_label', '')
        fit_kwargs = kwargs.pop('fit_kwargs', {})
        compiler_data = kwargs.pop('_compiler_data', ())

        return self.symbol_table._new_apply(
            self.apply_name,
            args,
            kwargs,
            o_len=self.o_len,
            pure=self.pure,
            learn_args=self.learn_args,
            fit_class=self.fit_class,
            fit_kwargs=fit_kwargs,
            _label=_label,
            compiler=self.compiler,
            compiler_data=compiler_data)        

scope = SymbolTable()


def as_apply(obj):
    """Smart way of turning object into an Apply
    """
    if isinstance(obj, Apply):
        rval = obj
    elif isinstance(obj, tuple):
        rval = Apply('pos_args', [as_apply(a) for a in obj], {}, o_len=len(obj))
    elif isinstance(obj, list):
        rval = Apply('pos_args', [as_apply(a) for a in obj], {}, o_len=None)
    elif isinstance(obj, dict):
        items = obj.items()
        # -- should be fine to allow numbers and simple things
        #    but think about if it's ok to allow Applys
        #    it messes up sorting at the very least.
        items.sort()
        if all(isinstance(k, basestring) for k in obj):
            named_args = [(k, as_apply(v)) for (k, v) in items]
            rval = Apply('dict', [], named_args, o_len=len(named_args))
        else:
            new_items = [(k, as_apply(v)) for (k, v) in items]
            rval = Apply('dict', [as_apply(new_items)], {}, o_len=None)
    else:
        rval = Literal(obj)
    assert isinstance(rval, Apply)
    return rval


class Apply(object):
    """
    Represent a symbolic application of a symbol to arguments.

    o_len - None or int if the function is guaranteed to return a fixed number
        `o_len` of outputs if it returns successfully
    pure - True only if the function has no relevant side-effects
    """

    def __init__(self, name, pos_args, named_args,
            learn_args=(), fit_class=None, fit_kwargs=None,
            compiler=None, compiler_data=None,
            o_len=None, pure=False, _label=None):
            
        self.name = name
        # -- tuples or arrays -> lists
        self.pos_args = list(pos_args)
        self.named_args = [[kw, arg] for (kw, arg) in named_args]
        # -- o_len is attached this early to support tuple unpacking and
        #    list coersion.
        self.o_len = o_len
        self.pure = pure
        if _label is None:
            _label = {}
        if compiler_data is None:
            compiler_data = ()
        self.compiler_data = compiler_data
        self._label = _label
        assert all(isinstance(v, Apply) for v in pos_args)
        assert all(isinstance(v, Apply) for k, v in named_args)
        assert all(isinstance(k, basestring) for k, v in named_args)

        self.compiler = compiler
        self.learn_args = learn_args
        self.fit_class = fit_class
        if fit_kwargs is None:
            fit_kwargs = {}
        self.fit_kwargs = fit_kwargs
        self.fit_obj = None
        
        LAs = self.learn_args
        self.initial_values = []
        assert isinstance(LAs, tuple), LAs
        assert 'fit' not in LAs
        for la in LAs:
            lv = self.learn_arg_dict[la]
            assert isinstance(lv, Literal)
            self.initial_values.append(lv.eval())
        
        self.initialize_fitter()


    def eval(self, memo=None):
        """
        Recursively evaluate an expression graph.

        This method operates directly on the graph of extended inputs to this
        node, making no attempt to modify or optimize the expression graph.

        Caveats:

          * If there are nodes in the graph that do not represent expressions,
            (e.g. nodes that correspond to statement blocks or assertions)
            then it's not clear what this routine should do, and you should
            probably not call it.

          * If there are Lambdas in the graph, this procedure will not evluate
            them -- see rec_eval for that.

        However, for many cases that are pure expression graphs, this
        offers a quick and simple way to evaluate them.
        """
        if memo is None:
            memo = {}
        if id(self) in memo:
            return memo[id(self)]
        else:
            args = [a.eval() for a in self.pos_args]
            kwargs = dict([(n, a.eval()) for (n, a) in self.named_args])
            f = scope._impls[self.name]
            memo[id(self)] = rval = f(*args, **kwargs)
            return rval

    def inputs(self):
        rval = self.pos_args + [v for (k, v) in self.named_args]
        assert all(isinstance(arg, Apply) for arg in rval)
        return rval

    @property
    def arg(self):
        # XXX: move this introspection to __init__, and change
        #      the basic data-structure to not use pos_args and named_args.
        # XXX: think though... we want the binding to be updated if pos_args
        # and named_args is modified... so maybe this is an ok way to do it?
        #
        # XXX: extend something to deal with Lambda objects instead of
        # decorated python functions.
        #
        # http://docs.python.org/reference/expressions.html#calls
        #
        binding = {}

        fn = scope._impls[self.name]
        # XXX does not work for builtin functions
        defaults = fn.__defaults__  # right-aligned default values for params
        code = fn.__code__

        extra_args_ok = bool(code.co_flags & 0x04)
        extra_kwargs_ok = bool(code.co_flags & 0x08)

        # -- assert that my understanding of calling protocol is correct
        try:
            if extra_args_ok and extra_kwargs_ok:
                assert len(code.co_varnames) >= code.co_argcount + 2
                param_names = code.co_varnames[:code.co_argcount + 2]
                args_param = param_names[code.co_argcount]
                kwargs_param = param_names[code.co_argcount + 1]
                pos_params = param_names[:code.co_argcount]
            elif extra_kwargs_ok:
                assert len(code.co_varnames) >= code.co_argcount + 1
                param_names = code.co_varnames[:code.co_argcount + 1]
                kwargs_param = param_names[code.co_argcount]
                pos_params = param_names[:code.co_argcount]
            elif extra_args_ok:
                assert len(code.co_varnames) >= code.co_argcount + 1
                param_names = code.co_varnames[:code.co_argcount + 1]
                args_param = param_names[code.co_argcount]
                pos_params = param_names[:code.co_argcount]
            else:
                assert len(code.co_varnames) >= code.co_argcount
                param_names = code.co_varnames[:code.co_argcount]
                pos_params = param_names[:code.co_argcount]
        except AssertionError:
            print 'YIKES: MISUNDERSTANDING OF CALL PROTOCOL:',
            print code.co_argcount,
            print code.co_varnames,
            print '%x' % code.co_flags
            raise

        if extra_args_ok:
            binding[args_param] == []

        if extra_kwargs_ok:
            binding[kwargs_param] == {}

        if len(self.pos_args) > code.co_argcount and not extra_args_ok:
            raise TypeError('Argument count exceeds number of positional params')

        # -- bind positional arguments
        for param_i, arg_i in zip(param_names, self.pos_args):
            binding[param_i] = arg_i

        if extra_args_ok:
            binding[args_param].extend(args[code.co_argcount:])

        # -- bind keyword arguments
        for aname, aval in self.named_args:
            try:
                pos = pos_params.index(aname)
            except ValueError:
                if extra_kwargs_ok:
                    binding[kwargs_param][aname] = aval
                    continue
                else:
                    raise TypeError('Unrecognized keyword argument', aname)
            param = param_names[pos]
            if param in binding:
                raise TypeError('Duplicate argument for parameter', param)
            binding[param] = aval

        assert len(binding) <= len(param_names)

        if len(binding) < len(param_names):
            for p in param_names:
                if p not in binding:
                    binding[p] = MissingArgument

        return binding

    def set_kwarg(self, name, value):
        for ii, (key, val) in enumerate(self.named_args):
            if key == name:
                self.named_args[ii][1] = as_apply(value)
                return
        arg = self.arg
        if name in arg and arg[name] != MissingArgument:
            raise NotImplementedError('change pos arg to kw arg')
        else:
            self.named_args.append([name, as_apply(value)])
            self.named_args.sort()

    def clone_from_inputs(self, inputs, learn_args='same', fit_class='same',
                          fit_kwargs='same',  
                          o_len='same', _label='same'):
        if len(inputs) != len(self.inputs()):
            raise TypeError()
        L = len(self.pos_args)
        pos_args = list(inputs[:L])
        named_args = [[kw, inputs[L + ii]]
                for ii, (kw, arg) in enumerate(self.named_args)]
        
        if learn_args == 'same':
            learn_args = self.learn_args
        if fit_class == 'same':
            fit_class = self.fit_class
        # -- danger cloning with new inputs can change the o_len
        if o_len == 'same':
            o_len = self.o_len
        if _label == 'same':
            _label = self._label
        if fit_kwargs == 'same':
            fit_kwargs = self.fit_kwargs
        return self.__class__(self.name, pos_args, named_args, 
                              learn_args=learn_args, 
                              fit_class=fit_class, 
                              fit_kwargs=fit_kwargs,
                              o_len=o_len, 
                              _label=_label)
          
    def replace_input(self, old_node, new_node):
        rval = []
        for ii, aa in enumerate(self.pos_args):
            if aa is old_node:
                self.pos_args[ii] = new_node
                rval.append(ii)
        for ii, (nn, aa) in enumerate(self.named_args):
            if aa is old_node:
                self.named_args[ii][1] = new_node
                rval.append(ii + len(self.pos_args))
        return rval
        
    @property
    def named_arg_dict(self):
        return dict(self.named_args)
        
    def replace_inputs(self, old_nodes, new_nodes):
        T = toposort(self)
        for o, n in zip(old_nodes, new_nodes):
            I = [t for t in T if o in t.inputs()]
            for i in I:
                i.replace_input(o, n)

    def get_node_by_label(self, lbl):
        T = dfs(self)
        return [t for t in T if label_matches(t._label, lbl)]
        
    def pprint(self, ofile, lineno=None, indent=0, memo=None):
        if memo is None:
            memo = {}
        if lineno is None:
            lineno = [0]

        if self in memo:
            print >> ofile, lineno[0], ' ' * indent + memo[self]
            lineno[0] += 1
        else:
            memo[self] = self.name + ('  [line:%i]' % lineno[0])
            print >> ofile, lineno[0], ' ' * indent + self.name
            lineno[0] += 1
            for arg in self.pos_args:
                arg.pprint(ofile, lineno, indent + 2, memo)
            for name, arg in self.named_args:
                print >> ofile, lineno[0], ' ' * indent + ' ' + name + ' ='
                lineno[0] += 1
                arg.pprint(ofile, lineno, indent + 2, memo)

    @property
    def learn_arg_dict(self):
        return dict([(la, self.named_arg_dict[la]) for la in self.learn_args])
        
    @property
    def all_learn_nodes(self):
        x = []
        for ln in self.inputs():
            if isinstance(ln, Apply):
                x.extend(ln.all_learn_nodes)
        x.extend(self.learn_arg_dict.values())
        return x
              
    def bnodes(self):
        n = []
        for inp in self.inputs():
            if inp.name != 'partial':
                n.extend(inp.bnodes())
        n.append(self)
        return n
        
    @property
    def leaves(self):
        G = nx.DiGraph()
        #for node in self.bnodes():
            #G.add_edges_from([(n_in, node) for n_in in node.inputs() if n_in.name != 'partial'])
        for node in dfs(self):
            G.add_edges_from([(n_in, node) for n_in in node.inputs()])
        L = [n for n,d in G.in_degree().items() if d == 0]
        A = self.all_learn_nodes
        return [n for n in L if n not in A]
        
    @property
    def learn_arg_vals(self):
        return dict([(la, self.learn_arg_dict[la].eval()) for la in self.learn_args])
        
    def initialize_fitter(self):
        if self.fit_class is not None:
            fit_kwargs = self.fit_kwargs
            lvs = self.learn_arg_vals
            fit_kwargs.update(lvs)
            self.fit_obj = self.fit_class(**fit_kwargs)
            assert hasattr(self.fit_obj, 'fit')
            for la in self.learn_args:
                assert hasattr(self.fit_obj, la)
        for inp in self.inputs():
            if isinstance(inp, Apply):
                inp.initialize_fitter()    


    def __str__(self):
        sio = StringIO()
        self.pprint(sio)
        return sio.getvalue()[:-1]  # remove trailing '\n'

    def __add__(self, other):
        return scope.add(self, other)

    def __radd__(self, other):
        return scope.add(other, self)

    def __sub__(self, other):
        return scope.sub(self, other)

    def __rsub__(self, other):
        return scope.sub(other, self)

    def __neg__(self):
        return scope.neg(self)

    def __mul__(self, other):
        return scope.mul(self, other)

    def __rmul__(self, other):
        return scope.mul(other, self)

    def __div__(self, other):
        return scope.div(self, other)

    def __rdiv__(self, other):
        return scope.div(other, self)

    def __floordiv__(self, other):
        return scope.floordiv(self, other)

    def __rfloordiv__(self, other):
        return scope.floordiv(other, self)

    def __pow__(self, other):
        return scope.pow(self, other)

    def __rpow__(self, other):
        return scope.pow(other, self)

    def __gt__(self, other):
        return scope.gt(self, other)

    def __ge__(self, other):
        return scope.ge(self, other)

    def __lt__(self, other):
        return scope.lt(self, other)

    def __le__(self, other):
        return scope.le(self, other)

    def __getitem__(self, idx):
        if self.o_len is not None and isinstance(idx, int):
            if idx >= self.o_len:
                #  -- this IndexError is essential for supporting
                #     tuple-unpacking syntax or list coersion of self.
                raise IndexError()
        return scope.getitem(self, idx)

    def __len__(self):
        if self.o_len is None:
            return object.__len__(self)
        return self.o_len

    def __call__(self, *args, **kwargs):
        return scope.call(self, args, kwargs)


def label_matches(l, L):
    for k in L:
        if k in l:
            if isinstance(L[k], list):
                return l[k] in L[k]
            else:
                return l[k] == L[k]
        else:
            return False

                    
def apply(name, *args, **kwargs):
    pos_args = [as_apply(a) for a in args]
    named_args = [(k, as_apply(v)) for (k, v) in kwargs.items()]
    named_args.sort()
    return Apply(name,
            pos_args=pos_args,
            named_args=named_args,
            o_len=None)


class Literal(Apply):
    def __init__(self, obj=None):
        try:
            o_len = len(obj)
        except TypeError:
            o_len = None
        Apply.__init__(self, 'literal', pos_args=(), named_args={}, o_len=o_len, pure=True, fit_class=None, learn_args=(), compiler=None)
        self._obj = obj

    def eval(self, memo=None):
        if memo is None:
            memo = {}
        return memo.setdefault(id(self), self._obj)

    @property
    def obj(self):
        return self._obj

    @property
    def arg(self):
        return {}

    def pprint(self, ofile, lineno=None, indent=0, memo=None):
        if lineno is None:
            lineno = [0]
        if memo is None:
            memo = {}
        if self in memo:
            print >> ofile, lineno[0], ' ' * indent + memo[self]
        else:
            # TODO: set up a registry for this
            if isinstance(self._obj, np.ndarray):
                msg = 'Literal{np.ndarray,shape=%s,min=%f,max=%f}' % (
                        self._obj.shape, self._obj.min(), self._obj.max())
            else:
                msg = 'Literal{%s}' % str(self._obj)
            memo[self] = '%s  [line:%i]' % (msg, lineno[0])
            print >> ofile, lineno[0], ' ' * indent + msg
        lineno[0] += 1

    def replace_input(self, old_node, new_node):
        return []

    def clone_from_inputs(self, inputs, o_len='same'):
        return self.__class__(self._obj)


class Lambda(object):

    # XXX: Extend Lambda objects to have a list of exception clauses.
    #      If the code of the expr() throws an error, these clauses convert
    #      that error to a return value.

    def __init__(self, name, params, expr):
        self.__name__ = name  # like a python function
        self.params = params  # list of (name, symbol[, default_value]) tuples
        self.expr = expr      # pyll graph defining this Lambda

    def __call__(self, *args, **kwargs):
        # -- return `expr` cloned from given args and kwargs
        if len(args) > len(self.params):
            raise TypeError('too many arguments')
        memo = {}
        for arg, param in zip(args, self.params):
            #print 'applying with arg', param, arg
            memo[param[1]] = as_apply(arg)
        if len(args) != len(self.params) or kwargs:
            raise NotImplementedError('named / default arguments',
                    (args, self.params))
        rval = clone(self.expr, memo)
        #print 'BEFORE'
        #print self.expr
        #print 'AFTER'
        #print rval
        return rval


class UndefinedValue(object):
    pass


# -- set up some convenience symbols to use as parameters in Lambda definitions
p0 = Literal(UndefinedValue)
p1 = Literal(UndefinedValue)
p2 = Literal(UndefinedValue)
p3 = Literal(UndefinedValue)
p4 = Literal(UndefinedValue)


@scope.define
def call(fn, args=(), kwargs={}):
    """ call fn with given args and kwargs.

    This is used to represent Apply.__call__
    """
    return fn(*args, **kwargs)


@scope.define
def callpipe1(fn_list, arg):
    """

    fn_list: a list lambdas  that return either pyll expressions or python
        values

    arg: the argument to the first function in the list

    return: `fn_list[-1]( ... (fn_list[1](fn_list[0](arg))))`

    """
    # XXX: in current implementation, if fs are `partial`, then
    #      this loop will expand all functions f at once, so that they
    #      will all be evaluated in the same scope/memo by rec_eval.
    #      Normally programming languages would evaluate each f in a private
    #      scope
    for f in fn_list:
        arg = f(arg)
    return arg


@scope.define
def partial(name, *args, **kwargs):
    # TODO: introspect the named instruction, to retrieve the
    #       list of parameters *not* accounted for by args and kwargs
    # then delete these stupid functions and just have one `partial`
    try:
        name = name.apply_name  # to retrieve name from scope.foo methods
    except AttributeError:
        pass

    my_id = len(scope._impls)
    # -- create a function with this name
    #    the name is the string used index into scope._impls
    temp_name = 'partial_%s_id%i' % (name, my_id)
    l = Lambda(temp_name, [('x', p0)],
            expr=apply(name, *(args + (p0,)), **kwargs))
    scope.define(l)
    # assert that the next partial will get a different id
    # XXX; THIS ASSUMES THAT SCOPE ONLY GROWS
    assert my_id < len(scope._impls)
    rval = getattr(scope, temp_name)
    return rval


def dfs(aa, seq=None, seqset=None):
    if seq is None:
        assert seqset is None
        seq = []
        seqset = set()
    # -- seqset is the set of all nodes we have seen (which may be still on
    #    the stack)
    if aa in seqset:
        return
    assert isinstance(aa, Apply)
    seqset.add(aa)
    for ii in aa.inputs():
        dfs(ii, seq, seqset)
    seq.append(aa)
    return seq


def toposort(expr):
    """
    Return apply nodes of `expr` sub-tree in topological order.

    Raises networkx.NetworkXUnfeasible if subtree contains cycle.

    """
    G = nx.DiGraph()
    for node in dfs(expr):
        G.add_edges_from([(n_in, node) for n_in in node.inputs()])
    order = nx.topological_sort(G)
    assert order[-1] == expr
    return order


def clone(expr, memo=None):
    if memo is None:
        memo = {}
    nodes = dfs(expr)
    for node in nodes:
        if node not in memo:
            new_inputs = [memo[arg] for arg in node.inputs()]
            new_node = node.clone_from_inputs(new_inputs)
            memo[node] = new_node
    return memo[expr]


def clone_merge(expr, memo=None, merge_literals=False):
    nodes = dfs(expr)
    if memo is None:
        memo = {}
    # -- args are somewhat slow to construct, so cache them out front
    #    XXX node.arg does not always work (builtins, weird co_flags)
    node_args = [(node.pos_args, node.named_args) for node in nodes]
    del node
    for ii, node_ii in enumerate(nodes):
        if node_ii in memo:
            continue
        new_ii = None
        if node_ii.pure:
            for jj in range(ii):
                node_jj = nodes[jj]
                if node_ii.name != node_jj.name:
                    continue
                if node_ii.name == 'literal':
                    if not merge_literals:
                        continue
                    if node_ii._obj != node_jj._obj:
                        continue
                else:
                    if node_args[ii] != node_args[jj]:
                        continue
                logger.debug('clone_merge %s %i <- %i' % (
                    node_ii.name, jj, ii))
                new_ii = node_jj
                break
        if new_ii is None:
            new_inputs = [memo[arg] for arg in node_ii.inputs()]
            new_ii = node_ii.clone_from_inputs(new_inputs)
        memo[node_ii] = new_ii

    return memo[expr]


##############################################################################
##############################################################################


class GarbageCollected(object):
    '''Placeholder representing a garbage-collected value '''


def rec_learn(node, memo=None):
    if memo is None:
        memo = {}
    
    if hasattr(node, 'learn_arg_dict'):
        learn_nodes = node.learn_arg_dict.values()
    else:
        learn_nodes = []
    for inp in node.inputs():
        if isinstance(inp, Apply):
            rec_learn(inp, memo=memo)
        #elif inp not in learn_nodes:
        #else:
        #    memo[inp] = rec_eval(inp, memo=memo) 

    if node.fit_obj is not None and node not in memo:
        args = _args = [memo[v] for v in node.pos_args]
        kwargs = _kwargs = dict([(k, memo[v])
            for (k, v) in node.named_args if v not in learn_nodes])
                                            
        node.fit_obj.fit(*args, **kwargs)
        
        for la in node.learn_args:
            nv = getattr(node.fit_obj, la)
            lv = Literal(nv)
            ov = node.learn_arg_dict[la]
            node.replace_input(ov, lv)
            memo[lv] = nv
    
    if node.compiler and node not in memo:
        print('compiling', node, node.name)
        args = _args = [memo[v] for v in node.pos_args]
        kwargs = _kwargs = dict([(k, memo[v])
            for (k, v) in node.named_args])
            
        compiler = node.compiler
        compiler_data = node.compiler_data
        dnode = compiler_data 
        f = getattr(scope, node.name) 
        try:
            data = memo[dnode]
        except:
            data = rec_eval(dnode)
        memo[node] = rec_eval(compiler(f, args, kwargs, data))
    else:
        memo[node] = rec_eval(node, memo=memo)


def learn(fn, learn_data, params):
    """ 
    """
    expr = fn(learn_data, **params)
    rec_learn(expr)
    L = expr.leaves
    assert len(L) == 1, L
    l = L[0]
    expr = clone(expr, {l: p0})    
    fn = get_fn(expr)
    return fn, expr
    
    
def get_fn(expr):
    L = expr.leaves
    assert len(L) == 1
    l = L[0]
    
    def fn(x):
        l = expr.leaves[0]
        expr0 = clone(expr, {l: Literal(x)})
        return rec_eval(expr0)
        
    return fn
    

def learnpipe(fnlist, learn_data, initial_vals_list):
    """ 
    """
    
    expr = learn_data
    exp_list = []
    for fn, ival in zip(fnlist, initial_vals_list):
        expr = fn(**ival)
        exp_list.append(expr)
    
    rec_learn(expr)
    
    return [exp.learn_arg_vals for exp in exp_list]

    
def rec_eval(expr, deepcopy_inputs=False, memo=None,
        max_program_len=None,
        memo_gc=True,
        print_trace=False,
        print_node_on_error=True,
        ):
    """
    expr - pyll Apply instance to be evaluated

    memo - optional dictionary of values to use for particular nodes

    deepcopy_inputs - deepcopy inputs to every node prior to calling that
        node's function on those inputs. If this leads to a different return
        value, then some function (XXX add more complete DebugMode
        functionality) in your graph is modifying its inputs and causing
        mis-calculation. XXX: This is not a fully-functional DebugMode because
        if the offender happens on account of the toposort order to be the last
        user of said input, then it will not be detected as a potential
        problem.

    """
    if max_program_len == None:
        max_program_len = DEFAULT_MAX_PROGRAM_LEN

    if deepcopy_inputs not in (0, 1, False, True):
        # -- I've been calling rec_eval(expr, memo) by accident a few times
        #    this error would have been appreciated.
        raise ValueError('deepcopy_inputs should be bool', deepcopy_inputs)

    node = as_apply(expr)
    topnode = node

    if memo is None:
        memo = {}
    else:
        memo = dict(memo)

    # TODO: optimize dfs to not recurse past the items in memo
    #       this is especially important for evaluating Lambdas
    #       which cause rec_eval to recurse
    #
    # N.B. that Lambdas may expand the graph during the evaluation
    #      so that this iteration may be an incomplete
    if memo_gc:
        clients = {}
        for aa in dfs(node):
            clients.setdefault(aa, set())
            for ii in aa.inputs():
                clients.setdefault(ii, set()).add(aa)
        def set_memo(k, v):
            assert v is not GarbageCollected
            memo[k] = v
            for ii in k.inputs():
                # -- if all clients of ii are already in the memo
                #    then we can free memo[ii] by replacing it
                #    with a dummy symbol
                if all(iic in memo for iic in clients[ii]):
                    #print 'collecting', ii
                    memo[ii] = GarbageCollected
    else:
        def set_memo(k, v):
            memo[k] = v

    todo = deque([topnode])
    while todo:
        if len(todo) > max_program_len:
            raise RuntimeError('Probably infinite loop in document')
        node = todo.pop()
        if print_trace:
            print 'rec_eval:print_trace', len(todo), node.name

        if node in memo:
            # -- we've already computed this, move on.
            continue

        # -- different kinds of nodes are treated differently:
        if node.name == 'switch':
            # -- switch is the conditional evaluation node
            switch_i_var = node.pos_args[0]
            if switch_i_var in memo:
                switch_i = memo[switch_i_var]
                try:
                    int(switch_i)
                except:
                    raise TypeError('switch argument was', switch_i)
                if switch_i != int(switch_i) or switch_i < 0:
                    raise ValueError('switch pos must be positive int',
                            switch_i)
                rval_var = node.pos_args[switch_i + 1]
                if rval_var in memo:
                    set_memo(node, memo[rval_var])
                    continue
                else:
                    waiting_on = [rval_var]
            else:
                waiting_on = [switch_i_var]
        elif isinstance(node, Literal):
            # -- constants go straight into the memo
            set_memo(node, node.obj)
            continue
        else:
            # -- normal instruction-type nodes have inputs
            waiting_on = [v for v in node.inputs() if v not in memo]

        if waiting_on:
            # -- Necessary inputs have yet to be evaluated.
            #    push the node back in the queue, along with the
            #    inputs it still needs
            todo.append(node)
            todo.extend(waiting_on)
        else:
            # -- not waiting on anything;
            #    this instruction can be evaluated.
            args = _args = [memo[v] for v in node.pos_args]
            kwargs = _kwargs = dict([(k, memo[v])
                for (k, v) in node.named_args])

            if memo_gc:
                for aa in args + kwargs.values():
                    assert aa is not GarbageCollected

            if deepcopy_inputs:
                args = copy.deepcopy(_args)
                kwargs = copy.deepcopy(_kwargs)

            try:
                rval = scope._impls[node.name](*args, **kwargs)

            except Exception, e:
                if print_node_on_error:
                    print '=' * 80
                    print 'ERROR in rec_eval'
                    print 'EXCEPTION', type(e), str(e)
                    print 'NODE'
                    print node  # -- typically a multi-line string
                    print '=' * 80
                raise

            if isinstance(rval, Apply):
                # -- if an instruction returns a Pyll apply node
                # it means evaluate that too. Lambdas do this.
                #
                # XXX: consider if it is desirable, efficient, buggy
                #      etc. to keep using the same memo dictionary
                foo = rec_eval(rval, deepcopy_inputs, memo,
                        memo_gc=memo_gc)
                set_memo(node, foo)
            else:
                set_memo(node, rval)

    return memo[topnode]


############################################################################
############################################################################

@scope.define_pure
def pos_args(*args):
    return args


@scope.define_pure
def getitem(obj, idx):
    return obj[idx]


@scope.define_pure
def identity(obj):
    return obj


@scope.define_pure
def add(a, b):
    return a + b


@scope.define_pure
def sub(a, b):
    return a - b


@scope.define_pure
def neg(a):
    return -a


@scope.define_pure
def mul(a, b):
    return a * b


@scope.define_pure
def div(a, b):
    return a / b


@scope.define_pure
def floordiv(a, b):
    return a // b


@scope.define_pure
def eq(a, b):
    return a == b


@scope.define_pure
def gt(a, b):
    return a > b


@scope.define_pure
def ge(a, b):
    return a >= b


@scope.define_pure
def lt(a, b):
    return a < b


@scope.define_pure
def le(a, b):
    return a <= b


@scope.define_pure
def exp(a):
    return np.exp(a)


@scope.define_pure
def log(a):
    return np.log(a)


@scope.define_pure
def pow(a, b):
    return a ** b


@scope.define_pure
def sin(a):
    return np.sin(a)


@scope.define_pure
def cos(a):
    return np.cos(a)


@scope.define_pure
def tan(a):
    return np.tan(a)


@scope.define_pure
def sum(x, axis=None):
    if axis is None:
        return np.sum(x)
    else:
        return np.sum(x, axis=axis)


@scope.define_pure
def sqrt(x):
    return np.sqrt(x)


@scope.define_pure
def minimum(x, y):
    return np.minimum(x, y)


@scope.define_pure
def maximum(x, y):
    return np.maximum(x, y)


@scope.define_pure
def array_union1(args):
    s = set()
    for a in args:
        s.update(a)
    return np.asarray(sorted(s))


@scope.define_pure
def array_union(*args):
    return array_union1(args)


@scope.define_pure
def asarray(a, dtype=None):
    if dtype is None:
        return np.asarray(a)
    else:
        return np.asarray(a, dtype=dtype)


@scope.define_pure
def str_join(s, seq):
    return s.join(seq)


def _bincount_slow(x, weights=None, minlength=None):
    """backport of np.bincount post numpy 1.6
    """
    if weights is not None:
        raise NotImplementedError()
    if minlength is None:
        rlen = np.max(x) + 1
    else:
        rlen = max(np.max(x) + 1, minlength)
    rval = np.zeros(rlen, dtype='int')
    for xi in np.asarray(x).flatten():
        rval[xi] += 1
    return rval


@scope.define_pure
def bincount(x, weights=None, minlength=None):
    if np_versions[0] == 1 and np_versions[1] < 6:
        # -- np.bincount doesn't have minlength arg
        return _bincount_slow(x, weights, minlength)
    else:
        if np.asarray(x).size:
            return np.bincount(x, weights, minlength)
        else:
            # -- currently numpy rejects this case,
            #    but it seems sensible enough to me.
            return np.zeros(minlength, dtype='int')


@scope.define_pure
def repeat(n_times, obj):
    return [obj] * n_times


@scope.define
def call_method(obj, methodname, *args, **kwargs):
    method = getattr(obj, methodname)
    return method(*args, **kwargs)


@scope.define_pure
def call_method_pure(obj, methodname, *args, **kwargs):
    method = getattr(obj, methodname)
    return method(*args, **kwargs)


@scope.define_pure
def copy_call_method_pure(obj, methodname, *args, **kwargs):
    # -- this method copies object before calling the method
    #    so that in the case where args and kwargs are not modified
    #    the call_method can be done in a no-side-effect way.
    #
    #    It is a mistake to use this method when args or kwargs are modified
    #    by the call to method.
    method = getattr(copy.copy(obj), methodname)
    return method(*args, **kwargs)


@scope.define_pure
def switch(pos, *args):
    # switch is an unusual expression, in that it affects control flow
    # when executed with rec_eval. args are not all evaluated, only
    # args[pos] is evaluated.
    ##raise RuntimeError('switch is not meant to be evaluated')
    #
    # .. However, in quick-evaluation schemes it is handy that this be defined
    # as follows:
    return args[pos]


@scope.define_pure
def Raise(etype, *args, **kwargs):
    raise etype(*args, **kwargs)


@scope.define_info(o_len=2)
def curtime(obj):
    return time.time(), obj


@scope.define
def pdb_settrace(obj):
    import pdb; pdb.set_trace()
    return obj

