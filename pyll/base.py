# file is called AST to not collide with std lib module 'ast'
#
# It provides types to build ASTs in a simple lambda-notation style
#

from StringIO import StringIO

# TODO: move things depending on numpy (among others too) to a library file
import numpy as np
np_versions = map(int, np.__version__.split('.')[:2])

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
                }

    def _new_apply(self, name, args, kwargs, o_len):
        pos_args = [as_apply(a) for a in args]
        named_args = [(k, as_apply(v)) for (k, v) in kwargs.items()]
        named_args.sort()
        return Apply(name,
                pos_args=pos_args,
                named_args=named_args,
                o_len=o_len)

    def int(self, arg):
        return self._new_apply('int', [as_apply(arg)], {}, o_len=None)

    def float(self, arg):
        return self._new_apply('float', [as_apply(arg)], {}, o_len=None)

    def list(self, init):
        return self._new_apply('list', [as_apply(init)], {}, o_len=None)

    def dict(self, *args, **kwargs):
        # XXX: figure out len
        return self._new_apply('dict', args, kwargs, o_len=None)

    def range(self, *args):
        return self._new_apply('range', args, {}, o_len=None)

    def len(self, obj):
        return self._new_apply('len', [obj], {}, o_len=None)

    def define(self, f, o_len=None):
        """Decorator for adding python functions to self
        """
        name = f.__name__
        if hasattr(self, name):
            raise ValueError('Cannot override existing symbol', name)
        def apply_f(*args, **kwargs):
            return self._new_apply(name, args, kwargs, o_len)
        setattr(self, name, apply_f)
        self._impls[name] = f
        return f

    def define_info(self, o_len):
        def wrapper(f):
            return self.define(f, o_len=o_len)
        return wrapper


scope = SymbolTable()


def as_apply(obj):
    """Smart way of turning object into an Apply
    """
    if isinstance(obj, Apply):
        rval = obj
    elif isinstance(obj, tuple):
        rval = Apply('pos_args', [as_apply(a) for a in obj], {}, len(obj))
    elif isinstance(obj, list):
        rval = Apply('pos_args', [as_apply(a) for a in obj], {}, None)
    elif isinstance(obj, dict):
        items = obj.items()
        # -- should be fine to allow numbers and simple things
        #    but think about if it's ok to allow Applys
        #    it messes up sorting at the very least.
        items.sort()
        named_args = [(k, as_apply(v)) for (k, v) in items]
        rval = Apply('dict', [], named_args, len(named_args))
    else:
        rval = Literal(obj)
    assert isinstance(rval, Apply)
    return rval


class Apply(object):
    """
    Represent a symbolic application of a symbol to arguments.
    """

    def __init__(self, name, pos_args, named_args, o_len=None):
        self.name = name
        # -- tuples or arrays -> lists
        self.pos_args = list(pos_args)
        self.named_args = [[kw, arg] for (kw, arg) in named_args]
        # -- o_len is attached this early to support tuple unpacking and
        #    list coersion.
        self.o_len = o_len
        assert all(isinstance(v, Apply) for v in pos_args)
        assert all(isinstance(v, Apply) for k, v in named_args)
        assert all(isinstance(k, basestring) for k, v in named_args)

    def eval(self, memo=None):
        """
        Recursively evaluate an expression graph.

        This method operates directly on the graph of extended inputs to this
        node, making no attempt to modify or optimize the expression graph.

        :note:
            If there are nodes in the graph that do not represent expressions,
            (e.g. nodes that correspond to statement blocks or assertions)
            then it's not clear what this routine should do, and you should
            probably not call it.

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

    def clone_from_inputs(self, inputs, o_len='same'):
        if len(inputs) != len(self.inputs()):
            raise TypeError()
        L = len(self.pos_args)
        pos_args  = list(inputs[:L])
        named_args = [[kw, inputs[L + ii]]
                for ii, (kw, arg) in enumerate(self.named_args)]
        # -- danger cloning with new inputs can change the o_len
        if o_len == 'same':
            o_len = self.o_len
        return self.__class__(self.name, pos_args, named_args, o_len)

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

    def __str__(self):
        sio = StringIO()
        self.pprint(sio)
        return sio.getvalue()[:-1] # -- remove trailing '\n'

    def __add__(self, other):
        return scope.add(self, other)

    def __radd__(self, other):
        return scope.add(other, self)

    def __sub__(self, other):
        return scope.sub(self, other)

    def __rsub__(self, other):
        return scope.sub(other, self)

    def __mul__(self, other):
        return scope.mul(self, other)

    def __rmul__(self, other):
        return scope.mul(other, self)

    def __div__(self, other):
        return scope.div(self, other)

    def __rdiv__(self, other):
        return scope.div(other, self)


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


class Literal(Apply):
    def __init__(self, obj):
        try:
            o_len = len(obj)
        except TypeError:
            o_len = None
        Apply.__init__(self, 'literal', [], {}, o_len)
        self._obj = obj

    def eval(self, memo=None):
        if memo is None:
            memo = {}
        return memo.setdefault(id(self), self._obj)

    @property
    def obj(self):
        return self._obj

    def pprint(self, ofile, lineno=None, indent=0, memo=None):
        if lineno is None:
            lineno = [0]
        if memo is None:
            memo = {}
        if self in memo:
            print >> ofile, lineno[0], ' ' * indent + memo[self]
        else:
            msg = 'Literal{%s}' % str(self._obj)
            memo[self] =  '%s  [line:%i]' % (msg, lineno[0])
            print >> ofile, lineno[0], ' ' * indent + msg
        lineno[0] += 1

    def replace_input(self, old_node, new_node):
        return []

    def clone_from_inputs(self, inputs, o_len='same'):
        return self.__class__(self._obj)


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


################################################################################
################################################################################


def rec_eval(expr, deepcopy_inputs=False, memo=None):
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
    node = as_apply(expr)
    if memo is None:
        memo = {}
    for aa in dfs(node):
        if isinstance(aa, Literal):
            memo[aa] = aa._obj
    todo = [node]
    topnode = node
    while todo:
        if len(todo) > 100000:
            raise RuntimeError('Probably infinite loop in document')
        node = todo.pop()
        if node not in memo:
            waiting_on = [v for v in node.inputs() if v not in memo]
            if waiting_on:
                todo.extend([node] + waiting_on)
            else:
                args = _args = [memo[v] for v in node.pos_args]
                kwargs = _kwargs = dict([(k, memo[v])
                    for (k, v) in node.named_args])
                if deepcopy_inputs:
                    import copy
                    args = copy.deepcopy(_args)
                    kwargs = copy.deepcopy(_kwargs)
                try:
                    rval = scope._impls[node.name](*args, **kwargs)
                except Exception, e:
                    print '=' * 80
                    print 'ERROR in rec_eval'
                    print 'EXCEPTION', type(e), str(e)
                    print 'NODE'
                    print node
                    print '=' * 80
                    raise
                memo[node] = rval
    return memo[topnode]


################################################################################
################################################################################

@scope.define
def pos_args(*args):
    return args


@scope.define
def getitem(obj, idx):
    return obj[idx]


@scope.define
def identity(obj):
    return obj


@scope.define
def add(a, b):
    return a + b


@scope.define
def sub(a, b):
    return a - b


@scope.define
def mul(a, b):
    return a * b


@scope.define
def div(a, b):
    return a / b


@scope.define
def exp(a):
    return np.exp(a)


@scope.define
def log(a):
    return np.log(a)


@scope.define
def sum(x, axis=None):
    if axis is None:
        return np.sum(x)
    else:
        return np.sum(x, axis=axis)


@scope.define
def sqrt(x):
    return np.sqrt(x)


@scope.define
def array_union(a, b):
    sa = set(a)
    sa.update(b)
    return np.asarray(sorted(sa))


@scope.define
def asarray(a, dtype=None):
    if dtype is None:
        return np.asarray(a)
    else:
        return np.asarray(a, dtype=dtype)


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


@scope.define
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


@scope.define
def repeat(n_times, obj):
    return [obj] * n_times

