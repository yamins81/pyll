import numpy as np

from pyll import scope, rec_eval, rec_learn, Literal, toposort, learn

class fitter(object):
    def __init__(self, a):
        self.a = a
    
    def fit(self, data):
        self.a += data.shape[0]/10.
        
        
class fitter2(object):
    def __init__(self, a, b):
        self.a = a
        self.b = b
    
    def fit(self, data):
        self.a += data.shape[0]/10.    
        self.b += 0.5

        
@scope.define_info(learn_args=('a',), fit_class=fitter)
def fn1(X, a):
    return X + a

    
@scope.define_info(learn_args=('a','b'), fit_class=fitter2)
def fn2(X, a, b):
    return a*X + b


def test_fn1():
    X0 = np.zeros((30, 3))
    A = scope.fn1(X0, a=0)
    rec_learn(A)
    
    arg0 = A.pos_args[0]
    X1 = Literal(np.ones((10, 3)))
    A.replace_input(arg0, X1)
    rval = rec_eval(A)
    
    assert (rval == 4 * np.ones((10, 3))).all()
    
    
def test_fn12():
    X0 = np.zeros((30, 3))
    A = scope.fn1(X0, a=0)
    B = scope.fn2(A, a=1, b=0.5)
    rval0 = rec_eval(B)
    assert (rval0 == 0.5 * np.ones((30, 3))).all(), rval0
    
    rec_learn(B)
    
    L = B.leaves
    assert len(L) == 1
    l = L[0]
    X1 = Literal(np.ones((10, 3)))
    B.replace_inputs([l], [X1])
    rval = rec_eval(B)
    
    assert (rval == 17.0 * np.ones((10, 3))).all(), rval


def fn12(X, a0, a1, b):
    A = scope.fn1(X, a=a0, _label={'name': 'step1', 'type': 'A'})
    B = scope.fn2(A, a=a1, b=b, _label={'name': 'step2', 'type': 'A'})
    return B


def test_learn():
    G, expr = learn(fn12, np.zeros((30, 3)), {'a0':1, 'a1':1, 'b':-.2})
    assert (G(np.ones((10, 3))) == 20.3 * np.ones((10, 3))).all()
    
    d = expr.get_node_by_label({'name': 'step2'})[0].learn_arg_vals
    assert d == {'a': 4.0, 'b': 0.3}
    
    assert len(expr.get_node_by_label({'type': 'A'})) == 2
    assert len(expr.get_node_by_label({'type': 'B'})) == 0
    
    return expr

   


