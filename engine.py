from typing import Any


from cProfile import label
import math
import numpy as np
import matplotlib.pyplot as plt
from draw import draw_dot
import operator

class Value:
    OPS = { '+': operator.add, '*': operator.mul }
    H = 0.0001

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self._prev = set[Value](_children)
        self._op = _op
        self.label = label
        self.grad = 0.0
        self._backward = lambda: None
        
    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad = ((self.data + Value.H) + other.data - out.data)/Value.H * out.grad
            other.grad = ((other.data + Value.H), self.data - out.data)/Value.H * out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*') 
        def _backward():
            self.grad = ((self.data + Value.H) * other.data - out.data)/Value.H * out.grad
            other.grad = ((other.data + Value.H) *self.data - out.data)/Value.H * out.grad

        out._backward = _backward
        return out 

    def tanh(self):
        x = self.data
        

    # for example for child d, self.grad * dL/dd = 1 * f
    def backprop(self):
        op = Value.OPS[self._op]
        c1, c2 = self._prev
        c1.grad = (op(c1.data + Value.H, c2.data) - self.data)/Value.H * self.grad
        c2.grad = (op(c2.data + Value.H, c1.data) - self.data)/Value.H * self.grad


def lol():
    h = 0.0001
    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')
    e = a*b; e.label='e'
    d = e + c; d.label = 'd'
    f = Value(-2.0, label='f')
    L = d * f; L.label = 'L'; L.grad = 1
    return L
    
def propagate(node: Value):
    if len(node._prev) == 0: return

    node.backprop()
    for child in node._prev:
        propagate(child)

L = lol()
propagate(L)
dot = draw_dot(L)
dot.render('graph.gv', view=True, format='png') 
