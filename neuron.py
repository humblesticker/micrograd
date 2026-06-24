import random
from draw import draw_dot
from engine import Value


class Neuron:
    def __init__(self, nin, lid, oi):
        self.w = [Value(random.uniform(-1,1), label=f'w{lid}-{i}{oi}') for i in range(nin)]
        self.b = Value(random.uniform(-1,1), label=f'b{lid}-{oi}')

    def __call__(self, x):
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh()

    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, nin, nout, id):
        self.neurons = [ Neuron(nin, id, o) for o in range(nout) ]

    def __call__(self, x):
        outs = [ n(x) for n in self.neurons ]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
        
            
class MLP:
    def __init__(self, nin, nouts):
        sz = [ nin ] + nouts
        self.layers = [ Layer(sz[i], sz[i+1], i) for i in range(len(nouts)) ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters() ]

n = MLP(3, [4, 4, 1])
parameters = n.parameters()

def u1():
    x = [2.0, 3.0, -1.0]
    y = 1.0
    out = n(x)
    print(out)

    loss = (out - y)**2
    return loss

def u2():
    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0]
    ]
    ys = [1.0, -1.0, -1.0, 1.0]

    count = 0
    while True:
        ypred = [n(x) for x in xs]
        loss = sum([(yp - yt)**2 for yt, yp in zip(ys, ypred)])
        print(loss)
        if loss.data < 0.001: 
            print(count, ypred)
            break

        for param in parameters:
            param.grad = 0.0
        loss.backward()

        for param in parameters:
            param.data += -param.grad * 0.05  
        count += 1

u2()
#dot = draw_dot(loss)
#dot.render('graph.gv', view=True, format='png')




