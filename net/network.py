import mxnet as mx
from mxnet import nd
from mxnet.gluon import nn, Block, rnn, contrib

def cross_entropy(p, g):

    return -nd.clip(g, 0, 1) * nd.log(nd.clip(p, 1e-5, 1)) - (1 - nd.clip(g, 0, 1)) * nd.log(nd.clip(1 - p, 1e-5, 1))

def ce(p, g):

    return -nd.clip(g, 0, 1) * nd.log(nd.clip(p, 1e-5, 1)) - (1 - nd.clip(g, 0, 1)) * nd.log(nd.clip(1 - p, 1e-5, 1))

def cross_entropy_2(p, g):

    return nd.abs(p - g)

def mse(p, g):

    return (p - g) ** 2

def lstm(nhidden, dropout = 0.5):

    net = mx.gluon.nn.Sequential()
    
    with net.name_scope():

        net.add(mx.gluon.rnn.LSTM(nhidden[0], dropout = dropout))
        
        net.add(mx.gluon.nn.Dense(1, 'sigmoid', flatten = False))

    return net

def birnn(nhidden, activation = False, dropout = 0.5):

    net = mx.gluon.nn.Sequential()

    with net.name_scope():

        #net.add(mx.gluon.nn.LayerNorm(axis = 2))
        
        net.add(mx.gluon.rnn.LSTM(nhidden[0], bidirectional = True, dropout = dropout))
        
        #net.add(mx.gluon.nn.LayerNorm(axis = 2))
        
        net.add(mx.gluon.nn.Dense(1, flatten = False))

        if activation :
        
            net.add(mx.gluon.nn.Activation('sigmoid'))
        
        #net.add(mx.gluon.nn.LayerNorm(axis = 2))

    return net

class BILSTM(Block):

    def __init__(self, n_inputs, n_hidden, n_layers = 1, dropout = 0.5):

        super(BILSTM, self).__init__()

        with self.name_scope():
        
            self.r = rnn.LSTM(n_hidden, n_layers, dropout = dropout, input_size = n_inputs, bidirectional = True)

            self.c_init = nn.Dense(2 * n_hidden, flatten = False, activation = 'tanh')

            self.h_init = nn.Dense(2 * n_hidden, flatten = False, activation = 'tanh')
        
            self.fc = nn.Dense(1, sigmoid, flatten = False)

            self.d = nn.Dropout(.5)

    def forward(self, inputs):

        t, b, f = inputs.shape
        
        c = self.d(self.c_init(inputs[0]).reshape([2, b, -1]))

        h = self.d(self.h_init(inputs[0]).reshape([2, b, -1]))

        ro, last = self.r(inputs, [c, h])

        o = self.d(self.fc(ro))

        return o
