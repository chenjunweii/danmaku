import mxnet as mx
import numpy as np
from mxnet.gluon import nn, Block, rnn, contrib
from .d3 import D3
from .wave2d import *
from .srnn import srnn

class Discriminator(Block):
    def __init__(self, config, arch = ''):
        super(Discriminator, self).__init__()
        self.arch = arch
        with self.name_scope():
            if 'd3' in arch:
                self.D = D3(**config)
                if 'lstm' in arch:
                    self.lstm = nn.Sequential()
                    with self.lstm.name_scope():
                        #self.lstm.add(nn.LayerNorm(axis = 2))
                        self.lstm.add(rnn.LSTM(int(config['feature'] / 4), bidirectional = True))
                        #self.lstm.add(nn.LayerNorm(axis = 2))
                        self.lstm.add(nn.Dense(1, 'sigmoid', flatten = False))
            if 'd2' in arch:
                self.D = Wave2D(**config)
                if 'lstm' in arch:
                    self.lstm = nn.Sequential()
                    with self.lstm.name_scope():
                        #self.lstm.add(nn.LayerNorm(axis = 2))
                        self.lstm.add(rnn.LSTM(int(config['feature'] / 4), bidirectional = True))
                        #self.lstm.add(nn.LayerNorm(axis = 2))
                        self.lstm.add(nn.Dense(1, 'sigmoid', flatten = False))
            elif arch == 'rnn':
                self.D = nn.Sequential()
                with self.D.name_scope():
                    self.D.add(rnn.LSTM(int(config['feature'] / 4)))
                    self.D.add(nn.Dense(1, flatten = False))
            elif arch == 'srnn':
                self.D = srnn(int(config['feature'] / 4), 2, 5)
            elif arch == 'reinforce':
                pass
    def forward(self, frame, score):
        if self.arch == 'd3':
            return mx.nd.sum(self.D(mx.nd.concat(*[frame, score], dim = 2)).swapaxes(0, 1), axis = 1)
        elif self.arch == 'd3-lstm' or self.arch == 'd2-lstm':
            out = self.lstm(self.D(mx.nd.concat(frame, score, dim = 2)))
            return mx.nd.concat(*[out[0], out[-1]], dim = 1) # First Step and Last Step
        elif self.arch == 'rnn' or self.arch == 'srnn':
            return self.D(mx.nd.concat(*[frame, score], dim = 2))[-1] # last state
