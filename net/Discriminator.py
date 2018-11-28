import mxnet as mx
import numpy as np
from mxnet.gluon import nn, Block, rnn, contrib
from .d3 import D3
from .wave2d import *
from .srnn import srnn
from .srnnseq import srnnseq
from .network import birnn, lstm

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
            elif 'd2' in arch:
                self.D = Wave2D(**config)
                if 'lstm' in arch:
                    self.lstm = nn.Sequential()
                    with self.lstm.name_scope():
                        #self.lstm.add(nn.LayerNorm(axis = 2))
                        self.lstm.add(rnn.LSTM(int(config['feature'] / 4), bidirectional = True))
                        #self.lstm.add(nn.LayerNorm(axis = 2))
                        self.lstm.add(nn.Dense(1, 'sigmoid', flatten = False))
            elif 'lstm' in arch:
                self.seq = birnn([256], True)
            elif 'srnnseq' in arch:
                self.seq = birnn([256], True)
                #self.seq = srnnseq(256, stride = 2, layers = 3)
            elif 'srnn' in arch:
                self.seq = srnn(256, stride = 2, layers = 3)

            elif arch == 'reinforce':
                pass
    def forward(self, frame, score):
        if self.arch == 'd3':
            return mx.nd.sum(self.D(mx.nd.concat(*[frame, score], dim = 2)).swapaxes(0, 1), axis = 1)
        elif self.arch == 'd2':
            return mx.nd.sum(self.D(mx.nd.concat(*[frame, score], dim = 2)).swapaxes(0, 1), axis = 1)
        elif self.arch == 'd3-lstm' or self.arch == 'd2-lstm':
            #print(self.D(mx.nd.concat(frame, score, dim = 2))[0].shape)
            out = self.lstm(self.D(mx.nd.concat(frame, score, dim = 2)))
            return mx.nd.concat(*[out[0], out[-1]], dim = 1) # First Step and Last Step
        elif 'lstm' in self.arch:
            out = self.seq(mx.nd.concat(frame, score, dim = 2))
            return mx.nd.concat(*[out[0], out[-1]], dim = 1) # First Step and Last Step
        elif 'srnnseq' in self.arch:
            out = self.seq(mx.nd.concat(frame, score, dim = 2))
            return mx.nd.concat(*[out[0], out[-1]], dim = 1) # First Step and Last Step
        elif 'srnn' in self.arch:
            out = self.seq(mx.nd.concat(frame, score, dim = 2))
            return mx.nd.concat(*[out[0], out[-1]], dim = 1) # First Step and Last Step

