import mxnet as mx
import numpy as np
from mxnet.gluon import nn, Block, rnn, contrib
from d3 import D3

class Generator(Block):

    def __init__(self, ):

        super(Generator, self).__init__()

        with self.name_scope():

            self.sequence = nn.Sequential()

            self.G = D3()

            self.D = D3()

    def forward(inputs):

        generated = generator(inputs)







