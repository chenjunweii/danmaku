import mxnet as mx
import numpy as np
from mxnet import nd
from mxnet.gluon import nn, Block, rnn, contrib


def time_major(inputs, fixed_time = False, fixed_feature = False):
    b, c, t, f = inputs.shape
    assert(c == 1)
    return inputs.reshape([b, t, f]).swapaxes(0,1).reshape([t, b, f])
def batch_major(inputs):
    t, b, f = inputs.shape
    return inputs.swapaxes(0,1).reshape([b, -1, t, f])
def conv2Dpad(outputs, shape):
    shape = list(shape)
    oshape = list(outputs.shape)
    if shape[2] != outputs.shape[2]:
        assert(shape[2] > outputs.shape[2])
        shape[1] = oshape[1] # set channel
        shape[2] = shape[2] - outputs.shape[2]
        shape[3] = oshape[3]
        try:
            assert(shape[2] < 1024)
        except:
            assert(shape[2] < 1024)
        concat = [outputs, nd.zeros(shape, outputs.context)]
        outputs = nd.concat(*concat, dim = 2)
    return outputs

class Norm(Block):
    def __init__(self):
        super(Norm, self).__init__()
        with self.name_scope():
            self.norm = nn.LayerNorm(3)
    def forward(self, current):
        #current = current.swapaxes(1, 2)
        current = self.norm(current)
        #current = current.swapaxes(1, 2)
        return current

class TransitionT(Block):
    def __init__(self, **kwargs):
        super(TransitionT, self).__init__()
        for k, v in kwargs.items():
            setattr(self, k, v)
        with self.name_scope():
            self.conv = nn.Conv2DTranspose(self.channels * self.layers, [2, 1], strides = [2, 1], padding = [0, 0])
            self.norm = Norm()
            self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, current):
        current = self.norm(current)
        current = self.lrelu(current)
        current = self.conv(current)
        return current

class Transition(Block):
    def __init__(self, **kwargs):
        super(Transition, self).__init__()
        for k, v in kwargs.items():
            setattr(self, k, v)
        with self.name_scope():
            self.lrelu = nn.LeakyReLU(0.2)
            self.conv = nn.Conv2D(1 if self.out else self.channels * self.layers, [1, 1])
            self.norm = Norm()
            self.pool = nn.AvgPool2D((2, 1), padding = (0,0)) if not self.out else None
    def forward(self, current):
        current = self.norm(current)
        current = self.lrelu(current)
        current = self.conv(current)
        current = self.pool(current) if not self.out else current
        return current

class DenseBlock(Block):
    def __init__(self, ** kwargs):
        super(DenseBlock, self).__init__()
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.conv1 = []
        self.conv2 = []
        self.norm1 = []
        self.norm2 = []
        with self.name_scope():
            self.do = nn.Dropout(0.5)
            self.lrelu = nn.LeakyReLU(0.2)
            for l in range(self.layers):
                self.conv1.append(nn.Conv2D(self.channels, [1, 3], padding = [0, 1]))
                self.conv2.append(nn.Conv2D(self.channels, [3, 1], padding = [1, 0]))
                self.norm1.append(Norm())
                self.norm2.append(Norm())
                self.register_child(self.conv1[-1])
                self.register_child(self.conv2[-1])
                self.register_child(self.norm1[-1])
                self.register_child(self.norm2[-1])
    def forward(self, current):
        outputs = []
        for n1, c1, n2, c2 in zip(self.norm1, self.conv1, self.norm2, self.conv2):
            current = n1(current)
            current = self.lrelu(current)
            current = c1(current)
            current = n2(current)
            current = self.lrelu(current)
            current = c2(current)
            outputs.append(current)
            current = nd.concat(*outputs, dim = 1) if len(outputs) > 1 else current

        return current


class DenseWaveT(Block):
    def __init__(self, **kwargs):
        super(DenseWaveT, self).__init__()
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.dbs = []
        self.ts = []
        with self.name_scope():
            for b in self.blocks:
                self.dbs.append(DenseBlock(**b))
                self.ts.append(TransitionT(**b))
                self.register_child(self.dbs[-1])
                self.register_child(self.ts[-1])

    def forward(self, current, shapes = None):
        for i, (d, t) in enumerate(zip(self.dbs, self.ts)):
            current = d(current)
            current = t(current)
            current = conv2Dpad(current, shapes[i]) if self.decoder else current
        return current
class DenseWave(Block):
    def __init__(self, **kwargs):
        super(DenseWave, self).__init__()
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.dbs = []
        self.ts = []
        with self.name_scope():
            for b in self.blocks:
                self.dbs.append(DenseBlock(**b))
                self.ts.append(Transition(**b))
                self.register_child(self.dbs[-1])
                self.register_child(self.ts[-1])

    def forward(self, current):
        shapes = []
        for db, t in zip(self.dbs, self.ts):
            print('shapes : ', current.shape)
            shapes.insert(0, current.shape)
            current = db(current)
            current = t(current)
        return (current, shapes) if self.encoder else current

class DenseWaveED(Block):

    def __init__(self, encoder, decoder):
        super(DenseWaveED, self).__init__()
        with self.name_scope():
            self.encoder = DenseWave(**encoder)
            self.decoder = DenseWaveT(**decoder)
            self.tout = Transition(out = True) # transition out
             
            self.fcin = nn.Dense(256, flatten = False)
            self.fcout = nn.Dense(1, flatten = False)
    
    def forward(self, current):
        current = self.fcin(current)
        current = batch_major(current)
        current, shapes = self.encoder(current)
        current = self.decoder(current, shapes)
        current = self.tout(current)
        current = time_major(current)
        current = self.fcout(current)
        return current
