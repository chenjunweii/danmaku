import mxnet as mx
import numpy as np
from mxnet import nd
from mxnet.gluon import nn, Block, rnn, contrib

def conv2rnn(inputs):
    b, c, t, f = inputs.shape
    return inputs.reshape([b, t, f]).swapaxes(0,1).reshape([t, b, f])
def rnn2conv(inputs):
    t, b, f = inputs.shape
    return inputs.swapaxes(0,1).reshape([b, -1, t, f])
def conv2Dpad(outputs, shape):
    shape = list(shape)
    oshape = list(outputs.shape)
    if shape[2] != outputs.shape[2]:
        assert(shape[2] > outputs.shape[2])
        shape[1] = oshape[1] # set channel
        shape[2] = shape[2] - outputs.shape[2]
        try:
            assert(shape[2] < 12)
        except:
            print('shape : ', shape[2])
            assert(shape[2] < 10)
        concat = [outputs, nd.zeros(shape, outputs.context)]
        outputs = nd.concat(*concat, dim = 2)
    return outputs


class Wave2D():
    def __init__(self, kernel, stride, dilation, layers, feature, arch = '', 
            auto = False, norm = False, device = None, last = True, flatten = False, reconstruct = False):
        super(Wave2D, self).__init__()
        self.arch = arch
        self.reconstruct = reconstruct
        self.layers = layers
        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        self.norm = norm
        with self.name_scope():
            self.activation = nn.Activation('relu')
            self.tanh = nn.Activation('tanh')
            self.sigmoid = nn.Activation('sigmoid')
            self.relu = nn.Activation('relu')
            self.fc = nn.Dense(1, flatten = False)
            self.dropout = nn.Dropout(0.5)
            self.enorm = []; self.dnorm = []
            self.add(layers, kernel, stride, dilation)

    def forward(self, inputs):
        output = rnn2conv(inputs)
        shapes = []; eos = []
        for e in self.encoder:
            shapes.append(output.shape)
            eos.append(output)
            eo = e((output))
            sigmoid = self.sigmoid(eo)
            tanh = self.tanh(eo)
            current = sigmoid * tanh
            if 'dense' in self.arch:
                output = nd.concat(current, output, dim = 1)
            else:
                output = current
        output = conv2rnn(output)
        output = (self.fc(self.dropout(output)))
        return output

    def add(self, channel, kernel, strides, padding, dilation):
        assert(len(channel) == self.layers)
        for l in range(self.layers):
            self.encoder.append(nn.Conv2D(channel[l], 
                kernel_size = kernel,
                strides = strides,
                padding = padding,
                dilation = dilation))
            
            
class Wave2DT():
    def __init__(self, kernel, stride, dilation, layers, feature, arch = '', 
            auto = False, norm = False, device = None, last = True, flatten = False, reconstruct = False):
        super(Wave2D, self).__init__()
        self.arch = arch
        self.reconstruct = reconstruct
        self.layers = layers
        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        self.norm = norm
        with self.name_scope():
            self.activation = nn.Activation('relu')
            self.tanh = nn.Activation('tanh')
            self.sigmoid = nn.Activation('sigmoid')
            self.relu = nn.Activation('relu')
            self.fc = nn.Dense(1, flatten = False)
            self.dropout = nn.Dropout(0.5)
            self.enorm = []; self.dnorm = []
            self.add(layers, kernel, stride, dilation)

    def forward(self, inputs):
        output = rnn2conv(inputs)
        shapes = []; eos = []
        for e in self.encoder:
            shapes.append(output.shape)
            eos.append(output)
            eo = e((output))
            sigmoid = self.sigmoid(eo)
            tanh = self.tanh(eo)
            current = sigmoid * tanh
            if 'dense' in self.arch:
                output = nd.concat(current, output, dim = 1)
            else:
                output = current
        output = conv2rnn(output)
        output = (self.fc(self.dropout(output)))
        return output
    def add(self, channel, kernel, strides, padding, dilation):
        assert(len(channel) == self.layers)
        for l in range(self.layers):
            conv = nn.Conv2D if 'bottleneck' in self.arch else nn.Conv2DTranspose
            dchannel = 1 if l == 0 else channel[-l]
            self.decoder.insert(0, conv(dchannel,
                kernel_size = [3, 3],
                strides = [2, 1],
                padding = [1, 1],
                dilation = [1, 1]))
            self.register_child(self.encoder[-1])
            self.register_child(self.decoder[0])
            if self.norm:
                self.enorm.append(self.block['norm'](axis = 2))
                self.dnorm.append(self.block['norm'](axis = 2))
                self.register_child(self.enorm[-1])
                self.register_child(self.dnorm[-1])
                if self.reconstruct:
                    self.rnorm.append(self.block['norm'](axis = 2))
                    self.register_child(self.rnorm[-1])

class D2(Block):
    def __init__(self, layers, feature, arch = '', 
            auto = False, norm = False, device = None, last = True, flatten = False, reconstruct = False):
        super(D2, self).__init__()
        self.arch = arch
        self.reconstruct = reconstruct
        self.layers = layers
        self.norm = norm
        with self.name_scope():
            self.activation = nn.Activation('relu')
            self.tanh = nn.Activation('tanh')
            self.sigmoid = nn.Activation('sigmoid')
            self.relu = nn.Activation('relu')
            #self.norm = nn.BatchNorm(axis = 1)
            self.fc = nn.Dense(1, flatten = False)
            self.dropout = nn.Dropout(0.5)
            self.pool = nn.AvgPool3D([3, 1, 1], [2, 1, 1])
            self.encoder = []; self.decoder = []
            self.enorm = []; self.dnorm = []; self.rnorm = []
            self.add([8, 16, 32])

    def forward(self, inputs):
        output = rnn2conv(inputs)
        shapes = []; eos = []; dos = []; los = []; # layer out
        for e in self.encoder:
            shapes.append(output.shape)
            eos.append(output)
            eo = e((output))
            sigmoid = self.sigmoid(eo)
            tanh = self.tanh(eo)
            current = sigmoid * tanh
            if 'dense' in self.arch:
                output = nd.concat(current, output, dim = 1)
            else:
                output = current
        for i, d in enumerate(self.decoder):
            do = d((output))
            do = conv2Dpad(do, shapes[-i - 1])
            sigmoid = self.sigmoid(do)
            tanh = self.tanh(do)
            current = sigmoid * tanh
            if 'dense' in self.arch:
                output = nd.concat(current, output, dim = 1)
            else:
                output = current
        output = conv2rnn(output)
        output = (self.fc(self.dropout(output)))
        return output
    def add(self, channel):
        assert(len(channel) == self.layers)
        for l in range(self.layers):
            self.encoder.append(nn.Conv2D(channel[l], 
                kernel_size = [3, 3],
                strides = [2, 1],
                padding = [1, 1],
                dilation = [1, 1]))
            conv = nn.Conv2D if 'bottleneck' in self.arch else nn.Conv2DTranspose
            dchannel = 1 if l == 0 else channel[-l]
            self.decoder.insert(0, conv(dchannel,
                kernel_size = [3, 3],
                strides = [2, 1],
                padding = [1, 1],
                dilation = [1, 1]))
            self.register_child(self.encoder[-1])
            self.register_child(self.decoder[0])
            if self.reconstruct:
                assert('bottleneck' not in self.arch)
                assert('encoder' in self.arch and 'decoder' in self.arch)
                channel = channel if l != layers - 1 else self.feature
                self.reconstructor.append(nn.Conv2DTranspose(dchannel,
                    kernel_size = [3, 3],
                    strides = [2, 2],
                    padding = [1, 1],
                    dilation = [1, 1]))
                self.register_child(self.reconstructor[-1])
            if self.norm:
                self.enorm.append(self.block['norm'](axis = 2))
                self.dnorm.append(self.block['norm'](axis = 2))
                self.register_child(self.enorm[-1])
                self.register_child(self.dnorm[-1])
                if self.reconstruct:
                    self.rnorm.append(self.block['norm'](axis = 2))
                    self.register_child(self.rnorm[-1])

               
