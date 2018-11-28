import mxnet as mx
import numpy as np
from mxnet import nd
from mxnet.gluon import nn, Block, rnn, contrib

def conv2rnn(inputs):
    b, c, t, f = inputs.shape
    if c == 1:
        return inputs.reshape([b, t, f]).swapaxes(0,1).reshape([t, b, f])
    elif c != 1:
        return inputs.swapaxes(1,2).swapaxes(0,1).reshape([t, b, -1])
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
        shape[3] = oshape[3]
        try:
            assert(shape[2] < 1024)
        except:
            assert(shape[2] < 1024)
        concat = [outputs, nd.zeros(shape, outputs.context)]
        outputs = nd.concat(*concat, dim = 2)
    return outputs


class Wave2DSEQ(Block):
    def __init__(self, kernel, stride, dilation, layers, feature, channel, padding, swap_in = True, swap_out = True, arch = '', 
            auto = False, norm = False, device = None, last = True, flatten = False, reconstruct = False, block = '', encoder = False):
        super(Wave2D, self).__init__()
        self.arch = arch
        self.swap_in = swap_in; self.swap_out = swap_out
        self.reconstruct = reconstruct
        self.layers = layers
        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        self.norm = norm
        self.sequence = []
        self.encoder = encoder;
        self.channel = channel
        if self.norm:
            self.norms = []
        with self.name_scope():
            self.activation = nn.Activation('relu')
            self.tanh = nn.Activation('tanh')
            self.sigmoid = nn.Activation('sigmoid')
            self.relu = nn.Activation('relu')
            self.lrelu = nn.LeakyReLU(0.1)
            if swap_out:
                self.fc = nn.Dense(1, flatten = False)
            self.dropout = nn.Dropout(0.5)
            self.add(layers, channel, kernel, stride, padding, dilation)

    def forward(self, inputs):
        output = rnn2conv(inputs) if self.swap_in else inputs
        outputs = []
        shapes = []
        oshapes = []
        seq = zip(self.sequence, self.norms) if self.norm else self.sequence
        #print('[*] In Encoder ')
        for i, s in enumerate(seq):
            shapes.append(output.shape)
            oshapes.insert(0, output.shape)
            outputs.insert(0, output)
            o = s[0](s[1](output)) if self.norm else s(output)
            #o = s[0](rnn2conv(s[1](conv2rnn(output)))) if self.norm else s(output)
            #sigmoid = self.sigmoid(o)
            #tanh = self.tanh(o)
            #current = sigmoid * tanh
            current = self.lrelu(o)#sigmoid * tanh
            if 'dense' in self.arch:
                output = nd.concat(current, output, dim = 1)
            else:
                output = current
        if self.swap_out:
            output = self.sigmoid(self.fc(self.dropout(conv2rnn(output))))
        return (output, outputs, oshapes) if self.encoder else output

    def add(self, layers, channel, kernel, strides, padding, dilation):
        assert(len(channel) == self.layers)
        for l in range(self.layers):
            self.sequence.append(nn.Conv2D(channel[l], 
                kernel_size = kernel[l],
                strides = strides[l],
                padding = padding[l],
                dilation = dilation[l]))
            self.register_child(self.sequence[-1])
            if self.norm:
                self.norms.append(nn.LayerNorm(axis = -1))
                self.register_child(self.norms[-1])


class Wave2DED(Block):
    def __init__(self, Encoder, Decoder):
        super(Wave2DED, self).__init__()
        with self.name_scope():
            self.E = Wave2D(**Encoder)
            self.D = Wave2DT(**Decoder)
    def forward(self, inputs):
        #print('[*] IN Encoder-Decoder')
        output, outputs, oshape = self.E(inputs)
        output = self.D(output, inputs, outputs, oshape)
        #print('[*] End Encoder-Decoder')
        return output

class Wave2DT(Block):
    def __init__(self, kernel, stride, dilation, layers, feature, channel, padding, swap_in = True, swap_out = True, arch = '', 
            auto = False, norm = False, device = None, last = True, flatten = False, reconstruct = False):
        super(Wave2DT, self).__init__()
        self.arch = arch
        self.swap_out = swap_out
        self.swap_in = swap_in
        self.reconstruct = reconstruct
        self.layers = layers
        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        self.sequence = []
        self.norm = norm
        self.channel = channel
        with self.name_scope():
            self.activation = nn.Activation('relu')
            self.tanh = nn.Activation('tanh')
            self.sigmoid = nn.Activation('sigmoid')
            self.relu = nn.Activation('relu')
            self.lrelu = nn.LeakyReLU(0.1)
            if swap_out:
                self.fc = nn.Dense(1, 'sigmoid', flatten = False)
            self.dropout = nn.Dropout(0.5)
            if self.norm:
                self.norms = []
            self.add(layers, channel, kernel, stride, padding, dilation)
        if self.norm:
            assert(len(self.norms) == len(self.sequence))
    def forward(self, inputs, oinput = None, encoder_outputs = None, eshape = None):
        output = rnn2conv(inputs) if self.swap_in else inputs
        outputs = []
        shapes = []
        seq = zip(self.sequence, self.norms) if self.norm else self.sequence
        #print('[*] In Decoder ')
        for i, s in enumerate(seq):
            shapes.append(output.shape)
            outputs.append(output)
            o = s[0](s[1](output)) if self.norm else s(output)
            #o = s[0](rnn2conv(s[1](conv2rnn(output)))) if self.norm else s(output)
            o = conv2Dpad(o, eshape[i]) if eshape is not None else o
            #sigmoid = self.sigmoid(o)
            #tanh = self.tanh(o)
            current = self.lrelu(o)#sigmoid * tanh
            #current = sigmoid * tanh
            if 'dense' in self.arch:
                output = nd.concat(current, output, dim = 1)

            elif 'unet' in self.arch:
                output = nd.concat(current, encoder_outputs[i], dim = 3)
            else:
                output = current
        if oinput is not None:
            output = conv2Dpad(output, rnn2conv(oinput).shape)
        return  self.sigmoid(self.fc(self.dropout(conv2rnn(output)))) if self.swap_out else self.sigmoid(output)
    def add(self, layers, channel, kernel, strides, padding, dilation):
        assert(len(channel) == self.layers)
        for l in range(self.layers):
            self.sequence.append(nn.Conv2DTranspose(channel[l],
                kernel_size = kernel[l],
                strides = strides[l],
                padding = padding[l],
                dilation = dilation[l]))
            self.register_child(self.sequence[-1])
            if self.norm:
                self.norms.append(nn.LayerNorm(axis = -1))
                self.register_child(self.norms[-1])


