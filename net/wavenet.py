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


class WaveNet(Block):
    def __init__(self, ** kwargs):
        super(WaveNet, self).__init__()
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.tseq = []
        self.sseq = []
        self.oseq = []
        self.pseq = []
        with self.name_scope():
            self.pseq.append(nn.Conv2D(8, [1, 1]))
            self.register_child(self.pseq[-1])
            self.pseq.append(nn.Conv2D(1, [1, 1]))
            self.register_child(self.pseq[-1])
            self.activation = nn.Activation('relu')
            self.tanh = nn.Activation('tanh')
            self.sigmoid = nn.Activation('sigmoid')
            self.relu = nn.Activation('relu')
            self.lrelu = nn.LeakyReLU(0.1)
            self.fc = nn.Dense(1, flatten = False)
            self.fc0 = nn.Dense(128, flatten = False)
            self.dropout = nn.Dropout(0.5)
            self.add()

    def forward(self, inputs):
        outputs = []
        output = rnn2conv(self.fc0(inputs))
        seq = zip(self.tseq, self.sseq, self.oseq)
        for i, (t, s, o) in enumerate(seq):
            outputs.insert(0, output)
            to = self.tanh(t(output))
            so = self.sigmoid(s(output))
            co = o(to * so)
            output = co + output
        output = sum([out for out in outputs]) + output
        for p in self.pseq:
            output = self.lrelu(p(output))

        output = conv2rnn(self.dropout(self.fc(output)))

        return output

    def add(self):
        assert(len(self.channel) == self.layers)
        for l in range(self.layers):
            td = 2 ** (l + 1)
            self.tseq.append(nn.Conv2D(16, 
                kernel_size = [2, 1],
                strides = [1, 1],
                padding = [int(td / 2), 0],
                dilation = [td, 1]))
            self.sseq.append(nn.Conv2D(16, 
                kernel_size = [2, 1],
                strides = [1, 1],
                padding = [int(td / 2), 0],
                dilation = [td, 1]))
            self.oseq.append(nn.Conv2D(8,
                kernel_size = [1, 1],
                strides = [1, 1],
                padding = [0, 0],
                dilation = [1, 1]))
            self.register_child(self.tseq[-1])
            self.register_child(self.sseq[-1])
            self.register_child(self.oseq[-1])


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


