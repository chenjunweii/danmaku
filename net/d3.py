import mxnet as mx
import numpy as np
from mxnet import nd
from mxnet.gluon import nn, Block, rnn, contrib

def conv2rnn(inputs):
    b, c, t, h, w = inputs.shape
    return inputs.swapaxes(1,2).swapaxes(0,1).reshape([t, b, -1])
    #return inputs.swapaxes(0, 1).reshape([t, b, -1])
    
def rnn2conv(inputs):
    t, b, f = inputs.shape
    return inputs.swapaxes(0,1).swapaxes(1,2).reshape([b, -1, t, 1, 1])
    #return inputs.swapaxes(0, 1).reshape(b, -1, t, 32, 32)

def conv3Dpad(outputs, shape):
    shape = list(shape)
    oshape = list(outputs.shape)
    if shape[2] != outputs.shape[2]:
        #print(shape[2])
        #print(outputs.shape[2])
        assert(shape[2] > outputs.shape[2])
        shape[1] = oshape[1] # set channel
        shape[2] = shape[2] - outputs.shape[2]
        try:
            assert(shape[2] <= 15)
        except:
            print('shape : ', shape[2])
            assert(shape[2] <= 15)
        concat = [outputs, nd.zeros(shape, outputs.context)]
        outputs = nd.concat(*concat, dim = 2)
    return outputs

def conv3Dpad_w(outputs, shape):
    shape = list(shape)
    oshape = list(outputs.shape)
    #print('target : ', shape)
    #print('current : ', oshape)
    if shape[2] != outputs.shape[2]:
        #print(shape[2])
        #print(outputs.shape[2])
        assert(shape[2] > outputs.shape[2])
        shape[1] = oshape[1] # set channel
        shape[2] = shape[2] - outputs.shape[2]
        shape[4] = oshape[4]# - outputs.shape[4]
        try:
            assert(shape[2] <= 30)
            #assert(shape[4] <= 15)
        except:
            print('shape[2] : ', shape[2])
            print('shape[4] : ', shape[4])
            assert(shape[2] <= 30)
            #assert(shape[4] <= 15)
        concat = [outputs, nd.zeros(shape, outputs.context)]
        outputs = nd.concat(*concat, dim = 2)
        #shape[2] = oshape[2]
        #shape[4] = shape[4] - outputs.shape[4]
        #concat = [outputs, nd.zeros(shape, outputs.context)]
        #outputs = nd.concat(*concat, dim = 4)
    #print('outputs : ', outputs.shape)
    return outputs

class D3(Block):
    def __init__(self, kernel, stride, layers, feature, dilation, device = None,
            last = True, flatten = False, auto = False, mirror = True, norm = False, arch = '', reconstruct = False):
        # Mirror = True : encoder == decoder 
        super(D3, self).__init__()
        self.arch = arch; self.reconstruct = reconstruct
        self.auto = auto; self.norm = norm; self.layers = layers; self.nd = dict(); self.feature = feature;
        self.block = dict()
        self.block['norm'] = nn.LayerNorm if 'ln' in arch else nn.BatchNorm
        if auto:
            self.current_layers = layers
            self.nd['layers'] = mx.nd.array([layers], ctx = device)
            self.longest = layers
        with self.name_scope():
            self.tanh = nn.Activation('tanh'); self.sigmoid = nn.Activation('sigmoid')
            self.relu = nn.Activation('relu'); self.lrelu = nn.LeakyReLU(0.1)
            self.fc = nn.Dense(1, flatten = flatten)
            if self.auto:
                self.lp = mx.gluon.nn.Sequential() # layer predictor
                with self.lp.name_scope():
                    self.lp.add(mx.gluon.rnn.LSTM(int(feature / 8), dropout = 0.5, bidirectional = True))
                    self.lp.add(mx.gluon.nn.Dense(1, flatten = True))
            self.dropout = nn.Dropout(0.5)
            self.encoder = []; self.decoder = []; self.reconstructor = []
            self.enorm = []; self.dnorm = []; self.rnorm = []
            self.add(layers, kernel, stride, dilation, int(feature / 4))
    def forward(self, inputs):
        t, b, f = inputs.shape
        if self.auto:
            self.check()
            self.lp(inputs)
        d3out = rnn2conv(inputs)#inputs.swapaxes(0,1).swapaxes(1,2).reshape([b, -1, t, 1, 1]) # original
        shapes = []; eos = []; dos = []; debug = False
        encoder = self.encoder if not self.norm else zip(self.encoder, self.enorm)
        for e in encoder:
            shapes.append(d3out.shape); eos.append(d3out)
            eo = e(d3out) if type(e) != tuple else e[0](rnn2conv(e[1](conv2rnn(d3out))))
            d3out_sigmoid = self.sigmoid(eo); d3out_tanh = self.tanh(eo)
            d3out_current = d3out_sigmoid * d3out_tanh
            if 'dense' in self.arch:
                d3out = mx.nd.concat(*[d3out_current, d3out], dim = 1)
            else:
                d3out = d3out_current
        rout = d3out if self.reconstruct else None # compressed
        decoder = self.decoder if not self.norm else zip(self.decoder, self.dnorm)
        for i, d in enumerate(decoder):
            do = d(d3out) if type(d) != tuple else d[0](rnn2conv(d[1](conv2rnn(d3out))))
            if 'unet' in self.arch or 'dense' in self.arch or 'encoder' in self.arch :
                padded = conv3Dpad_w(do, shapes[-i - 1])
            else:
                padded = do
            d3out_sigmoid = self.sigmoid(padded); d3out_tanh = self.tanh(padded)
            d3out_current = self.lrelu(padded)#d3out_sigmoid * d3out_tanh
            if 'dense' in self.arch:
                d3out = mx.nd.concat(*[d3out_current, d3out], dim = 1)
            elif 'unet' in self.arch:
                if i != len(self.decoder):
                    d3out = (mx.nd.concat(*[d3out_current, eos[-i -1]], dim = 1))
            else:
                d3out = d3out_current
        reconstructor = self.reconstructor if not self.norm else zip(self.reconstructor, self.rnorm)
        if self.reconstruct:
            for i, r in enumerate(reconstructor):
                ro = r(rout) if type(r) != tuple else r[0](rnn2conv(r[1](conv2rnn(rout))))
                if 'dense' in self.arch or 'unet' in self.arch or 'encoder' in self.arch:
                    padded = conv3Dpad_w(ro, shapes[-i - 1])
                else:
                    padded = ro
                rout_current = self.lrelu(padded)
                if 'dense' in self.arch:
                    rout = mx.nd.concat(*[rout_current, rout], dim = 1)
                else:
                    rout = rout_current
        unswap = conv2rnn(d3out)#d3out.swapaxes(1,2).swapaxes(0,1).reshape([d3out.shape[2], b, -1])
        output = (self.fc(self.dropout(unswap)))
        if self.reconstruct:
            rout = conv2rnn(rout)#rout.swapaxes(1,2).swapaxes(0,1).reshape([rout.shape[2], b, -1])
            return output, rout
        else:
            return output
        
    def check(self):
        self.last_layers = self.current_layers
        self.current_layers = max(int(self.nd['layers'].asnumpy()), 1)
        if self.current_layers > self.longest:
            self.initialize()
            self.longest = self.current_layers
            self.add_layer(self.current_layer - self.last_layer, self.feature / 2)
    def add(self, layers, kernel, stride, dilation, channel):
        for l in range(layers):
            self.encoder.append(nn.Conv3D(channel, 
                kernel_size = [kernel, 1, 1],
                strides = [stride, 1, 1],
                padding = [dilation, 0, 0],
                dilation = [dilation, 1, 1]))
            conv = nn.Conv3D if 'bottleneck' in self.arch else nn.Conv3DTranspose
            self.decoder.insert(0, conv(channel, 
                kernel_size = [kernel, 1, 1],
                strides = [stride, 1, 1],
                padding = [dilation, 0, 0],
                dilation = [dilation, 1, 1]))
            self.register_child(self.encoder[-1])
            self.register_child(self.decoder[0])
            if self.reconstruct:
                assert('bottleneck' not in self.arch)
                assert('encoder' in self.arch and 'decoder' in self.arch)
                channel = channel if l != layers - 1 else self.feature
                self.reconstructor.append(nn.Conv3DTranspose(channel,
                    kernel_size = [kernel, 1, 1],
                    strides = [stride, 1, 1],
                    padding = [dilation, 0, 0],
                    dilation = [dilation, 1, 1]))
                self.register_child(self.reconstructor[-1])
            if self.norm:
                self.enorm.append(self.block['norm'](axis = 2))
                self.dnorm.append(self.block['norm'](axis = 2))
                self.register_child(self.enorm[-1])
                self.register_child(self.dnorm[-1])
                if self.reconstruct:
                    self.rnorm.append(self.block['norm'](axis = 2))
                    self.register_child(self.rnorm[-1])

    def pruning(self):
        pass

