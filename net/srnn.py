from mxnet import nd, gluon
from mxnet.gluon import nn, Block, rnn, contrib

def list2rnn(inputs):
    b, f = inputs[0].shape
    return nd.concat(*inputs, dim = 1).reshape(-1, b, f)

class desrnn(Block):
    def __init__(self, hidden, stride = 2, layers = 1):
        super(desrnn, self).__init__()
        self.desrnns = []
        self.norms = []
        self.fcs = []
        with self.name_scope():
            for l in range(layers):
                self.desrnns.append(rnn.LSTMCell(hidden))
                #self.norms.append(nn.LayerNorm(axis = 2))
                #self.register_child(self.norms[-1])
                self.register_child(self.desrnns[-1])
        self.hidden = hidden
        self.stride = stride
        self.layers = layers
    def forward(self, inputs, state = None, length = None):
        t, b, f = inputs.shape
        c = nd.zeros([b, self.hidden], ctx = inputs.context)
        h = nd.zeros([b, self.hidden], ctx = inputs.context)
        state = [c, h]
        outputs = inputs#self.norms[0](inputs)
        for l in range(self.layers):
            currents = []
            ratio = self.stride ** (l + 1)
            for _t in range(t * ratio):
                anchor = int(_t / ratio)
                if _t % self.stride == 0:
                    output, state = self.desrnns[l](outputs[anchor], state)
                else:
                    if anchor + self.stride > len(outputs):
                        end = None; start = - self.stride
                    elif anchor - self.stride < 0:
                        start = 0; end = start + self.stride
                    output, state = self.desrnns[l](outputs[start:end].mean(0), state)
                currents.append(output)
            outputs = list2rnn(currents[:length])
            #outputs = self.norms[l+1](outputs if l != self.layers - 1 else outputs
        return outputs

class endsrnn(Block):
    def __init__(self):
        super(endsrnn, self).__init__()
        self.en = srnn(256, 2, 1)
        self.de = desrnn(256, 2, 1)
        self.fc = nn.Dense(1, 'sigmoid', flatten = False)
    def forward(self, inputs, state = None):
        o = self.en(inputs)
        #o = self.de(o, None)
        o = self.de(o, None, len(inputs))
        o = self.fc(o)
        return o

class srnn(Block):
    def __init__(self, hidden, stride = 2, layers = 1):
        super(srnn, self).__init__()
        self.srnns = []
        self.norms = []
        with self.name_scope():
            for l in range(layers):
                self.srnns.append(rnn.LSTMCell(hidden))
                #self.norms.append(nn.LayerNorm(axis = 2))
                self.register_child(self.srnns[-1])
                #self.register_child(self.norms[-1])
        self.hidden = hidden
        self.stride = stride
        self.layers = layers
    def forward(self, inputs, state = None):
        t, b, f = inputs.shape
        c = nd.zeros([b, self.hidden], ctx = inputs.context)
        h = nd.zeros([b, self.hidden], ctx = inputs.context)
        state = [c, h]
        outputs = inputs#self.norms[0](inputs)
        for l in range(self.layers):
            currents = []
            for _t, x in enumerate(outputs):
                output, state = self.srnns[l](x, state)
                if _t % self.stride == 0:
                    currents.append(output)
            outputs = list2rnn(currents)
            #outputs = self.norms[l+1](outputs) if l != self.layers - 1 else outputs
        return outputs 

if __name__ == '__main__':
    
    s = srnn(100, 2, 3)
    ds = desrnn(100, 2, 3)

    s.collect_params().initialize()
    ds.collect_params().initialize()

    inputs = nd.zeros([64, 7, 100])

    o = s(inputs)

    print('inputs : ', inputs.shape)

    print('compressed : ', o.shape)
    
    print('reconstruct : ', do.shape)
