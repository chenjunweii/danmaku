from mxnet import nd, gluon
from mxnet.gluon import nn, Block, rnn, contrib

def list2rnn(inputs):
    b, f = inputs[0].shape
    return nd.concat(*inputs, dim = 1).reshape(-1, b, f)

class desrnnseq(Block):
    def __init__(self, hidden, stride = 2, layers = 1):
        super(desrnnseq, self).__init__()
        self.desrnns = []
        self.norms = []
        self.fcs = []
        with self.name_scope():
            for l in range(layers):
                self.desrnns.append(rnn.GRUCell(hidden))
                #self.norms.append(nn.LayerNorm(axis = 2))
                #self.register_child(self.norms[-1])
                self.register_child(self.desrnns[-1])
        self.hidden = hidden
        self.stride = stride
        self.layers = layers
    def forward(self, inputs, state = None, length = None):
        t, b, f = inputs.shape
        c = nd.zeros([b, self.hidden], ctx = inputs.context)
        state = [c]#, h]
        outputs = inputs#self.norms[0](inputs)
        for s in state:
            s[0].detach();# s[1].detach()
        for l in range(self.layers):
            currents = []
            in_length = len(outputs)
            out_length = in_length * self.stride
            for _t in range(out_length):
                anchor = int(_t / self.stride)
                if anchor == 0:
                    output, state = self.desrnns[l](outputs[anchor], state)
                else:
                    if anchor + self.stride > len(outputs):
                        end = None; start = - self.stride
                    elif anchor - self.stride < 0:
                        start = 0; end = start + self.stride
                    else:
                        start = anchor; end = anchor + self.stride # dilation 
                    output, state = self.desrnns[l](outputs[start:end].mean(0), state)
                currents.append(output)
            outputs = list2rnn(currents[:length])
            #outputs = self.norms[l+1](outputs if l != self.layers - 1 else outputs
        return outputs

class endsrnnseq(Block):
    def __init__(self):
        super(endsrnnseq, self).__init__()
        self.en = srnnseq(256, stride = 2, layers = 2)
        self.de = desrnnseq(256, stride = 2, layers = 2)
        #self.fc = nn.Dense(1, 'sigmoid', flatten = False)
    def forward(self, inputs, state = None):
        o = self.en(inputs)
        #o = self.de(o, None)
        o = self.de(o, None, len(inputs))
        #o = self.fc(o)
        return o

class srnnseq(Block):
    def __init__(self, hidden, stride = 2, layers = 1):
        super(srnnseq, self).__init__()
        self.srnns = []
        self.norms = []
        with self.name_scope():
            for l in range(layers):
                self.srnns.append(rnn.GRU(hidden))
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
        state = [c]
        outputs = inputs#self.norms[0](inputs)
        for s in state:
            s[0].detach();# s[1].detach()
        for l in range(self.layers):
            currents = []
            outputs = self.srnns[l](outputs)
            for _t, x in enumerate(outputs):
                if _t % self.stride == 0:
                    currents.append(x)
            outputs = list2rnn(currents)
        return outputs 

if __name__ == '__main__':
    
    s = srnnseq(100, 2, 3)
    ds = desrnnseq(100, 2, 3)

    s.collect_params().initialize()
   
    ds.collect_params().initialize()

    inputs = nd.zeros([64, 7, 100])

    o = s(inputs)

    print('inputs : ', inputs.shape)

    print('compressed : ', o.shape)
    
    do = ds(o, None, len(inputs))

    print('do : ', do.shape)
