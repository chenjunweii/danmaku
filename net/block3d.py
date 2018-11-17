from mxnet.gluon import nn, Block, rnn, contrib
from mxnet import nd
import numpy as np



class ED3(Block):

    def __init__(self, kernel, n_layers, device = None, last = True, connection = 'dense'):

        # kernel = 2D Kernel

        super(ED3, self).__init__()

    def forward():

        pass

def conv2Dpad(outputs, shape):

    shape = list(shape)

    #print('target : ', shape)

    oshape = list(outputs.shape)
    
    #print('current : ', oshape)

    if shape[3] != outputs.shape[3]:

        assert(shape[3] > outputs.shape[3])

        shape[1] = oshape[1] # set channel

        shape[3] = shape[3] - outputs.shape[3]

        try:

            assert(shape[3] < 12)

        except:

            print('shape : ', shape[3])
            
            assert(shape[3] < 10)

        concat = [outputs, nd.zeros(shape, outputs.context)]

        outputs = nd.concat(*concat, dim = 3)

    return outputs

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

            assert(shape[2] <= 15)
            
            #assert(shape[4] <= 15)

        except:

            print('shape[2] : ', shape[2])
            
            print('shape[4] : ', shape[4])
            
            assert(shape[2] <= 15)
            
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

    def __init__(self, kernel, n_layers, feature_size, device = None, last = True, connection = 'dense'):

        # kernel = 2D Kernel

        super(D3, self).__init__()
        
        self.n_layers = n_layers

        self.connection = connection

        self.d3 = []

        self.c = feature_size

        self.k = 1

        c = self.c

        k = self.k

        self.kd2 = kernel

        with self.name_scope():

            self.activation = nn.Activation('relu')
            
            self.tanh = nn.Activation('tanh')
            
            self.sigmoid = nn.Activation('sigmoid')
            
            self.relu = nn.Activation('relu')

            #self.norm = nn.BatchNorm(axis = 1)

            self.fc = nn.Dense(1, flatten = False)
            
            self.dropout = nn.Dropout(0.5)
            
            self.encoder = []

            self.decoder = []

            self.enorm = []
            
            self.dnorm = []
                
            for l in range(n_layers):

                c2 = int(c / 2)

                c4 = int(c / 4)

                stride = 1 #if l > 0 else 15

                stride_de = 1 #if l > 0 else 15
                
                dilation = 1 # 2 ** (l)

                dilation_de = 1 # ** (l + n_layers)

                g = 1

                ks = 3 # if l != 0 else stride

                ks_de = 3

                #"""
                self.encoder.append(nn.Conv3D(c2, 
                    kernel_size = [ks, 1, 1],
                    strides = [stride, 1, 1],
                    padding = [dilation, 0, 0],
                    dilation = [dilation, 1, 1],
                    groups = g))
                #"""
                
                #stride_w = 2 if l > 0 else 2
                """
                self.encoder.append(nn.Conv3D(c2, 
                    kernel_size = [ks, 1, ks],
                    strides = [1, 1, stride_w],
                    padding = [dilation, 0, 0],
                    dilation = [dilation, 1, 1],
                    groups = g))

                """
                
                channel = c4 if l == 0 else c2

                #"""
                self.decoder.insert(0, nn.Conv3DTranspose(channel, 
                    kernel_size = [ks_de, 1, 1],
                    strides = [stride_de, 1, 1],
                    padding = [dilation_de, 0, 0],
                    dilation = [dilation_de, 1, 1],
                    groups = g))
                #"""
                """ 
                #c2 = 1 if l == 0 else c2
                self.decoder.insert(0, nn.Conv3DTranspose(c2, 
                    kernel_size = [ks_de, 1, ks],
                    strides = [1, 1, 1],
                    padding = [dilation_de, 0, 0],
                    dilation = [dilation_de, 1, 1],
                    groups = g))
                """
                self.register_child(self.encoder[-1])
                
                self.register_child(self.decoder[0])

                """
                self.enorm.append(nn.LayerNorm(axis = 1))
                
                self.dnorm.append(nn.LayerNorm(axis = 1))
                
                self.register_child(self.enorm[-1])
                
                self.register_child(self.dnorm[-1])

                """
    
    def forward(self, inputs):

        t, b, f = inputs.shape

        swap = inputs.swapaxes(0,1).swapaxes(1,2).reshape([b, -1, t, self.k, self.k]) # original

        d3out = swap

        dense_list = []; shapes = []; eos = []; dos = []; 

        debug = False

        for e in self.encoder:
        #for e, n in zip(self.encoder, self.enorm):

            shapes.append(d3out.shape)

            eos.append(d3out)

            eo = e((d3out))

            if debug:

                print('eo : ', eo.shape)

            #d3out = self.activation(eo)
            
            d3out_sigmoid = self.sigmoid(eo)

            d3out_tanh = self.tanh(eo)

            # Half Dense

            #d3out = d3out_sigmoid * d3out_tanh

            # Full Dense

            #"""

            d3out_c = d3out_sigmoid * d3out_tanh

            concat_list = [d3out_c, d3out]

            d3out = (nd.concat(*concat_list, dim = 1))

            #"""

        #for i, (d, n) in enumerate(zip(self.decoder, self.dnorm)):

        for i, d in enumerate(self.decoder):

            do = d((d3out))

            if debug:

                print('do : ', do.shape)

            #do = self.activation(do)

            #d3out = conv3Dpad(do, shapes[-i - 1])# + eos[-i - 1]
            
            padded = conv3Dpad_w(do, shapes[-i - 1])# + eos[-i - 1]
            
            d3out_sigmoid = self.sigmoid(padded)

            d3out_tanh = self.tanh(padded)

            # No Dense

            #d3out = d3out_sigmoid * d3out_tanh
            
            # Full Dense

            #"""

            d3out_c = d3out_sigmoid * d3out_tanh

            concat_list = [d3out_c, d3out]

            d3out = (nd.concat(*concat_list, dim = 1))

            #"""

            # Half Dense
            
            #"""
            
            #d3out = d3out_sigmoid * d3out_tanh

            #"""

            """
            if i != len(self.decoder) - 1:

                concat_list = [d3out, eos[-i - 1]]

                d3out = (nd.concat(*concat_list, dim = 1))

            """
                    
        # droupout 

        unswap = d3out.swapaxes(1,2).swapaxes(0,1).reshape([t, b, -1])

        output = (self.fc(self.dropout(unswap)))

        if debug:

            print('---------------')
        
        return output

class Dense(Block):

    def __init__(self, arch, n_layers, device = None, last = True):

        super(Dense, self).__init__()
        
        self.n_layers = n_layers

        self.d3 = nn.Sequential()

        with self.d3.name_scope():
            
            self.norm = nn.BatchNorm(axis = 2)

    def forward(self, inputs):

        outputs = [decoder_inputs]


class D2(Block):

    def __init__(self, kernel, n_layers, feature_size, device = None, last = True, connection = 'dense'):

        # kernel = 2D Kernel

        super(D2, self).__init__()
        
        self.n_layers = n_layers

        self.connection = connection

        self.d3 = []

        self.c = feature_size

        self.k = 1

        c = self.c

        k = self.k

        self.kd2 = kernel

        with self.name_scope():

            self.activation = nn.Activation('relu')
            
            self.tanh = nn.Activation('tanh')
            
            self.sigmoid = nn.Activation('sigmoid')
            
            self.relu = nn.Activation('relu')

            #self.norm = nn.BatchNorm(axis = 1)

            self.fc = nn.Dense(1, flatten = False)
            
            self.dropout = nn.Dropout(0.5)
            
            self.pool = nn.AvgPool3D([3, 1, 1], [2, 1, 1])

            if kernel != 'x':
            
                for n in range(self.n_layers):

                    if kernel == 1:

                        tk = 1 + n * 2 # time kernel

                        tkp = n

                        self.d3.append(nn.Conv3D(c, [3,k,k], [1,1,1], [1,0,0], dilation = [1,1,1]))
                    
                    elif kernel == 32:

                        self.d3.append(nn.Conv3D(c, [3,3,3], [1,1,1], [1,1,1], dilation = [1,1,1]))

                    self.register_child(self.d3[-1])

            elif kernel == 'x':

                self.encoder = []

                self.decoder = []

                self.enorm = []
                
                self.dnorm = []
                
                for l in range(n_layers):

                    c2 = int(c / 2)

                    c4 = int(c / 2)

                    stride = 1

                    stride_de = 1

                    dilation = 2 #** (l)

                    dilation_de = 2 #** (l + n_layers)

                    g = 1

                    ks = 3

                    self.encoder.append(nn.Conv2D(c2, 
                        kernel_size = [1, ks],
                        strides = [1, stride],
                        padding = [0, dilation],
                        dilation = [1, dilation]))

                    channel = c4 if l == n_layers - 1 else c2

                    self.decoder.append(nn.Conv2D(channel, 
                        kernel_size = [1, ks],
                        strides = [1, stride_de],
                        padding = [0, dilation_de],
                        dilation = [1, dilation_de]))

                    self.register_child(self.encoder[-1])
                    
                    self.register_child(self.decoder[-1])

                    #"""
                    
                    #self.enorm.append(nn.LayerNorm(axis = 1))
                    
                    #self.dnorm.append(nn.LayerNorm(axis = 1))
                    
                    #self.register_child(self.enorm[-1])
                    
                    #self.register_child(self.dnorm[-1])

                    #"""
    
    def forward(self, inputs):

        t, b, f = inputs.shape

        swap = None

        #swap = inputs.swapaxes(0,1).swapaxes(1,2).reshape([b, -1, t, self.k, self.k]) # original
        swap = inputs.swapaxes(0,1).swapaxes(1,2).reshape([b, f, 1, t])
        #swap = inputs.swapaxes(0,1).swapaxes(1,2).reshape([b, -1, self.k, self.k, t])
        d3out = swap

        dense_list = []

        if self.kd2 == 'x':

            shapes = []

            eos = []

            dos = []

            los = []; # layer out

            for e in self.encoder:
            #for e, n in zip(self.encoder, self.enorm):
 
                shapes.append(d3out.shape)

                eos.append(d3out)

                eo = e((d3out))

                d3out_sigmoid = self.sigmoid(eo)

                d3out_tanh = self.tanh(eo)

                # Half Dense

                #d3out = d3out_sigmoid * d3out_tanh

                # Full Dense

                #"""

                d3out_c = d3out_sigmoid * d3out_tanh
                
                #los.append(d3out_c)

                concat_list = [d3out_c, d3out]

                d3out = (nd.concat(*concat_list, dim = 1))

                #"""

            #for i, (d, n) in enumerate(zip(self.decoder, self.dnorm)):

            for i, d in enumerate(self.decoder):

                do = d((d3out))


                padded = conv2Dpad(do, shapes[-i - 1])# + eos[-i - 1]
                
                d3out_sigmoid = self.sigmoid(padded)

                d3out_tanh = self.tanh(padded)

                # No Dense

                #d3out = d3out_sigmoid * d3out_tanh
                
                # Full Dense

                #"""

                d3out_c = d3out_sigmoid * d3out_tanh

                #los.append(d3out_c)

                concat_list = [d3out_c, d3out]

                d3out = (nd.concat(*concat_list, dim = 1))

            
            #d3out = nd.concat(*los, dim = 1)

                #"""

                # Half Dense
                
                #"""
                
                #d3out = d3out_sigmoid * d3out_tanh

                #if i != len(self.decoder) - 1:

                #concat_list = [d3out, eos[-i - 1]]

                #d3out = (nd.concat(*concat_list, dim = 1))
                    
                #"""

        # droupout 

        unswap = d3out.swapaxes(1,2).swapaxes(0,1).reshape([t, b, -1])

        output = (self.fc(self.dropout(unswap)))
        
        #return output

        return output
