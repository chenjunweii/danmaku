import copy
from mxnet.gluon import nn, Block, rnn, contrib
from mxnet import nd
import numpy as np

class Encoder(Block):

    def __init__(self, n_inputs, n_hidden, n_layers = 1, dropout = 0.5):
        
        super(Encoder, self).__init__()
        
        with self.name_scope():
        
            self.encoder = rnn.LSTM(n_hidden, n_layers, dropout = dropout, input_size = n_inputs, bidirectional = True)

    def forward(self, inputs, states):

        all_states, last_state = self.encoder(inputs, states)

        return last_state, all_states

    def begin_state(self, *args, **kwargs):
    
        return self.rnn.begin_state(*args, **kwargs)


class EDWA(Block):

    def __init__(self, n_input, n_encoder_hidden, n_decoder_hidden, n_output, n_sequence_max, n_alignment, dropout = 0.5):
        
        super(EDWA, self).__init__()

        self.n_sequence_max = n_sequence_max
        
        with self.name_scope():
        
            self.e = (Encoder(n_input, n_encoder_hidden, 1, 0.5))

            self.d = (Decoder(n_input, n_encoder_hidden, n_decoder_hidden, n_output, n_sequence_max, n_alignment, dropout = dropout))

    def forward(self, inputs, states):
            
        t, n, c = inputs.shape

        pad = nd.zeros([self.n_sequence_max - t, n, c], ctx = inputs.context)

        inputs = nd.concat(inputs, pad, dim = 0)

        last_state, all_states = self.e(inputs, states)

        outputs = self.d(inputs, last_state, all_states, t)

        nd_outputs = outputs[0]
        
        for i in range(1, len(outputs)):

            nd_outputs = nd.concat(nd_outputs, outputs[i], dim = 0)

        return all_states

    def begin_state(self, *args, **kwargs):
    
        return self.rnn.begin_state(*args, **kwargs)


class DecoderInitState(Block):

    """解码器隐含状态的初始化"""
    
    def __init__(self, encoder_hidden_dim, decoder_hidden_dim):
        
        super(DecoderInitState, self).__init__()
        
        with self.name_scope():
        
            self.dense = nn.Dense(decoder_hidden_dim, in_units = encoder_hidden_dim, activation = "tanh", flatten = False)

    def forward(self, encoder_state):
        
        return [self.dense(encoder_state)]

class Wave(Block):

    def __init__(self,
            n_input_states,
            n_output_states,
            size,
            device,
            last = False,
            wave = None,
            nearest = False,
            config = None,
            overlap = True,
            layer = ''):

        """
        
        overlap 
            
            False : 
            
                if kernel is overlapped

                    overlap

        """
        
        super(Wave, self).__init__()

        self.device = device

        self.layer = layer

        self.size = size

        print('kernel : ', self.size['kernel'])

        self.size['context'] = int((self.size['kernel'] - 1) / 2)
        
        self.overlap = overlap

        self.last = last

        self.fc = nn.Sequential()

        self.fg = nn.Sequential()

        #self.d3 = nn.Conv2D(1, (1,1))

        #self.norm = nn.Activation('relu')#nn.BatchNorm(axis = 1)
        self.norm = nn.BatchNorm(axis = 1)
        #self.norm = nn.LayerNorm(axis = 1)

        #self.relu = nn.Activation('relu')
        #self.relu = nn.SELU()

        if last:

            self.wave = wave

            with self.fc.name_scope():

                #if last:

                self.fc.add(nn.Dropout(0.5))
                
                self.fc.add(nn.Dense(n_output_states, flatten = True))

        else:

            with self.fc.name_scope():

                #if last:

                self.fc.add(nn.Dense(n_output_states, activation = 'sigmoid', flatten = True))

                #else:

                #self.fc.add(nn.Dense(n_output_states, flatten = True))

                #self.fc.add(nn.LeakyReLU(0.2))

                #self.fc.add(nn.LayerNorm(axis = 1, center = True, scale = True))

            with self.fg.name_scope():

                    #if last:

                self.fg.add(nn.Dense(n_output_states, activation = 'tanh', flatten = True))

            #self.fg.add(nn.LayerNorm(axis = 1, center = True, scale = True))
            #else:

    @property
    
    def patch(self):

        return self.size['dilate'] * self.size['context'] * 2 # contains both direction

    def nearest_context(self, currnet_input, states, concat_list, nearest_k, current_time, total_time):

        total_time = [total_time]

        for state, wave in zip(states[1:], self.wave):

            nearest_next = current_time / wave.authorised_stride + (0 if total_time[-1] % authorised_stride > wave.context_size * wave.dilate_size else 1)

            nearest_prev = nearest_next - 1

            next_idx = (concat_list + 1)

            prev_idx = (concat_list - 1)

            concat_list.insert(prev_idx, nearest_prev)

            concat_list.insert(next_idx, nearest_next)

        return concat_list
            
    def context(self, inputs, decoder_inputs = None):

        dilated_context_size = self.size['context'] * self.size['dilate']

        element = (np.arange(- dilated_context_size, dilated_context_size + 1, self.size['dilate']) + self.current_time).astype(int)

        element_insufficient = np.take(element, np.where(element < 0)[0])

        element_exceed = np.take(element, np.where(element >= self.total_time)[0])

        element_authorised = np.take(element, np.where(np.logical_and(element < self.total_time, element >= 0))[0])

        concat_list_exceed = [] if len(element_exceed) == 0 \
                else [nd.zeros([self.n_batch, self.n_states * len(element_exceed)], self.device)]

        concat_list_authorised = [inputs[c] for c in element_authorised]

        concat_list_insufficient = [] if len(element_insufficient) == 0 \
            else [nd.zeros([self.n_batch, self.n_states * len(element_insufficient)], self.device)]

        return concat_list_insufficient + concat_list_authorised + concat_list_exceed

    def context_out(self, inputs, decoder_inputs = None):

        decoder_total_times, decoder_batch, decoder_n_states = decoder_inputs.shape

        inputs_total_times, inputs_batch, inputs_n_states = inputs.shape

        decoder_portion_idx = self.current_time / decoder_total_times

        inputs_center_idx = np.ceil(inputs_total_times * decoder_portion_idx).astype(int)

        dilated_context_size = self.size['context'] * self.size['dilate']

        #element = (np.arange(- dilated_context_size, dilated_context_size + 1, self.size['dilate']) + inputs_center_idx).astype(int)
        element = (np.arange(- dilated_context_size, dilated_context_size + 1, self.size['dilate'])).astype(int)

        element_insufficient = np.take(element, np.where(element < 0)[0])

        element_exceed = np.take(element, np.where(element >= inputs_total_times)[0])

        element_authorised = np.take(element, np.where(np.logical_and(element < inputs_total_times, element >= 0))[0])

        concat_list_exceed = [] if len(element_exceed) == 0 \
                else [nd.zeros([self.n_batch, inputs_n_states * len(element_exceed)], self.device)]

        concat_list_authorised = [inputs[c] for c in element_authorised]

        concat_list_insufficient = [] if len(element_insufficient) == 0 \
                else [nd.zeros([self.n_batch, inputs_n_states * len(element_insufficient)], self.device)]

        concat_list_authorised.insert(int(len(concat_list_authorised) / 2), decoder_inputs[self.current_time])

        return concat_list_insufficient + concat_list_authorised + concat_list_exceed

    @property

    def check_overlap(self):

        return True if self.size['stride'] <= 2 * self.size['context'] * self.size['dilate'] else False
    
    def forward(self, inputs, decoder_inputs = None):

        if decoder_inputs is None:
        
            self.total_time, self.n_batch, self.n_states = inputs.shape # if TNC

        elif self.last:

            self.total_time, self.n_batch, self.n_states = decoder_inputs.shape # if TNC

        else:

            ValueError('[!] Not Last Layer ...')

        self.current_time = 0

        if (not self.overlap) and self.check_overlap:

            self.size['authorised_stride'] = 2 * self.size['context'] * self.size['dilate'] + 1

        else:

            self.size['authorised_stride'] = self.size['stride']

        if self.last:

            assert(self.size['stride'] == 1)

            self.size['authorised_stride'] = self.size['stride']

        outputs = []

        start = 0; end = 0;

        #print('self.layer : ', self.layer)

        #print('inputs : ', inputs.shape)

        #print('[*] Authorized Stride : ', self.size['authorised_stride'])

        while self.current_time < self.total_time:

            if decoder_inputs is None:

                concat_list = self.context(inputs)

            elif self.last:

                #concat_list = self.context_out(inputs, decoder_inputs)

                concat_list = self.context(inputs)

            concat_data = nd.concat(*concat_list, dim = 1) # [time, batch, feature]

            concat_data_norm = self.norm(concat_data) # norm for context

            #batch_major = concat_data_norm.reshape([self.n_batch, 3, 1, self.n_states])

            #merge = self.d3(batch_major)

            #merge = (self.relu(merge))

            merge = concat_data_norm

            if not self.last:
                
                sigmoid_output = self.fc(merge)#.expand_dims(0)

                tanh_output = self.fg(merge)#.expand_dims(0)
            
                outputs.append((sigmoid_output * tanh_output).expand_dims(0)) # time major

            else:
                
                sigmoid_output = self.fc(merge)#.expand_dims(0)

                outputs.append((sigmoid_output).expand_dims(0)) # time major

            self.current_time += self.size['authorised_stride']

        outputs = nd.concat(*outputs, dim = 0)

        return outputs

    def begin_state(self, *args, **kwargs):
    
        return self.rnn.begin_state(*args, **kwargs)

class WaveArch(object):

    def __init__(self):

        self.inputs = None

        self.outputs = []

        self.size = []

        self.overlap = []

        self.last = []

        self.n_layers = 0

    def show(self):

        for n in range(self.n_layers):

            print('kernel : {}, stride : {}, dilate : {}'.format(self.size[n]['kernel'], self.size[n]['stride'], self.size[n]['dilate']))

class WaveDecoder(Block):

    def __init__(self, arch, n_layers, device = None, last = True):

        super(WaveDecoder, self).__init__()
        
        self.n_layers = n_layers

        self.last = last # if last layer, True => last nhidden = 1, False => last nihdden != 1

        with self.name_scope():
            
            self.decoder = nn.Sequential()
        
            self.norm = nn.BatchNorm(axis = 2)
            #self.norm = nn.LayerNorm(axis = 2)

        with self.decoder.name_scope():

            for l in range(self.n_layers - 1):

                self.decoder.add(Wave(arch.inputs[l], arch.outputs[l], arch.size[l],
                    overlap = arch.overlap[l], last = False, device = device, layer = 'wave_{}'.format(l + 1)))

            self.out = Wave(arch.inputs[-1], arch.outputs[-1], arch.size[-1],
                    overlap = arch.overlap[-1], last = last, device = device, layer = 'wave_{}'.format(self.n_layers))

    @property

    def patch(self):

        #print('out : ', self.out.patch)

        #print('prev : ', np.prod([d.patch for d in self.decoder]))

        return np.prod([d.patch for d in self.decoder]) * self.out.patch

    def forward(self, decoder_inputs, connection = 'residual'):

        outputs = [decoder_inputs]

        outputs_dense = []

        outputs_residual = []

        dense_concat_list = [decoder_inputs]

        connection = 'dense'
        #connection = 'residual'

        # 新想法 用不同 stride 或 kernel，平行下去跑，層數減少，例如第一層同時有 kernel = 3　和 5　...

        if connection == 'residual':

            for d in self.decoder:

                outputs.append(self.norm(d(outputs[-1]) + outputs[-1]))

            outputs.append(self.out(self.norm(outputs[-1] + outputs[-1]))) # + [-1] is for residual connection

        elif connection == 'dense':

            #for d, n in zip(self.decoder, self.norm):

            for idx, d in enumerate(self.decoder):

                if idx == 0:

                    dense_concat_list.append(d(self.norm(outputs[-1])))

                else:

                    dense_concat_list.append(d(self.norm(outputs[-1])))

                dense_node = (nd.concat(*dense_concat_list, dim = 2))

                outputs.append((dense_node)) # not use norm because later we may concat different pat

            outputs.append(self.out((outputs[-1])))
        
        elif connection == 'dual':

            outputs_residual.append(decoder_inputs)
            
            outputs_dense.append(decoder_inputs)

            for idx, d in enumerate(self.decoder):

                if idx == 0:

                    dense_concat_list.append(d((outputs[-1])))

                    residual.append()

                else:

                    dense_concat_list.append(d(self.norm(outputs[-1])))

                dense_node = (nd.concat(*dense_concat_list, dim = 2))

                dense_outputs.append((dense_node)) # not use norm because later we may concat different pat

            dense_outputs.append(self.out((outputs[-1])))

        else:

            for d in self.decoder:

                outputs.append(d((outputs[-1])))

            outputs.append(self.out((outputs[-1]))) # + [-1] is for residual connection

        return outputs[-1]
    
    def begin_state(self, *args, **kwargs):
    
        return self.rnn.begin_state(*args, **kwargs)

# Encoder - WaveDecoder

class EWD(Block):

    def __init__(self, n_input, n_encoder_state, n_wave_output, decoder_arch, n_layers, dropout = 0.5, device = None):
    
        super(EWD, self).__init__()

        with self.name_scope():
        
            self.e = Encoder(n_input, n_encoder_state, 1, dropout)

            self.wd = WaveDecoder(decoder_arch, n_layers, device = device)
            
            self.norm = nn.BatchNorm(axis = 2)
            #self.norm = nn.LayerNorm(axis = 2)

    @property
    
    def patch(self):

        return self.wd.patch

    def forward(self, inputs, states):

        encoder_last_states, encoder_all_states = self.e(self.norm(inputs), states)

        outputs = self.wd(encoder_all_states) # get states can still norm , norm([prev state, current state, next state])
        #outputs = self.wd(inputs)
        return outputs

class MPEWD(Block):

    def __init__(self, n_input, n_encoder_state, n_wave_output, decoder_arch, n_layers = 1, dropout = 0.5, device = None):
    
        super(MPEWD, self).__init__()

        self.wd = []

        self.n_path = decoder_arch.n_layers # 暫時用這個代替

        print('npath : ', self.n_path)

        self.wa = [None] * self.n_path

        for p in range(self.n_path):

            self.wa[p] = copy.deepcopy(decoder_arch)

            self.wa[p].size = [decoder_arch.size[p]]

            self.wa[p].n_layers = n_layers

        with self.name_scope():
        
            self.e = Encoder(n_input, n_encoder_state, 1, dropout)

            for p in range(self.n_path):

                self.wd.append(WaveDecoder(self.wa[p], n_layers, device = device, last = False))
                
                self.register_child(self.wd[p])
            
            self.norm = nn.BatchNorm(axis = 2)
            
            #self.norm = nn.LayerNorm(axis = 2)
            
            self.relu = nn.Swish()

            #self.fc = nn.Dense(1, activation = "sigmoid", flatten = False)
            self.fc = nn.Dense(1, flatten = False)

    @property
    
    def patch(self):

        return self.wd[-1].patch

    def forward(self, inputs, states, block = ''):

        encoder_last_states, encoder_all_states = self.e(self.norm(inputs), states)

        outputs = [encoder_all_states]

        block = 'dense'

        o = None

        if block == 'dense':

            for wd in self.wd:

                outputs.append(wd((outputs[0])))

            output = nd.concat(*outputs[:], dim = 2)

            normed = self.norm(output)

            o = self.fc(normed)

        return o


class WD(Block):

    def __init__(self, n_input, n_encoder_state, n_wave_output, decoder_arch, n_layers, dropout = 0.5, device = None):
    
        super(WD, self).__init__()

        with self.name_scope():
        
            self.wd = WaveDecoder(decoder_arch, n_layers, device = device)
            
            self.norm = nn.BatchNorm(axis = 2)

    @property
    
    def patch(self):

        return self.wd.patch

    def forward(self, inputs):

        outputs = self.wd(self.norm(inputs))

        return outputs

class EWDE(Block):

    def __init__(self, n_input, n_encoder_state, n_wave_output, decoder_arch, n_layers, dropout = 0.5, device = None):
    
        super(EWDE, self).__init__()

        with self.name_scope():
        
            self.e = Encoder(n_input, n_encoder_state, 1, dropout)
            
            self.e2 = Encoder(n_input, n_encoder_state, 1, dropout)

            self.wd = WaveDecoder(decoder_arch, n_layers, device = device, last = False)

            self.fc = nn.Dense(1, flatten = False)
            
            self.norm = nn.BatchNorm(axis = 2)

    @property
    
    def patch(self):

        return self.wd.patch

    def forward(self, inputs, states, states2):

        encoder_last_states, encoder_all_states = self.e(self.norm(inputs), states)

        wd_outputs = self.wd(encoder_all_states) # get states can still norm , norm([prev state, current state, next state])
        
        decoder_last_state, decoder_all_states = self.e2(outputs, states) # get states can still norm , norm([prev state, current state, next state]

        outputs = self.fc(wd_outputs)
        
        return outputs
