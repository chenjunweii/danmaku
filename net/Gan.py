import mxnet as mx
import numpy as np
from random import choice
from mxnet.gluon import nn, Block, rnn, contrib
from mxnet import nd, autograd
from mxnet.ndarray.random import *

from .d3 import D3
from .d2 import D2
from .network import ce, mse, birnn
from .Discriminator import Discriminator
from .srnn import srnn, desrnn, endsrnn
from .srnnseq import srnnseq, desrnnseq, endsrnnseq
from .wave2d import *

def gan_mse(p, g, device):

    #return (p - mx.nd.ones_like(p, ctx = device)) ** 2 if g == 'real' else (p - mx.nd.zeros_like(p, ctx = device)) ** 2
    #return mx.nd.abs(p - mx.nd.ones_like(p, ctx = device)) if g == 'real' else mx.nd.abs(p - mx.nd.zeros_like(p, ctx = device))

    g = mx.nd.ones_like(p) if g == 'real' else mx.nd.zeros_like(p)

    return (p - g)
            
    #g = mx.nd.ones_like(p) + mx.nd.random.normal(loc = 0, scale = 0.1, ctx = device) if g == 'real' else mx.nd.zeros_like(p) + \
    #         mx.random.normal(loc = 0, scale = 0.1, ctx = device)
    return -nd.clip(g, 0, 1) * nd.log(nd.clip(p, 1e-5, 1)) - (1 - nd.clip(g, 0, 1)) * nd.log(nd.clip(1 - p, 1e-5, 1))

    #return nd.abs(p - g)

def gan_ce(p, g, device):

    #return (p - mx.nd.ones_like(p, ctx = device)) ** 2 if g == 'real' else (p - mx.nd.zeros_like(p, ctx = device)) ** 2
    return mx.nd.abs(p - mx.nd.ones_like(p, ctx = device)) if g == 'real' else mx.nd.abs(p - mx.nd.zeros_like(p, ctx = device))


def l1_loss(p, g):

    return mx.nd.abs(p - g)

class Gan(Block):
    def __init__(self, G, D, device, arch = ''):
        super(Gan, self).__init__()
        self.true = None; self.false = None
        self.device = device
        self.noise = [#gamma,
                      normal]
                      #poisson,
                      #uniform]#,
                      #exponential]
                      #negative_binomial,
                      #generalized_negative_binomial]
        with self.name_scope():

            if 'd3' in arch:
                self.G = D3(**G)
                self.D = Discriminator(D, arch = 'd3-lstm') # Bi LSTM, if d3 => stride = 2
                self.G_config = G
                self.D_config = D
            elif 'd2' in arch:
                self.G = Wave2DED(*G)
                self.D = Discriminator(D, arch = 'd2-lstm') # Bi LSTM, if d3 => stride = 2
                self.G_config, G_config_Decoder = G
                self.D_config = D

            elif 'lstm' in arch:
                self.G = birnn([256], True)
                self.D = Discriminator([256], arch)
                self.G_config = {'reconstruct' : False}

            elif 'srnn' in arch:

                self.G = endsrnn()
                self.D = Discriminator([256], arch)
                self.G_config = dict()
                
                self.G_config['reconstruct'] = False

            elif 'srnnseq' in arch:

                self.G = endsrnnseq()
                self.D = Discriminator([256], arch)
                self.G_config = dict()
                
                self.G_config['reconstruct'] = False

    def initialize(self, initializer, ctx):
        self.G.collect_params().initialize(initializer, ctx)
        self.D.collect_params().initialize(initializer, ctx)
    def set_optimizer(self, G_optimizer, G_options, D_optimizer, D_options):
        self.trainerG = mx.gluon.Trainer(self.G.collect_params(), G_optimizer, G_options)
        self.trainerD = mx.gluon.Trainer(self.D.collect_params(), D_optimizer, D_options)
    def forward(self, inputs, groundtruth, proportion, mode, step = 0):
        # random inputs 從別的影片拿 # => 當做 Pretraining => 從其他影片來做 Feature Extraction, 截到相同長度
        # groundtruth : gt scores
        # inputs + Noise
        if mode == 'test':
            if self.G_config['reconstruct']:
                fake_scores, reconstruct = self.G(inputs)
            else:
                fake_scores = self.G(inputs)
            return fake_scores
        elif mode == 'train':

            D_loss_real = []; D_loss_fake = []
            with autograd.record():
                if self.G_config['reconstruct']:
                    fake_scores, reconstruct = self.G(inputs)
                else:
                    fake_scores = self.G(inputs)
                t, b, f = fake_scores.shape
                #D_loss_real = (gan_mse(self.D(inputs, groundtruth), 'real', self.device))
                #D_loss_fake = (gan_mse(self.D(inputs, fake_scores.detach()), 'fake', self.device))
                D_loss_real = self.D(inputs, groundtruth)
                D_loss_fake = self.D(inputs, fake_scores.detach()) * -1
                D_loss = (D_loss_real + D_loss_fake)
                D_loss.backward()

            if step % 25 == 0:
                self.trainerD.step(b)
        
            with autograd.record():
                fake_scores = self.G(inputs)
                t, b, f = fake_scores.shape
                #G_loss = (gan_mse(self.D(inputs, fake_scores), 'real', self.device)) # hope scores can be classify as true
                G_loss = (self.D(inputs, fake_scores))# * -1
                G_loss.backward()
            self.trainerG.step(b)
            return G_loss, D_loss
