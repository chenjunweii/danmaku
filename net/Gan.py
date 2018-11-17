import mxnet as mx
import numpy as np
from random import choice
from mxnet.gluon import nn, Block, rnn, contrib
from mxnet import nd, autograd
from mxnet.ndarray.random import *

from .d3 import D3
from .d2 import D2
from .network import ce, mse
from .Discriminator import Discriminator
from .srnn import srnn, desrnn, endsrnn
from .wave2d import *

def gan_mse(p, g, device):

    #return (p - mx.nd.ones_like(p, ctx = device)) ** 2 if g == 'real' else (p - mx.nd.zeros_like(p, ctx = device)) ** 2
    #return mx.nd.abs(p - mx.nd.ones_like(p, ctx = device)) if g == 'real' else mx.nd.abs(p - mx.nd.zeros_like(p, ctx = device))

    g = mx.nd.ones_like(p) if g == 'real' else mx.nd.zeros_like(p)
            
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
    def __init__(self, G, D, device):
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
        self.G_config = G
        self.D_config = D
        with self.name_scope():
            #self.G = Wave2DED(*G)
            self.G = D3(**G)
            #self.D = Wave2D(**D)
            self.D = Discriminator(D, arch = 'd3-lstm') # Bi LSTM, if d3 => stride = 2
            #self.G = endsrnn()

    def initialize(self, initializer, ctx):
        self.G.collect_params().initialize(initializer, ctx)
        self.D.collect_params().initialize(initializer, ctx)
    def set_optimizer(self, G_optimizer, G_options, D_optimizer, D_options):
        self.trainerG = mx.gluon.Trainer(self.G.collect_params(), G_optimizer, G_options)
        self.trainerD = mx.gluon.Trainer(self.D.collect_params(), D_optimizer, D_options)
    def forward(self, inputs, groundtruth, proportion, mode, mask = None, active = None):
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
                D_loss_real.append(gan_mse(self.D(inputs, groundtruth), 'real', self.device))
                D_loss_fake.append(gan_mse(self.D(inputs, fake_scores.detach()), 'fake', self.device))
                #D_loss_fake.append(gan_mse(self.D(inputs,
                #    choice(self.noise)(loc = 0.1, scale = 0.1, shape = groundtruth.shape, ctx = self.device)), 'fake', self.device))
                #D_loss.append(gan_mse(self.D(choice(self.noise)(shape = inputs.shape, ctx = self.device),
                #   choice(self.noise)(shape = groundtruth.shape, ctx = self.device)), 'fake', self.device)) # fake summary loss
                #D_loss.append(gan_mse(self.D(choice(self.noise)(shape = groundtruth.shape, ctx = self.device), inputs), 'false', self.device))
                #D_loss.append(gan_mse(self.D(groundtruth, inputs), 'true', self.device)) # true loss
                D_loss = (sum(D_loss_real) / len(D_loss_real) + sum(D_loss_fake) / len(D_loss_fake)) / 2
            D_loss.backward()

            self.trainerD.step(b)

            G_loss = []

            with autograd.record():
                G_loss.append(gan_mse(self.D(inputs, fake_scores), 'real', self.device)) # hope scores can be classify as true
                #G_loss.append(mse(scores.mean(axis = 0), proportion))
                if self.G_config['reconstruct']:
                    G_loss.append(l1_loss(inputs, reconstruct).mean())
                G_loss = sum(G_loss) / len(G_loss)

            G_loss.backward()
            self.trainerG.step(b)
            
            #fake_scores = self.G(fake_inputs)
            
            # scores => 15 %
            

            # D Loss

            #D_loss.backward()
            
            #self.trainerD.step(b)

            return G_loss, D_loss


