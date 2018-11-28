import os
import h5py
import numpy as np
import mxnet as mx
from mxnet import nd, gluon, autograd

# Local

from net.build import build
from net.network import cross_entropy_2
from data.data import data
from data.preprocess import preprocess_list, pad, loss_mask
from .init import init

class vs(init):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.data = data(**kwargs)
        self.device = mx.gpu(self.gpu)
    def Train(self):
        self.Init()
        s = 0
        savedir = os.path.join("model", self.arch)
        if not os.path.isdir(savedir):
            os.makedirs(savedir)
        while self.data.epoch < self.epoch:
            seqlist, tarlist, nepoch = self.data.next()
            cbatch = len(seqlist)
            olen = [seq.shape[0] for seq in seqlist] # original length
            self.nps['input'] = pad(seqlist, olen).swapaxes(0, 1)
            self.nds['input'] = nd.array(self.nps['input'], self.device)
            self.nps['target'] = pad(tarlist, olen).swapaxes(0, 1)
            self.nds['target'] = nd.array(self.nps['target'], self.device)

            if 'gan' in self.arch:
                prediction = self.net(self.nds['input'], self.nds['target'], 0.15, 'train', s)
                print('-' * 20)
                print('[*] Step {} G Loss {} LR {}'.format(s, prediction[0].mean().asnumpy(), self.net.trainerG.learning_rate))
                print('[*] Step {} D Loss {} LR {}'.format(s, prediction[1].mean().asnumpy(), self.net.trainerD.learning_rate))
            else:
                with autograd.record():
                    prediction = self.net(self.nds['input'])
                    pad_mask, active = loss_mask(cbatch, olen)
                    loss = nd.mean(cross_entropy_2(prediction, self.nds['target']))# * nd.array(pad_mask, self.device)) / float(np.sum(pad_mask))
                    loss.backward()
                self.trainer.step(cbatch)
                print('[*] Step {} Loss {} LR {}'.format(s, loss.asnumpy(), self.trainer.learning_rate))
            #if s % 20 == 0:
            #    self.evaluater.evaluate()
            if nepoch and self.data.epoch % self.sepoch == 0:

                savepath = os.path.join(savedir, '{}-{}.params'.format(self.prefix, self.data.epoch))
                print('[!] Checkpoint is save to {}'.format(savepath))
            #    net.save_params(savepath)
            s += 1
        
    def test(self):

        pass


