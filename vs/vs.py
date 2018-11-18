import os
import h5py
import numpy as np
import mxnet as mx
from mxnet import nd, gluon, autograd

# Local

from net.build import build
from data.data import data
from data.preprocess import preprocess_list, pad

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

        while self.data.epoch < self.epoch:
            
            seqlist, tarlist = self.data.next()
            olen = [seq.shape[0] for seq in seqlist] # original length
            self.nps['input'] = pad(seqlist, olen).swapaxes(0, 1)
            self.nds['input'] = nd.array(self.nps['input'], self.device)
            self.nps['target'] = pad(tarlist, olen).swapaxes(0, 1)
            self.nds['target'] = nd.array(self.nps['target'], self.device)

            if self.arch == 'gan':
                prediction = self.net(self.nds['input'], self.nds['target'], 0.15, 'train')
            else:
                prediction = net(nds['input'])
                pad_mask, active = loss_mask(current_batch, olen)
                loss = nd.sum(cross_entropy_2(prediction, nds['target']) * nd.array(pad_mask, self.device)) / float(np.sum(pad_mask))
            if self.arch == 'gan':
                print('-' * 20)
                print('[*] Step {} G Loss {} LR {}'.format(s, prediction[0].mean().asnumpy(), self.net.trainerG.learning_rate))
                print('[*] Step {} D Loss {} LR {}'.format(s, prediction[1].mean().asnumpy(), self.net.trainerD.learning_rate))
            else:
                print('[*] Step {} Loss {} LR {}'.format(s, loss.asnumpy(), trainer.learning_rate))


            self.evaluater.evaluate()

            #savepath = os.path.join("..", "mx", arch, '{}_{}.mx'.format(prefix, s))
            #print('[!] Checkpoint is save to {}'.format(savepath))
            #    net.save_params(savepath)

            if self.arch != 'gan':
                
                loss.backward()
            
                trainer.step(minibatch)

            s += 1
        

    def test(self):

        pass


