import os
import h5py
import numpy as np
import mxnet as mx
from evaluate import evaluation
from evaluate.evaluater import evaluater
from net.build import build
class init(object):

    def __init__(self):

        pass

    def Init(self):

        self.nds = dict(); self.nps = dict();

        self.net = build(self.arch, self.batch, self.feature, self.device, 'train', self.nhidden, self.nds)
        
        self.net.initialize(mx.init.MSRAPrelu(), ctx = self.device)
        
        fscore = []
        
        precision = []
        
        recall = []

        steps = []

        s = 0

        if self.checkpoint is not None:

            print ('[*] Restore From CheckPoint => {}'.format(self.checkpoint))

            self.net.load_params(self.checkpoint, ctx = device)
                
            s = int(self.checkpoint.split('_')[-1].split('.')[0])
            
            savelogpath = os.path.join("log", arch, '{}_{}.h5'.format(prefix, s))
            
            h5 = h5py.File(savelogpath, 'r')

            steps = list(np.asarray(h5['step']))

            fscore = list(np.asarray(h5['f-score']))
            
            self.lr = float(np.asarray(h5['lr']).astype(float))

        lr_scheduler = mx.lr_scheduler.FactorScheduler(self.lr_decay_step, self.lr_decay_rate)

        if 'gan' in self.arch:

            G_optimizer = 'rmsprop'
                
            G_options = {'learning_rate': self.lr,
                       'lr_scheduler' : lr_scheduler,
                       'clip_gradient': 0.01,
                       #'momentum' : 0.9,
                       'wd' : 0.0001}
            
            D_optimizer = 'rmsprop'
            
            D_options = {'learning_rate': self.lr,
                       'lr_scheduler' : lr_scheduler,
                       'clip_gradient': 0.01,
                       #'momentum' : 0.9,
                       'wd' : 0.0001}

            self.net.set_optimizer(G_optimizer, G_options, D_optimizer, D_options)
        
        else:

            optimizer = 'adam'
            
            options = {'learning_rate': self.lr,
                       'lr_scheduler' : lr_scheduler,
                       'clip_gradient': 0.01,
                       #'momentum' : 0.9,
                       'wd' : 0.0001}
        
            self.trainer = mx.gluon.Trainer(self.net.collect_params(), optimizer, options)

        print ('[*] Start Training ...')

        print('[*] Evaluation Metric : {}'.format(self.metric.title()))

        self.evaluater = evaluater(**vars(self))

        setattr(self.data, 'evaluater', self.evaluater)


        
