import os
import numpy as np
import mxnet as mx
from mxnet import nd
from data.preprocess import pad
from data.summary import get_summary
from data.utils import load_pickle, save_pickle
from .evaluation import evaluate_summary
from .plot import plot
from copy import deepcopy

class evaluater(object):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.fhistory = [] # fscore
        self.shistory = [] # step 
        self.max = 0
    def evaluate(self):
        fms = []; rs = []; ps = []; summaries = dict()
        print('[*] Evaluate Keys : {}'.format(len(self.data.info['test'])))
        for seq, smy, aid in zip(self.data.sequence['test'], self.data.summary['test'], self.data.ids['test']):
            self.nds['input'] = nd.array(seq, self.device).expand_dims(1)
            if 'gan' in self.arch:
                probs = self.net(self.nds['input'], None, 0.15, 'test').asnumpy().flatten()
            else:
                probs = self.net(self.nds['input']).asnumpy().flatten()
            info = deepcopy(self.data.info[aid])
            info.score = probs
            pred = get_summary(**vars(info))
            summaries[aid] = pred
            assert(len(pred) == info.nframes)
            fm, p, r = evaluate_summary(pred, smy, eval_metric = self.metric)
            fms.append(fm); ps.append(p); rs.append(r)
        f = np.mean(fms) * 100;
        self.fhistory.append(f)
        self.shistory.append(self.data.epoch)
        print('--------------------------------------')
        print('[*] {} Evaluation F-Score: {:.2f}'.format(self.dataset, f))
        print('[*] {} Evaluation Precision : {:.2f}'.format(self.dataset, np.mean(ps) * 100))
        print('[*] {} Evaluation Recall : {:.2f}'.format(self.dataset, np.mean(rs) * 100))
        if f > self.max:
            self.max = f
            save_pickle(summaries, os.path.join(self.directory, self.dataset, '{}-summary.info'.format(self.dataset)))
        plot(self.fhistory, self.shistory, self.arch, self.dataset, self.prefix, self.data.epoch, 'f-score')
        #plot(precision, arch, self.dataset, prefix, s, 'precision')
        #plot(recall, arch, self.dataset, prefix, s, 'recall')
        #plot_table(f, p, r, arch, prefix)
                    
        return fms, ps, rs
