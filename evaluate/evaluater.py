import numpy as np
import mxnet as mx
from mxnet import nd
from data.preprocess import pad
from data.summary import get_summary
from .evaluation import evaluate_summary

class evaluater(object):
    
    def __init__(self, **kwargs):

        for k, v in kwargs.items():

            setattr(self, k, v)

    def evaluate(self):

        fms = []; rs = []; ps = []
        
        print('[*] Evaluate Keys : {}'.format(len(self.data.info['test'])))

        for seq, smy, aid in zip(self.data.sequence['test'], self.data.summary['test'], self.data.info['test']):

            self.nds['input'] = nd.array(seq, self.device).expand_dims(0)

            if self.arch == 'gan':

                probs = self.net(self.nds['input'], None, 0.15, 'test')

            else:
                
                probs = self.net(self.nds['input'])

            info = self.data.info[aid]

            boundary = info.boundary

            nframes = info.nframes

            fpsegment = info.fpsegment 

            positions = info.positions

            summary = smy

            prediction = get_summary(probs.asnumpy().flatten(), boundary, nframes, fpsegment, positions)
                
            fm, p, r = evaluate_summary(prediction, summary, eval_metric = self.metric)
            
            fms.append(fm)
            ps.append(p)
            rs.append(r)
           
        print('--------------------------------------')
        print('[*] {} Evaluation F-Score: {:.2f}'.format(self.dataset, np.mean(fms) * 100))
        print('[*] {} Evaluation Precision : {:.2f}'.format(self.dataset, np.mean(ps) * 100))
        print('[*] {} Evaluation Recall : {:.2f}'.format(self.dataset, np.mean(rs) * 100))

            #plot(fscore, arch, self.dataset, prefix, s, 'f-score')
            #plot(precision, arch, self.dataset, prefix, s, 'precision')
            #plot(recall, arch, self.dataset, prefix, s, 'recall')
            #plot_table(f, p, r, arch, prefix)
                    
        return fms, ps, rs
