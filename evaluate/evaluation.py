import io
import os
import sys
import math
import time
import h5py
import numpy as np
import mxnet as mx
from mxnet import nd
from scipy import io
from smooth import smooth
import matplotlib.pyplot as plt

from preprocess import *
from vsumm2shot import *

def evaluate_summary(machine_summary, user_summary, eval_metric = 'avg'):
    
    machine_summary = machine_summary.astype(np.float32)
    user_summary = user_summary.astype(np.float32)
    n_users,n_frames = user_summary.shape

    # binarization
    machine_summary[machine_summary > 0] = 1

    user_summary[user_summary > 0] = 1

    if abs(n_frames - len(machine_summary)) > 10:
        raise ValueError('Padding Error')
    
    if len(machine_summary) > n_frames:
        machine_summary = machine_summary[:n_frames]
    
    elif len(machine_summary) < n_frames:
        zero_padding = np.zeros((n_frames - len(machine_summary)))
        machine_summary = np.concatenate([machine_summary, zero_padding])

    f_scores = []
    prec_arr = []
    rec_arr = []

    for user_idx in range(n_users):
        gt_summary = user_summary[user_idx,:]
        overlap_duration = (machine_summary * gt_summary).sum()
        precision = np.array(overlap_duration / (machine_summary.sum() + 1e-8)).astype(np.float64)
        recall = np.array(overlap_duration / (gt_summary.sum() + 1e-8)).astype(np.float64)

        #print('overlap_duration : ', overlap_duration)
        #print('machine_summary.sum() : ', machine_summary.sum())
        #print('gt_summary.sum() : ', gt_summary.sum())
        
        if precision == 0 and recall == 0:
            f_score = 0.
        else:
            f_score = (2 * precision * recall) / (precision + recall)

        f_scores.append(f_score)
        
        prec_arr.append(precision)
        
        rec_arr.append(recall)

    if eval_metric == 'avg':

        final_f_score = np.mean(f_scores)
        final_prec = np.mean(prec_arr)
        final_rec = np.mean(rec_arr)

        #rint('final_f_score : ', final_f_score)

        #print('fscore calculate ', (2 * final_prec * final_rec) / (final_prec + final_rec))

        #assert (final_f_score == ((2 * final_prec * final_rec) / (final_prec + final_rec)))
   
    elif eval_metric == 'max':
        
        final_f_score = np.max(f_scores)
        max_idx = np.argmax(f_scores)
        final_prec = prec_arr[max_idx]
        final_rec = rec_arr[max_idx]
    
    return final_f_score, final_prec, final_rec

class evaluater(object):
    
    def __init__(self, **kwargs):

        split = splits[split_id]

        self.sbd = sbd

        self.net = net

        self.nds = nds

        self.arch = arch

        self.device = device

    def evaluate(self, net, nds):

        fms = []; rs = []; ps = []

        print('[*] Evaluate Keys : {}'.format(len(self.keys)))

        max_shots = 0

        max_nframes = 0
        
        min_shots = 10000

        min_nframes = 0
        
        for key_idx, key in enumerate(self.keys):

            seq = self.dataset[key]['features'][...]

            #scr = np.mean(self.dataset2[key]['user_summary'][...], 0)

            olen = len(seq)

            if self.datatype == 'raw':

                seq = preprocess(seq)

            tseq = np.expand_dims(seq, 0).swapaxes(0, 1) # time major

            self.nds['input'] = nd.array(tseq, self.device)

            probs = None

            elif self.arch == 'gan':

                probs = self.net(self.nds['input'], None, 0.15, 'test')

            else:
                
                #nds['encoder_state_h'] = nd.zeros([2, 1, 256], ctx = mx.gpu(0))

                #nds['encoder_state_c'] = nd.zeros([2, 1, 256], ctx = mx.gpu(0))

                #probs = self.net(self.nds['input'], [nds['encoder_state_h'], nds['encoder_state_c']])

                probs = self.net(self.nds['input'])

            if self.datatype != 'raw':

                cps = self.dataset[key]['change_points'][...]
                
                #cps_custom = self.dataset2[key]['sbd_{}'.format(self.sbd)][...]

                num_frames = self.dataset[key]['n_frames'][()]

                nfps = self.dataset[key]['n_frame_per_seg'][...].tolist()
                
                #nfps_custom = sbd2nps(cps_custom)

                #if len(cps) != len(cps_custom):

                #    raise

                positions = self.dataset[key]['picks'][...]

                user_summary = self.dataset[key]['user_summary'][...]

                machine_summary = generate_summary(probs.asnumpy().flatten(), cps, num_frames, nfps, positions)
                
            fm, p, r = evaluate_summary(machine_summary, user_summary, eval_metric = self.eval_metric)
            
            fms.append(fm)
            ps.append(p)
            rs.append(r)

        return fms, ps, rs
