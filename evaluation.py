import io
import os
import sys
import h5py
import argparse
import numpy as np
import mxnet as mx
from mxnet import nd
from scipy import io
from smooth import smooth
from RL_utils import read_json, write_json
import math
import time
#from RL_vsum_tools import generate_summary
from RL_knapsack import *
#from pylab import plot, show, title, xlabel, ylabel, subplot, savefig, clf, subplots_adjust, ylim, xlim

import matplotlib.pyplot as plt

from preprocess import *

from vsumm2shot import *

def generate_summary_raw(ypred, cps, n_frames, nfps, positions, proportion = 0.15):
    """ Generate keyshot-based video summary i.e. a binary vector
    ypred: predicted importance scores.
    cps: change points, 2D matrix, each row contains a segment.
    n_frames: original number of frames.
    nfps: number of frames per segment.
    positions: positions of subsampled frames in the original video.
    proportion: length of video summary (compared to original video length).
    """
    n_segs = cps.shape[0] # shot boundary 的數量
    
    frame_scores = ypred#np.zeros((n_frames), dtype = np.float32) # 生成一個原始長度的 vector

    seg_score = []

    for seg_idx in range(n_segs): # Change Points 也就是 shot boundary

        start,end = int(cps[seg_idx,0]),int(cps[seg_idx,1]+1)

        scores = frame_scores[start:end] # 把 shot 裡的分數取出來

        seg_score.append(float(scores.mean())) # 做平均

    limits = int(math.floor(n_frames * proportion)) 

    picks = knapsack_dp(seg_score, nfps, n_segs, limits) # 找出前 15 % 並把選出來的 shots 存在變數 picks

    summary = np.zeros((1), dtype = np.float32) # this element should be deleted
    
    for seg_idx in range(n_segs):
    
        nf = nfps[seg_idx]
        
        if seg_idx in picks: # 如果該 shot 在 picks 裡，就設爲 1
        
            tmp = np.ones((nf), dtype = np.float32)
            #start,end = int(cps[seg_idx,0]),int(cps[seg_idx,1]+1)

            #summary[start : end] = 1
        
        else:
        
            tmp = np.zeros((nf), dtype = np.float32)
        
        summary = np.concatenate((summary, tmp)) # 然後把每一段的 binary vector 都串在一起

    summary = np.delete(summary, 0) # delete the first element
    
    return summary

def generate_summary(ypred, cps, n_frames, nfps, positions, proportion = 0.15):
    """ Generate keyshot-based video summary i.e. a binary vector
    ypred: predicted importance scores.
    cps: change points, 2D matrix, each row contains a segment.
    n_frames: original number of frames.
    nfps: number of frames per segment.
    positions: positions of subsampled frames in the original video.
    proportion: length of video summary (compared to original video length).
    """
    n_segs = cps.shape[0] # shot boundary 的數量
    
    frame_scores = np.zeros((n_frames), dtype = np.float32) # 生成一個原始長度的 vector

    if positions.dtype != int:
    
        positions = positions.astype(np.int32)
    
    if positions[-1] != n_frames:
        
        positions = np.concatenate([positions, [n_frames]])

    for i in range(len(positions) - 2):

        pos_left,pos_right = positions[i],positions[i+1]

        frame_scores[pos_left : pos_right] = ypred[i] # 把 subsampling 對應的位置填上一樣的 Importance Score

    seg_score = []

    for seg_idx in range(n_segs): # Change Points 也就是 shot boundary

        start,end = int(cps[seg_idx,0]),int(cps[seg_idx,1]+1)

        scores = frame_scores[start:end] # 把 shot 裡的分數取出來

        seg_score.append(float(scores.mean())) # 做平均

    limits = int(math.floor(n_frames * proportion)) 

    picks = knapsack_dp(seg_score, nfps, n_segs, limits) # 找出前 15 % 並把選出來的 shots 存在變數 picks

    summary = np.zeros((1), dtype = np.float32) # this element should be deleted
    
    for seg_idx in range(n_segs):
    
        nf = nfps[seg_idx]
        
        if seg_idx in picks: # 如果該 shot 在 picks 裡，就設爲 1
        
            tmp = np.ones((nf), dtype = np.float32)
        
        else:
        
            tmp = np.zeros((nf), dtype = np.float32)
        
        summary = np.concatenate((summary, tmp)) # 然後把每一段的 binary vector 都串在一起

    summary = np.delete(summary, 0) # delete the first element
    
    return summary

def evaluate_summary(machine_summary, user_summary, eval_metric = 'avg'):
    """ Compare machine summary with user summary (keyshot-based).
    machine_summary and user_summary should be binary vectors of ndarray type.
    eval_metric = {'avg', 'max'} """
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

class evaluate_while_train(object):
    
    def __init__(self, net, nds, arch, dataset, datatype, metric, device, sbd):

        dpath = os.path.join('raw_{}.h5'.format(dataset.lower())) if datatype == 'raw' else os.path.join('..', '..', 'pytorch-vsumm-reinforce', 'datasets', 'eccv16_dataset_{}_google_pool5.h5'.format(dataset.lower()))

        if datatype == 'raw':

            dpath2 = os.path.join('..', '..', 'pytorch-vsumm-reinforce', 'datasets'    , 'eccv16_dataset_{}_google_pool5.h5'.format(dataset.lower()))

        else:
            
            dpath2 = os.path.join('raw_{}.h5'.format(dataset.lower()))

        splits_path = os.path.join('..', '..', 'pytorch-vsumm-reinforce', 'datasets', '{}_splits.json'.format(dataset.lower()))

        splits = read_json(splits_path)

        split_id = 0
    
        assert split_id < len(splits), "split_id (got {}) exceeds {}".format(split_id, len(splits))

        split = splits[split_id]

        self.sbd = sbd

        self.net = net

        self.nds = nds

        self.arch = arch

        self.device = device

        self.datatype = datatype

        self.train_keys = split['train_keys']

        self.test_keys = split['test_keys']
        
        self.all_keys = self.train_keys + self.test_keys

        self.metric = metric

        if datatype != 'raw':

            if metric == 'transfer':

                self.keys = self.all_keys

            elif metric == 'canonical':

                self.keys = self.test_keys
            
            elif metric == 'augmented':

                self.keys = self.test_keys
            
            self.dataset2 = h5py.File(dpath2, 'r')

        else:
            
            if metric == 'transfer':

                self.keys = self.all_keys

            elif metric == 'canonical':

                self.keys = self.test_keys
            
            elif metric == 'augmented':

                self.keys = self.test_keys

            self.dataset2 = h5py.File(dpath2, 'r')

        self.eval_metric = 'avg' if dataset == 'TVSum' else 'max'
        
        self.dataset = h5py.File(dpath, 'r')

    def evaluate(self, net, nds):

        self.net = net
        
        self.nds = nds

        fms = []
        
        rs = []
        
        ps = []

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

            if self.arch == 'ewd' or self.arch == 'mpewd':

                probs = self.net(self.nds['input'], [self.nds['test_encoder_state_h'], self.nds['test_encoder_state_c']])
            
            elif self.arch == 'dpp':

                probs = self.net(self.nds['input'], None, 'test')
            
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
                
                #user_summary = self.dataset2[key]['user_summary_{}'.format(self.sbd)][...]

                #machine_summary = generate_summary(probs.asnumpy().flatten(), cps_custom, num_frames, nfps_custom, positions)

            else:

                cps = self.dataset2[key]['change_points'][...]
                
                #cps_custom = self.dataset[key]['sbd_{}'.format(self.sbd)][...]
                #cps_custom = self.dataset[key]['sbd_senet'][...]

                num_frames = self.dataset2[key]['n_frames'][()]
                
                #nframes = self.dataset[key]['nframes'][()]

                nfps = self.dataset2[key]['n_frame_per_seg'][...].tolist()

                #nfps_custom = sbd2nps(cps_custom)

                positions = self.dataset2[key]['picks'][...] #generate_positions(nframes)
            
                user_summary = self.dataset2[key]['user_summary'][...]

                #print('user_summary[0] sum : ', user_summary[0].sum())

                #user_summary = self.dataset[key]['user_summary_{}'.format(self.sbd)][...]

                #user_summary = userscore2usersummary(user_score_custom, cps_custom, nframes, nfps_custom, positions)
                #print('user_summary_custom [0] sum : ', generate_summary_rawuser_summary[0].sum())
                
                #gtscore_2 = self.dataset2[key]['gtscore'][...]

                gtscore = self.dataset[key]['gtscore'][...]

                #gt_1 = user_summary.mean(0).flatten()

                #print('gt_1 : ', gt_1.shape)

                #print('user_summary : ', user_summary.shape)

                p = probs.asnumpy().flatten()

                #p = unprocess(p, olen)
        
                machine_summary = generate_summary(p, cps, num_frames, nfps, positions)
                
                #machine_summary = generate_summary(p, cps_custom, nframes_custom, nfps_custom, generate_positions(nframes))

                #machine_summary = generate_summary_raw(p, cps_custom, nframes, nfps_custom, positions)

                #for a, b in zip(cps, cps_custom):

                #    print('{} => {}'.format(a, b))
                
                #machine_summary = generate_summary_raw(p, cps, num_frames, nfps, positions)

                #print('diff : ', (np.abs(machine_summary_senet - machine_summary)).sum())

                """
                print('senet : ', machine_summary_senet.sum())
                
                print('re: ', machine_summary.sum())
                
                print('senet : ', machine_summary_senet[:100])
                
                print('re: ', machine_summary[:100])

                """
                
                #machine_summary = generate_summary_raw(gt_1, cps, num_frames, nfps, positions)
                
                #machine_summary = probs.asnumpy().flatten()

                #print('machine_summary : ', machine_summary.shape)
            
                #machine_summary_select = machine_summary
                
                #user_summary = self.dataset2[key]['user_summary'][...] # mean = 0.15 # first 15 % then mean

                #gt_summary = self.dataset2[key]['gtsummary'][...]

                #machine_summary = probs.reshape(-1).asnumpy()

                #length = len(machine_summary)

                #machine_summary = smooth(machine_summary, 101, 'flat')

                #length_smoothed = len(machine_summary)

                #border = int((length_smoothed - length) / 2)

                #machine_summary = machine_summary[border :  -border]

                #l = int(len(machine_summary) * 15 / 100.0)

                #select = machine_summary.argsort()[::-1][:l]

                #select = user_summary.mean(0).flatten().argsort()[::-1][:l]

                #machine_summary_select = np.zeros_like(machine_summary)

                #for s in select:

                #    machine_summary_select[s] = 1

                #print('user_summary : ', user_summary.mean(0)[:100])

                #print('machine_summary : ', machine_summary[:100])

                #print('machine_summary_select : ', machine_summary_select[:100])

            #gt_score = self.dataset[key]['gtscore'][...]

            #gt_summary = self.dataset[key]['gtsummary'][...]

            #print('user summary : ', user_summary.shape)

            #print('machine_summary : ', machine_summary.shape)
            fm, p, r = evaluate_summary(machine_summary, user_summary, eval_metric = self.eval_metric)
            
            fms.append(fm)
            ps.append(p)
            rs.append(r)

        return fms, ps, rs
