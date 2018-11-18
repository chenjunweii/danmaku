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
import matplotlib.pyplot as plt

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

    elif eval_metric == 'max':
        
        final_f_score = np.max(f_scores)
        max_idx = np.argmax(f_scores)
        final_prec = prec_arr[max_idx]
        final_rec = rec_arr[max_idx]
    
    return final_f_score, final_prec, final_rec
