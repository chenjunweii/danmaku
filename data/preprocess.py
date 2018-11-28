import math
import numpy as np
from log import log
from config import verbose

sr = 15

def pad(seqlist, olen):

    # seqlist => batch major

    mlen = max(olen)

    nhidden = seqlist[0].shape[-1]

    for i in range(len(seqlist)):

        padding = np.zeros((mlen - olen[i], nhidden))

        seqlist[i] = np.vstack((seqlist[i], padding))

    return np.array(seqlist, dtype = np.float32)

def loss_mask(batch, olen):

    mask = np.zeros([batch, max(olen), 1]) # batch major

    active = sum(olen)

    for b in range(batch):
        
        mask[b, : olen[b] - 1] = 1

    return mask.swapaxes(0,1), active # time major

def unpad(seqnd, olen):

    seqlist = []

    batch_major_tensor = np.swapaxes(seqnd.asnumpy(), 0, 1)

    for i in range(len(olen)):

        seqlist.append(batch_major_tensor[:olen[i]])

    return seqlist

def preprocess_list(seq_list, scr_list, bd_list = None, sr = sr):

    pseq_list = []; pscr_list = [];

    for seq, scr in zip(seq_list, scr_list):

        pseq, pscr = preprocess(seq, scr = scr, sr = sr)

        pseq_list.append(pseq); pscr_list.append(pscr)

    return pseq_list, pscr_list

def get_positions(nframes, sr = sr):

    n = nframes - nframes % sr

    return np.array([i * sr for i in range(math.ceil(nframes / sr))])

def unprocess_list(scr_list, sr):

    oscr_list = []

    return oscr_list

def preprocess(seq, scr = None, boundary = None, sr = sr, m = 'avg'):

    olen = len(seq)

    shape = list(seq.shape)

    plen = int(olen / float(sr))

    shape[0] = plen

    pseq = np.zeros(shape)

    if scr is not None:

        shape[-1] = 1

        pscr = np.zeros(shape)

    for i in range(plen):

        pseq[i] = np.mean(seq[i * sr : (i + 1) * sr], 0)

        #pseq[i] = seq[i * sr]
        
        #pseq[i] = seq[int(((2*i + 1) * sr) / 2)]

        #pseq[i] = np.mean(seq[i * sr : (i + 1) * sr], 0)
        
        #pseq[i] = np.mean(seq[max(0, int((i - 0.5) * sr)) : int((i + 0.5) * sr)], 0)

        if scr is not None:
#
            #pscr[i] = np.mean(scr[max(0, int((i - 0.5) * sr)) : int((i + 0.5) * sr)], 0)

            pscr[i] = scr[i * sr]
            #pscr[i] = np.mean(scr[i * sr : (i + 1) * sr])
            
            #pscr[i] = scr[i * sr : (i + 1) * sr]
            
            #pscr[i] = scr[int(((2 * i + 1) * sr) / 2)]

    if scr is not None:

        return pseq, pscr

    else:

        return pseq

def unprocess(scr, olen, sr = sr):

    plen = len(scr)

    oscr = np.zeros([olen, 1])

    for i in range(plen):
        
        #oscr[i * sr : (i + 1) * sr] = scr[i]
        oscr[int((i - 0.5) * sr) : int((i + 0.5) * sr)] = scr[i]
        #pscr[i] = scr[int(((2*i + 1) * sr) / 2)]

    return oscr


def preprocess_user_score(uscr, sr = 15):

    olen = uscr.shape[-1]

    plen = int(olen / float(sr))

    pscr = np.zeros([plen])

    print('uscr : ', uscr)

    gtscr = np.mean(uscr / 5, axis = 0)

    print('gt_scr : ', gtscr)

    for i in range(plen):

        pscr[i] = np.mean(gtscr[int((max(0, i - 0.5)) * sr) : int((i + 0.5) * sr)])

    return pscr



def preprocess_user_score_paper(uscr, cp, nframes, nfps, sr = 15, proportion = 0.15):

    n_segs = cp.shape[0] # shot boundary 的數量
    
    uscr_shot = []

    shot_score = []

    limits = int(math.floor(nframes * proportion))
    
    for u in uscr:

        us = [] #

        for shot in cp:

            us.append(np.mean(u[shot[0] : shot[1]]))

        uscr_shot.append(us)

        picks = knapsack_dp(us, nfps, n_segs, limits) # 找出前 15 % 並把選出來的 shots 存在變數 picks
        
        summary = np.zeros((1), dtype = np.float32) # this element should be deleted
        
        for seg_idx in range(n_segs):
        
            nf = nfps[seg_idx]
            
            if seg_idx in picks: # 如果該 shot 在 picks 裡，就設爲 1
            
                tmp = np.ones((nf), dtype = np.float32)
            
            else:
            
                tmp = np.zeros((nf), dtype = np.float32)
            
            summary = np.concatenate((summary, tmp)) # 然後把每一段的 binary vector 都串在一起

        summary = np.delete(summary, 0) # delete the first element

        shot_score.append(summary)

    uscr_shot = np.array(uscr_shot)

    shot_score = np.array(shot_score)

    shot_score = shot_score.mean(axis = 0)
    
    print('uscr_shot : ', uscr_shot.shape)

    olen = uscr.shape[-1]

    plen = int(olen / float(sr))

    pscr = np.zeros([plen])

    print('shot_score : ', shot_score.shape)

    for i in range(plen):

        #print(shot_score)

        #pscr[i] = np.mean(shot_score[int((max(0, i - 0.5)) * sr) : int((i + 0.5) * sr)])
        pscr[i] = shot_score[int(i * sr)]#np.mean(shot_score[int((max(0, i - 0.5)) * sr) : int((i + 0.5) * sr)])

    #print(pscr[:100])

    return pscr

    raise

    n_segs = cps.shape[0] # shot boundary 的數量
    
    frame_scores = np.zeros((n_frames), dtype = np.float32) # 生成一個原始長度的 vector

    if positions.dtype != int:
    
        positions = positions.astype(np.int32)
    
    if positions[-1] != n_frames:
        
        positions = np.concatenate([positions, [n_frames]])
    
    for i in range(len(positions) - 1):

        pos_left,pos_right = positions[i],positions[i+1]

        frame_scores[pos_left:pos_right] = ypred[i] # 把 subsampling 對應的位置填上一樣的 Importance Score

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

