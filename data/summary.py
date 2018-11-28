import math
import numpy as np
from .knapsack import knapsack_dp

def get_summary(score, boundary, nframes, fpsegment, positions, proportion = 0.15, subsampled = False, **kwargs):
    n_segs = boundary.shape[0] # shot boundary 的數量
    frame_scores = np.zeros((nframes), dtype = np.float32) # 生成一個原始長度的 vector
    if positions.dtype != int:
        positions = positions.astype(np.int32)
    if positions[-1] != nframes:
        positions = np.concatenate([positions, [nframes]])
    for i in range(len(positions) - 2):
        pos_left, pos_right = positions[i], positions[i+1]
        frame_scores[pos_left : pos_right] = score[i] # 把 subsampling 對應的位置填上一樣的 Importance Score
    seg_score = []
    for seg_idx in range(n_segs): # Change Points 也就是 shot boundary
        start, end = int(boundary[seg_idx,0]),int(boundary[seg_idx,1]+1)
        scores = frame_scores[start : end] # 把 shot 裡的分數取出來
        seg_score.append(float(scores.mean())) # 做平均
    limits = int(math.floor(nframes * proportion)) 
    picks = knapsack_dp(seg_score, fpsegment, n_segs, limits) # 找出前 15 % 並把選出來的 shots 存在變數 picks
    summary = np.zeros((1), dtype = np.float32) # this element should be deleted
    for seg_idx in range(n_segs):
        nf = fpsegment[seg_idx]
        if seg_idx in picks: # 如果該 shot 在 picks 裡，就設爲 1
            tmp = np.ones((nf), dtype = np.float32)
        else:
            tmp = np.zeros((nf), dtype = np.float32)
        summary = np.concatenate((summary, tmp)) # 然後把每一段的 binary vector 都串在一起
    summary = np.delete(summary, 0) # delete the first element

    return summary
