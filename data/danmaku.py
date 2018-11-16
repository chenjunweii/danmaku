import numpy as np
from bilibili.bilibili import *
from scipy.stats import zscore
from utils import normalize

key = '03fc8eb101b091fb'

def request_danmaku(cid = None, aid = None):

    cid = GetVideoInfo(aid, key).cid if cid is None else cid

    danmaku = ParseDanmuku(cid)

    t = [] # second

    for d in danmaku:

        t.append(d.t_video)

    return t # second


if __name__ == '__main__':

    aid = '20610043'

    info = GetVideoInfo('20610043', key)

    nframs = 12

    raise

    danmaku = get_danmaku(aid)

    score = danmaku2score(danmaku, info.duration)

    print(score)


