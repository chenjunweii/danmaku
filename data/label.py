import numpy as np
import utils
from kts.kts import vid2shots, shot2bound
from log import log
from .hecate import hecate

def get_score(danmaku, nframes = None, nsamples = None, fps = None, subsampled = False, **kwargs):

    # score in orignal length
    
    if subsampled:

        score = np.zeros([nsamples])

        for t in danmaku:

            start = int(t - 1) ; end = int(t + 1)

            score[start : end] += 1

    else:

        score = np.zeros([nframes])

        for t in danmaku:

            score[int((t - 1) * fps) : int((t + 1) * fps)] += 1

    return utils.normalize(score)

def request_score(info = None, directory = None):

    assert((info is None and dir is not None) or (info is not None and dir is None))

    danmaku = request_danmaku(cid)

    return get_danmaku(danmaku)

def get_fpsegment(boundary):

    """

    sbd to number of frame per segment

    """

    return [int(s[1] - s[0] + 1) for s in boundary]

def get_boundary(f, duration = 30, fps = None, Max = None, capture = None, nframes = None, method = None, ):

    if method == 'kts':

        # f : frames

        Max = int(len(f) / (duration)) if Max is None else Max

        sbd, nframes = vid2shots(f, Max)

        return shot2bound(sbd, nframes)

    elif method == 'hecate':

        # f : filename

        return hecate(f, capture, nframes)
