import numpy as np
import utils
from kts.kts import vid2shots
from log import log
from .hecate import hecate

def get_score(danmaku, nframes, fps, **kwargs):

    # score in orignal length

    score = np.zeros([nframes])

    for t in danmaku:

        score[int((t - 0.5) * fps) : int((t + 0.5) * fps)] += 1

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

def get_boundary(f, capture = None, nframes = None, method = None):

    if method == 'kts':

        # f : frames

        sbd, nframes = vid2shots(f, **kwargs)

        return shot2bound(sbd, nframes)

    elif method == 'hecate':

        # f : filename

        return hecate(f, capture, nframes)
