import cv2
import subprocess
import numpy as np
from log import log
from config import verbose

@ log(verbose = verbose)

def hecate(f, capture = None, nframes = None):
    
    o = subprocess.check_output(
            'hecate -i {} --print_shot_info'.format(f),
            stderr = subprocess.STDOUT,
            shell = True)

    o = str(o)

    o = o.split('shots: ')[-1]

    strs = o.split(',')

    shots = []

    capture = cv2.VideoCapture(f) if capture is None else capture

    nframes = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if nframes is None else nframes # get nframes

    for i, s in enumerate(strs):

        start, end = s.split(':')

        start = int(start.split('[')[-1])

        end = int(end.split(']')[0])

        if i != 0:

            shots[-1][-1] = start - 1

        shots.append([start, end])

    shots[-1][-1] = nframes - 1
    
    #for s in shots:

    #    print(s)

    return np.asarray(shots)

if __name__ == '__main__':

    i = 'TVSum/video/-esJrBWj2d8.mp4'

    print(hecate(i))
