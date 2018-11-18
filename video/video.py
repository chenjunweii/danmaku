import os
import cv2
import numpy as np
from utils import get_fps

def write_video(frames, filename, fps):

    h, w, c = frames[0].shape

    fourcc = cv2.VideoWriter_fourcc(*'x264')

    out = cv2.VideoWriter(filename, fourcc, fps, (w, h))

    print('[*] Video Frames To Write : {}'.format(len(frames)))

    for frame in frames:

        out.write(frame)

        cv2.imshow('Shot', frame)

    out.release()
    
    cv2.destroyAllWindows()


def concat_video_shot(filename, shots):

    pass

def generate_summary_video(filename, shots):

    pass

def split_video_shot(filename, boundary):

    nshot = len(boundary)

    nfpsegment = 0

    sidx = 0

    frames = []

    shots = []

    start, end = boundary[sidx]

    capture = cv2.VideoCapture(filename)

    splitted = filename.split('/')

    directory = splitted[:-1]

    directory = os.path.join(*directory, 'shots')

    if not os.path.isdir(directory):

        os.makedirs(directory)

    aid, ext = splitted[-1].split('.')

    ret, frame = capture.read()

    fps = get_fps(capture = capture)

    fidx = 0
    
    while True:
    
        frames.append(frame)

        if len(frames) == (end - start):

            #shots.append(frames)

            fn = os.path.join(directory, '{}_{}.{}'.format(aid, sidx, 'mkv'))
            
            write_video(frames, fn, fps)

            print('[*] Video Shot {} Write To {}'.format(sidx, fn))

            frames = []

            sidx += 1

            start, end = boundary[sidx]
        
        ret, frame = capture.read()

        fidx += 1

        if frame is None:
            
            break

    assert(len(shots) == nshot)


        