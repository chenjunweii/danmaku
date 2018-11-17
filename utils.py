import cv2
from log import log

import data.preprocess #.get_positions

import data.danmaku# import get_danmaku, get_score

import data.label

import data.summary

get_summary = data.summary.get_summary

get_fpsegment = data.label.get_fpsegment

get_positions = data.preprocess.get_positions

get_boundary = data.label.get_boundary

get_score = data.label.get_score

def get_nframes(fn = None, capture = None):

    capture = capture if capture is not None else cv2.VideoCapture(fn)

    return int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

def get_fps(fn = None, capture = None):
    
    capture = capture if capture is not None else cv2.VideoCapture(fn)

    return int(capture.get(cv2.CAP_PROP_FPS))

def get_duration(fn = None, capture = None):
    
    capture = capture if capture is not None else cv2.VideoCapture(fn)

    return int(get_nframes(capture = capture) / get_fps(capture = capture))

def get_capture(fn = None):

    return cv2.VideoCapture(fn)

def normalize(data):

    return (data - min(data)) / (max(data) - min(data))
