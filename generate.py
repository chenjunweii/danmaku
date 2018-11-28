import os
import cv2
import argparse
from data.utils import load_pickle
from video.video import generate_summary_video

parser = argparse.ArgumentParser(description = '')

parser.add_argument('-d', '--dataset', type = str, help = 'dataset')

parser.add_argument('-g', '--gpu', type = int, default = 0, help = 'GPU ID')

parser.add_argument('-fo', type = str, default = 'mkv', help = 'Format Out')

parser.add_argument('-fi', type = str, default = 'mp4', help = 'Format In ')

parser.add_argument('-fps', type = int, default = 1, help = 'Target FPS')

args = parser.parse_args()

directory = os.path.join('dataset', args.dataset)

path = dict()

path['summary'] = os.path.join(directory, '{}-summary.info'.format(args.dataset))

path['summaries'] = os.path.join(directory, 'summary')

path['video'] = os.path.join(directory, 'video')

path['info'] = os.path.join(directory, '{}_fps_1.info'.format(args.dataset))

if not os.path.isdir(path['summaries']):

    os.makedirs(path['summaries'])

summaries = load_pickle(path['summary'])

infos = load_pickle(path['info'])

for k, s in summaries.items():

    fin = os.path.join(path['video'], '{}.{}'.format(k, args.fi))

    fout = os.path.join(path['summaries'], '{}.{}'.format(k, args.fo))

    generate_summary_video(fin, fout, s)


