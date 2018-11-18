import argparse
from data.subsampling import subsample

parser = argparse.ArgumentParser(description = '')

parser.add_argument('-d', '--dataset', type = str, help = 'dataset')

parser.add_argument('-g', '--gpu', type = int, default = 0, help = 'GPU ID')

parser.add_argument('-fps', type = int, default = 1, help = 'Target FPS')

args = parser.parse_args()

subsample(arg.dataset, args.fps, 'dataset')
