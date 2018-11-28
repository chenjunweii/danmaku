from vs.vs import vs
import argparse
import mxnet as mx
from data import data
from data.preprocess import preprocess_list
from config import train, verbose

parser = argparse.ArgumentParser(description = '')

parser.add_argument('-d', '--dataset', type = str, help = 'dataset')

parser.add_argument('-g', '--gpu', type = int, default = 0, help = 'GPU ID')

parser.add_argument('-a', '--arch', type = str, default = None, help = 'Network Architecture')

#parser.add_argument('-a', '--arch', type = str, default = None, help = 'Network Architecture')

args = parser.parse_args()

for k, v in vars(args).items():

    train.config[k] = v

s = vs(**train.config)

s.Train()
