import argparse
import mxnet as mx
from data import data
from config import train, verbose

parser = argparse.ArgumentParser(description = '')
parser.add_argument('-d', type = str, help = 'dataset')
parser.add_argument('-o', type = str, default = '.', help = 'output directory')
parser.add_argument('-f', type = str, default = 'mp4', help = 'format')
parser.add_argument('-c', type = str, default = '', help = 'country')

args = parser.parse_args()

train.config['dataset'] = args.d

data = data.data(**train.config)

data.next()
