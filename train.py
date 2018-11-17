import argparse
import mxnet as mx
from data import data
from data.preprocess import preprocess_list
from config import train, verbose

parser = argparse.ArgumentParser(description = '')
parser.add_argument('-d', type = str, help = 'dataset')
parser.add_argument('-o', type = str, default = '.', help = 'output directory')
parser.add_argument('-f', type = str, default = 'mp4', help = 'format')
parser.add_argument('-c', type = str, default = '', help = 'country')

args = parser.parse_args()

train.config['dataset'] = args.d

data = data.data(**train.config)

seq, scr = data.next()

seq, scr = preprocess_list(seq, scr)

nds = dict()

nps = dict()

lr = args.LR

device = mx.gpu(args.gpu)

wave = WaveArch()

cfg = None

net, spec = build(arch, nhidden, nds, args.batch, device, 'train', args.datatype, wave = wave)

if arch != 'gan':
    net.collect_params().initialize(mx.init.Xavier(), ctx = device)
    #net.collect_params().initialize(mx.init.MSRAPrelu(), ctx = device)
else:
    net.initialize(mx.init.MSRAPrelu(), ctx = device)


