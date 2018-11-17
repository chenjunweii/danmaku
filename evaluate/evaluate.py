import io
import os
import sys
import h5py
import pickle
import argparse
import numpy as np
import mxnet as mx
from mxnet import nd
from scipy import io
from RL_utils import read_json, write_json
import math
from evaluation import generate_summary, evaluate_summary
from optimizer.adam_opt_mx import adam_opt, build
from RL_knapsack import *
from block import WaveArch
parser = argparse.ArgumentParser(description = '')

parser.add_argument('-c', '--checkpoint', type = str,  required = True)

parser.add_argument('-m', '--metric', type = str, required = True, choices = ['tvsum', 'summe'],
                     help = "evaluation metric ['tvsum', 'summe']")    
parser.add_argument('-d', type = str,  default = '')

parser.add_argument('-t', action = 'store_true',  default = True)

parser.add_argument('-inf', type = str,  default = '')

parser.add_argument('-a', '--arch', type = str, default = 'lstm')

parser.add_argument('--save-dir', type = str, default = 'log', help = "path to save output (default: 'log/')")

parser.add_argument('--verbose', action = 'store_true', help = "whether to show detailed test results")

parser.add_argument('--save-results', action = 'store_true', help = "whether to save output results")

parser.add_argument('-p', '--prefix', type = str, default = 'default')


args = parser.parse_args()

if __name__ == '__main__':

    if args.d.lower() == 'summe' or args.d == 'tvsum':

        dpath = os.path.join('..', '..', 'pytorch-vsumm-reinforce', 'datasets', 'eccv16_dataset_{}_google_pool5.h5'.format(args.d.lower()))

    else:

        raise ValueError('[!] Dataset is not supported')

    dataset = h5py.File(dpath, 'r')

    print("Initialize dataset {}".format(args.d))

    num_videos = len(dataset.keys())

    splits_path = os.path.join('..', '..', 'pytorch-vsumm-reinforce', 'datasets', '{}_splits.json'.format(args.d.lower()))

    splits = read_json(splits_path)

    split_id = 0

    assert split_id < len(splits), "split_id (got {}) exceeds {}".format(split_id, len(splits))

    split = splits[split_id]

    train_keys = split['train_keys']

    test_keys = split['test_keys']

    print("# total videos {}. # train videos {}. # test videos {}".format(num_videos, len(train_keys), len(test_keys)))

    device = mx.gpu(0)

    #idx = (inf['idx']).astype(int)

    fscore = []
    
    # Initialize the network    

    nds = dict()

    arch = args.checkpoint.split('/')[-2]

    #config = os.path.join('config', arch, prefix + '.h5')

    #nhidden = list(np.asarray(config[nhidden]))

    wave = WaveArch()

    cfg = None

    cfgpath = os.path.join("config", arch, '{}.txt'.format(args.prefix))

    if arch == 'ewd':

        if os.path.isfile(cfgpath):

            cfg = open(cfgpath, 'rb')

            wave = pickle.load(cfg)

            wave.show()

    net, spec = build(arch, [64], nds, 1, device, 'test', wave = wave)
    
    net.collect_params().initialize(mx.init.Xavier(), ctx = device)                                                                         

    if args.checkpoint is not None:

        print ('[*] Restore From CheckPoint => {}'.format(args.checkpoint))

        net.load_params(args.checkpoint, ctx = device)

    print ('[*] Start To Evaluate ...')

    eval_metric = 'avg' if args.metric == 'tvsum' else 'max'

    fms = []

    all_keys = test_keys + train_keys

    for key_idx, key in enumerate(all_keys):

        seq = dataset[key]['features'][...]

        tseq = np.expand_dims(seq, 0).swapaxes(0, 1) # time major
    
        nds['input'] = nd.array(tseq, device)

        probs = None

        if arch == 'ewd':

            probs = net(nds['input'], [nds['encoder_state_h'], nds['encoder_state_c']])

        else:

            probs = net(nds['input'])

        cps = dataset[key]['change_points'][...]
        
        num_frames = dataset[key]['n_frames'][()]
        
        nfps = dataset[key]['n_frame_per_seg'][...].tolist()
        
        positions = dataset[key]['picks'][...]
        
        user_summary = dataset[key]['user_summary'][...]

        machine_summary = generate_summary(probs.asnumpy(), cps, num_frames, nfps, positions)
        
        fm, _, _ = evaluate_summary(machine_summary, user_summary, eval_metric)
        
        fms.append(fm)

        if args.verbose:
        
            table.append([key_idx+1, key, "{:.1%}".format(fm)])
        
        if args.save_results:
            
            h5_res.create_dataset(key + '/score', data=probs)
            
            h5_res.create_dataset(key + '/machine_summary', data=machine_summary)

            h5_res.create_dataset(key + '/gtscore', data=dataset[key]['gtscore'][...])

            h5_res.create_dataset(key + '/fm', data=fm)

    if args.verbose:
        
        print(tabulate(table))

    if args.save_results: h5_res.close()

    mean_fm = np.mean(fms)

    print("Average F-score {:.1%}".format(mean_fm))

#    for i in idx:

 #       fscore.append(evaluate_summary(inf[str(i)), )
    
