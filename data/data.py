import os
import cv2
import csv
import copy
import h5py
import mxnet as mx
import numpy as np
from scipy import io
from random import shuffle
from .danmaku import *
from .downloader import download_list
from .label import get_fpsegment, get_score
from .utils import *
from .summary import get_summary
from .preprocess import get_positions
from .key import split_dataset
from utils import *
from fe.fe import fe
from bilibili.bilibili import GetVideoInfo

class data(object):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.epoch = 0
        self.ctx = mx.cpu() if self.device is None else mx.gpu(self.device)
        prefix = '{}-fps-{}'.format(self.dataset, self.tfps) if self.tfps >= 1 else self.dataset
        self.path['cache'] = os.path.join(self.directory, self.dataset, '{}.h5'.format(prefix))
        self.path['info'] = os.path.join(self.directory, self.dataset, '{}.info'.format(prefix))
        self.path['video'] = os.path.join(self.directory, self.dataset, 'video')
        self.train_list = os.path.join('list', '{}.json'.format(self.dataset))
        if self.fcheck or not os.path.isfile(self.path['info']):
            self.generate(self.train_list, 'train')
            if self.test_list is not None:
                assert(self.test)
                self.generate(self.test_list, 'test')
        else:
            self.info = load_pickle(self.path['info'])
        if self.train:
            self.load('train')
        if self.test:
            self.load('test')
    def next(self, mode = 'train', shuffle = True):
        if (self.counter + self.batch) >= len(self.sequence[mode]):
            self.counter = 0
            #if shuffle:
            #    self.shuffle()
            self.epoch += 1
            self.nepoch = True
            print('[*] Epoch : ', self.epoch)
            if self.epoch % self.eepoch == 0:
                self.evaluater.evaluate()
        else:
            self.counter += self.batch
            self.nepoch = False
        return self.sequence[mode][self.counter : self.counter + self.batch], \
                    self.score[mode][self.counter : self.counter + self.batch], \
                    self.nepoch 
    def load(self, mode):
        self.sequence[mode] = []; self.nframes[mode] = [];
        self.score[mode] = []; self.summary[mode] = [];
        self.ids[mode] = []
        with h5py.File(self.path['cache'], 'r') as h5:
            if mode not in self.info or self.fsplit:
                print('[!] Dataset {} have not splitted yet'.format(self.dataset))
                self.info = split_dataset(self.info, filename = self.path['info'])
            for k in self.info[mode]:
                if 'av' in k or 'ep' in k:
                    self.sequence[mode].append(h5[k]['features'][...])
                    self.nframes[mode].append(self.info[k].nframes)
                    self.score[mode].append(np.expand_dims(self.info[k].score, 1))
                    self.check(self.sequence[mode][-1], self.score[mode][-1])
                    self.summary[mode].append(np.expand_dims(self.info[k].summary, 0))
                    self.ids[mode].append(k)

    def shuffle(self):
        l = list(zip(self.sequence['train'], self.score['train']))
        shuffle(l)
        self.sequence['train'], self.score['train'] = zip(*l)
        self.sequence['train'] = list(self.sequence['train'])
        self.score['train'] = list(self.score['train'])
    def check(self, seq, scr, limit = 10):
        seqlen = len(seq)
        scrlen = len(scr)
        if (seqlen != scrlen):
            print(seqlen)
            print(scrlen)
            mlen = max(seqlen, scrlen)
            assert(abs(mlen - seqlen) < limit and abs(mlen - scrlen) < limit)
            if mlen > seqlen:
                pad = np.zeros([mlen - seqlen, seq.shape[-1]])
                seq = np.vstack((seq, pad))
            else:
                pad = np.zeros([mlen - scrlen, 1])
                scr = np.vstack((scr, pad))
        assert(len(seq) == len(scr))

    def generate(self, data_txt, mode):
        print('[*] -----------------------')
        print('[*] Generate Original H5 Cache For Dataset {}'.format(self.dataset))
        net = None
        with h5py.File(self.path['cache'], 'a') as h5:
            keys = h5.keys()
            avbases, json = download_list(data_txt, self.path['video'], self.ext)
            if 'country' in json or json['type'] == 'dependent':
                self.info = load_pickle(self.path['info'])
            total = sum([len(v) for k, v in avbases.items()])
            t = 0
            for avbase, ep in avbases.items():
                for i, aid in enumerate(ep):
                    t += 1
                    g = h5.create_group(aid) if aid not in keys else h5[aid]
                    if 'complete' not in list(g.keys()):
                        g['complete'] = np.zeros(1)
                        g['fcomplete'] = np.zeros(1)
                    if int(g['complete'][...]) != 1:
                        fn = os.path.join(os.path.join(self.path['video'], aid + self.ext))
                        if self.fextract or int(g['fcomplete'][...]) != 1:
                            print('[*] {} Extracting {}'.format(self.dataset, aid))
                            net = fe(self.gpu) if net is None else net
                            features = net.extract(fn)
                            assign(g, 'features', features)
                            assign(g, 'fcomplete', np.ones(1))
                        print('[*] Feature Extraction for {} is Completed'.format(aid))
                        print('[*] Request Information for {}'.format(aid))
                        extra = dict()
                        if ('country' in json or json['type'] == 'dependent'):# and not self.info[aid].complete:
                            info = self.info[aid]
                            extra['danmaku'] = info.danmaku
                        else:
                            info = GetVideoInfo(aid.strip('av'), key)
                            extra['danmaku'] = request_danmaku(cid = info.cid)
                        capture = get_capture(fn)
                        extra['duration'] = get_duration(capture = capture)
                        extra['nframes'] = get_nframes(capture = capture)
                        extra['fps'] = get_fps(capture = capture)
                        #print("[*] Compute Boundary")
                        extra['boundary'] = None#get_boundary(fn, capture, extra['nframes'], 'hecate')
                        extra['positions'] = None#get_positions(extra['nframes'])
                        extra['fpsegment'] = None#get_fpsegment(extra['boundary'])
                        extra['score'] = None#get_score(**extra)
                        extra['summary'] = None#get_summary(**extra)
                        extra['complete'] = True
                        
                        for k, v in extra.items():
                            setattr(info, k, v)
                        self.info[aid] = info
                        print('[*] Cache Generate Completed {:.2f}'.format(float(t) / total * 100))
        save_pickle(self.info, self.path['info'])
