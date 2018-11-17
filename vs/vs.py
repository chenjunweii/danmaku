import mxnet as mx

# Local

from net.build import build
from data.data import data
from data.preprocess import preprocess_list

from .init import init

class vs(init):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.data = data(**kwargs)
        self.device = mx.gpu(self.gpu)
    def Train(self):

        self.Init()

        seq, scr = self.data.next()
        seq, scr = preprocess_list(seq, scr)
        nds = dict()
        net = build(self.arch, self.batch, self.feature, self.device, 'train', self.nhidden, nds)
        net.initialize(mx.init.MSRAPrelu(), ctx = self.device)
    def test(self):

        pass


