from random import shuffle
from .utils import load_pickle, save_pickle

def split_dataset(f, filename = None, proportion = 0.8):

    infos = load_pickle(f) if isinstance(f, str) else f

    filename = f if isinstance(f, str) else filename

    keys = list(infos.keys())

    shuffle(keys)

    length = len(keys)

    cut = int(length * proportion)

    infos['train'] = keys[:cut]

    infos['test'] = keys[cut:]

    save_pickle(infos, filename)
    
    print('[*] Add Split Keys into {}'.format(filename))

    return infos


def load_keys(filename):

    infos = load_pickle(filename)

    print('[*] Training Samples : {}'.format(len(info['train'])))
    
    print('[*] Testing Samples : {}'.format(len(info['test'])))

