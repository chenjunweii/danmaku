import shuffle
from .utils import load_pickle, save_pickle

def split_dataset(filename, proportion):

    infos = load_pickle(filename) if isinstance(filename, str) else filename

    keys = list(infos.keys)

    shuffle(keys)

    length = len(keys)

    cut = int(length * proportion)

    infos['train'] = key[:cut]

    infos['test'] = key[cut:]

    save_pickle(infos, filename)
    
    print('[*] Add Split Keys into {}'.format(filename))

    return infos


def load_keys(filename):

    infos = load_pickle(filename)

    print('[*] Training Samples : {}'.format(len(info['train'])))
    
    print('[*] Testing Samples : {}'.format(len(info['test'])))

