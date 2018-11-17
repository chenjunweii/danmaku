
config = dict()

config['dataset'] = None
config['train'] = True
config['test'] = False
config['train_list'] = None
config['test_list'] = None
config['device'] = 0
config['batch'] = 7
config['directory'] = 'dataset'
config['ext'] = '.mp4'
config['sr'] = 15 # sub-sampling rate
config['force'] = True # Force Re-Generate Cache
config['boundary'] = 'hecate'
config['verbose'] = True
config['fcheck'] = False # true force re-check cache, info
config['fextract'] = False # true => force re-extract feature
config['nhidden'] = 256
config['feature'] = 1024
config['lr_decay_step'] = 250
config['lr_decay_rate'] = 0.9
# initialize

config['epoch'] = 0
config['counter'] = - config['batch'] # epoch counter
config['sequence'] = dict()
config['score'] = dict()
config['nframes'] = dict()
config['info'] = dict()
config['path'] = dict()

