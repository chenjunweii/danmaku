
config = dict()

config['dataset'] = None
config['train'] = True # using training set
config['test'] = True # using testing set
config['train_list'] = None
config['test_list'] = None
config['device'] = 0
config['batch'] = 2
config['directory'] = 'dataset'
config['ext'] = '.mp4'
config['tfps'] = 1 # sub-sampling rate
config['force'] = True # Force Re-Generate Cache
config['boundary'] = 'kts'
config['verbose'] = True
config['fcheck'] = False # true force re-check cache, info
config['fextract'] = False # true => force re-extract feature
config['fsplit'] = False # Force Re-Split Dataset
config['nhidden'] = [256]
config['feature'] = 1024
config['lr_decay_step'] = 250
config['lr_decay_rate'] = 0.9
config['checkpoint'] = None
config['lr'] = 0.00005
config['metric'] = 'avg'
config['prefix'] = 'default'
config['se'] = 1 # save epoch
config['epoch'] = 60
config['eepoch'] = 1 # evaluate epoch
config['sepoch'] = 10 # evaluate epoch
# initialize

config['counter'] = - config['batch'] # epoch counter
config['summary'] = dict()
config['sequence'] = dict()
config['score'] = dict()
config['nframes'] = dict()
config['info'] = dict()
config['path'] = dict()
config['ids'] = dict()

