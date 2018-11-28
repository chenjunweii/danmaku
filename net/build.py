from .network import lstm, birnn
from .block3d import D3
from .Gan import Gan
from .wave2d import Wave2DED
from .wavenet import WaveNet
from .srnnseq import endsrnnseq
from .densewave import DenseWaveED

def build(arch, nbatch, feature, device, mode, nhidden = None, nds = None):

    print('[*] Build {}'.format(arch))

    if arch == 'lstm':

        net = lstm(nhidden)

    elif arch == 'bilstm':

        net = birnn(nhidden, nds)

    elif arch == 'bilstm2':

        net = BILSTM(feature, nhidden[0])

    elif arch == 'd2':

        layers = 4

        G_Encoder = {
                'kernel' : [[3, 3]] * layers,
                'stride' : [[4, 2]] * layers,
                'channel' : [32, 64, 256, 512],# * layers,
                'layers' : layers,
                'dilation' : [[1, 1]] * layers,
                'padding' : [[1, 1]] * layers,
                'device' : device,
                'feature' : feature,
                'auto' : False,
                'flatten' : False,
                'norm' : False,
                'swap_in' : True,
                'swap_out' : False,
                'reconstruct' : False,
                'arch' : 'd2',
                'encoder' : True
            }
        
        G_Decoder = {
                'kernel' : [[3, 3]] * layers,
                'stride' : [[4, 1]] * layers,
                'channel' : [256, 64, 32, 1],# * layers,
                'layers' : layers,
                'dilation' : [[1, 1]] * layers,
                'padding' : [[1, 1]] * layers,
                'device' : device,
                'feature' : feature,
                'auto' : False,
                'flatten' : False,
                'norm' : False,
                'swap_in' : False,
                'swap_out' : True,
                'reconstruct' : False,
                'arch' : 'd2'
            }
        
        net = Wave2DED(G_Encoder, G_Decoder)

    elif arch == 'wavenet':

        layers = 10

        wave = {
                'kernel' : [[3, 3]] * layers,
                'stride' : [[1, 1]] * layers,
                'channel' : [1] * layers,
                'layers' : layers,
                'dilation' : [[2, 1]] * layers,
                'padding' : [[1, 0]] * layers,
                'device' : device,
                'feature' : feature,
                'nhidden' : nhidden,
                'arch' : 'd2',
            }
        
        net = WaveNet(**wave)
    
    elif arch == 'd3':

        net = D3('x', 2, feature, connection = 'dense')

    elif arch == 'srnnseq':

        net = endsrnnseq()

    elif arch == 'densewave':

        from .config.densewave import config

        net = DenseWaveED(*config)

    elif arch == 'gan-lstm':

        net = Gan(None, None, device, arch)

    elif arch == 'gan-srnn':

        net = Gan(None, None, device, arch)
    elif arch == 'gan-srnnseq':

        net = Gan(None, None, device, arch)

    elif arch == 'gan-d3':

        stride = 2; kernel = 3; layers = 7; dilation = 1; norm = False

        # [time, feature]

        G = {
              'kernel' : kernel,
              'stride' : stride,
              'layers' : layers,
              'dilation' : dilation,
              'device' : device,
              'feature' : feature,
              'auto' : False,
              'flatten' : False,
              'norm' : False,
              'reconstruct' : True,
              'arch' : 'encoder-decoder-d3'
            }
        
        D = {
              'kernel' : kernel,
              'stride' : 2,
              'layers' : 3,
              'dilation' : dilation,
              'device' : device,
              'feature' : feature,
              'auto' : False,
              'flatten' : False,
              'norm' : False,
              'reconstruct' : False,
              'arch' : 'bottleneck-d3-lstm'
            }
        net = Gan(G, D, device, arch)




    elif arch == 'gan-d2':

        layers = 2

        G_Encoder = {
                'kernel' : [[3, 3]] * layers,
                'stride' : [[1, 2]] * layers,
                'channel' : [1] * layers,
                'layers' : layers,
                'dilation' : [[2, 1]] * layers,
                'padding' : [[2, 1]] * layers,
                'device' : device,
                'feature' : feature,
                'auto' : False,
                'flatten' : False,
                'norm' : False,
                'swap_in' : True,
                'swap_out' : False,
                'reconstruct' : False,
                'arch' : 'd2',
                'encoder' : True
            }
        
        G_Decoder = {
                'kernel' : [[3, 3]] * layers,
                'stride' : [[1, 1]] * layers,
                'channel' : [1] * layers,
                'layers' : layers,
                'dilation' : [[2, 1]] * layers,
                'padding' : [[2, 1]] * layers,
                'device' : device,
                'feature' : feature,
                'auto' : False,
                'flatten' : False,
                'norm' : False,
                'swap_in' : False,
                'swap_out' : True,
                'reconstruct' : False,
                'arch' : 'd2'
            }

        layers = 3

        D = {
                'kernel' : [[3, 3]] * layers,
                'stride' : [[1, 2]] * layers,
                'channel' : [1] * layers,
                'layers' : layers,
                'dilation' : [[2, 2]] * layers,
                'padding' : [[1, 1]] * layers,
                'device' : device,
                'feature' : feature,
                'auto' : False,
                'flatten' : False,
                'norm' : True,
                'swap_in' : True,
                'swap_out' : True,
                'block' : 'Discriminator',
                'encoder' : False
            }
        
        net = Gan([G_Encoder, G_Decoder], D, device, arch)

    else:

        raise ValueError('[!] Architecture is not supported')

    return net
