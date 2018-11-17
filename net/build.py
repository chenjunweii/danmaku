


def build(arch, nhidden, nds, nbatch, device, mode, datatype, wave = None, lr = None, lr_scheduler = None):

    print('[*] Build {}'.format(arch))

    if arch == 'lstm':

        net = lstm(nhidden, nds, device, mode)

    elif arch == 'bilstm':

        net = birnn(nhidden, nds, device, mode)

    elif arch == 'bilstm2':

        net = BILSTM(feature_size, nhidden[0])

    elif arch == 'd2':

        net = D2('x', 2, feature_size, connection = 'dense')
    
    elif arch == 'd3':

        net = D3('x', 2, feature_size, connection = 'dense')

    elif arch == 'gan':

        stride = 2; kernel = 3; layers = 3; dilation = 1; norm = False

        # [time, feature]

        EN = {
                'kernel' : [[3, 3], [3, 3], [3, 3], [3, 3], [3,3]],
                'stride' : [[2, 2], [2, 2], [2, 2], [2, 2], [2,2]],
                'channel' : [8, 8, 16, 16, 32],
                'layers' : 5,
                'dilation' : [[1, 1], [1, 1], [1, 1], [1,1], [1,1]],
                'padding' : [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1]],
                'device' : device,
                'feature' : feature_size,
                'auto' : False,
                'flatten' : False,
                'norm' : True,
                'swap_in' : True,
                'swap_out' : False
            }
        
        DE = {
                'kernel' : [[3, 3], [3, 3], [3, 3], [3,3], [3,3]],
                'stride' : [[2, 1], [2, 1], [2, 1], [2, 1], [2, 1]],
                'channel' : [16, 16, 8, 8, 1],
                'layers' : 5,
                'dilation' : [[1, 1], [1, 1], [1, 1], [1,1], [1,1]],
                'padding' : [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1]],
                'device' : device,
                'feature' : feature_size,
                'auto' : False,
                'flatten' : False,
                'norm' : True,
                'swap_in' : False,
                'swap_out' : True
            }

        G = {
              'kernel' : kernel,
              'stride' : stride,
              'layers' : layers,
              'dilation' : dilation,
              'device' : device,
              'feature' : feature_size,
              'auto' : False,
              'flatten' : False,
              'norm' : True,
              'reconstruct' : True,
              'arch' : 'encoder-decoder-unet-bn-d3'
            }
        
        D = {
              'kernel' : kernel,
              'stride' : 2,
              'layers' : 3,
              'dilation' : dilation,
              'device' : device,
              'feature' : feature_size,
              'auto' : False,
              'flatten' : False,
              'norm' : True,
              'reconstruct' : False,
              'arch' : 'bottleneck-bn-d3-lstm'
            }
        D2 = {
                'kernel' : [[3, 3], [3, 3], [3, 3], [3, 3], [3, 3]],
                'stride' : [[2, 2], [2, 2], [2, 2], [2, 2], [2, 2]],
                'channel' : [1, 1, 1, 1, 1],
                'layers' : 5,
                'dilation' : [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1]],
                'padding' : [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1]],
                'device' : device,
                'feature' : feature_size,
                'auto' : False,
                'flatten' : False,
                'norm' : True,
                'swap_in' : True,
                'swap_out' : True,
                'block' : 'Discriminator',
            }
        
        #net = Gan([EN, DE], D2, device)
        net = Gan(G, D, device)

    else:

        raise ValueError('[!] Architecture is not supported')

    return net, arch
