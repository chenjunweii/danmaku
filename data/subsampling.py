import os
import copy
import h5py
import numpy as np
from data.utils import load_pickle, save_pickle, assign
from data.label import get_boundary, get_fpsegment, get_score
from data.preprocess import get_positions
from data.summary import get_summary


def fpsegment_mapping(fpsegment, nframes, rate):

    pass


def boundary_mapping(boundary, nframes, rate):
    oboundary = []
    for b in boundary:
        start, end = b
        ostart = (start * rate)
        oend = end * rate
        if oend >= nframes - 1:
            oend = nframes - 1
        oboundary.append([ostart, oend])
    return np.asarray(oboundary)

def subsample(dataset, tfps, directory):

    pinfo = os.path.join(directory, dataset, '{}.info'.format(dataset))

    pcache = os.path.join(directory, dataset, '{}.h5'.format(dataset))

    infos = load_pickle(pinfo)

    rate = {k : int(float(v.fps) / tfps) for k, v in infos.items() if 'av' in k or 'ep' in k}
    
    psh5 = subsample_feature(pcache, tfps, rate)

    subsample_info(pinfo, psh5, tfps, rate)
    
    print('[*] SubSample Finished')

def subsample_feature(original, tfps, rate):

    subsampled = '{}_fps_{}.h5'.format(original.split('.')[0], tfps)

    with h5py.File(original, 'r') as oh5, h5py.File(subsampled) as sh5:
        
        for aid in oh5.keys():

            print('[*] Sub-Sampling {} '.format(aid))

            og = oh5[aid] # original group

            assert(int(og['fcomplete'][...]) == 1)

            sg = sh5.create_group(aid) if aid not in sh5.keys() else sh5[aid]

            ofeatures = og['features'][...]

            sfeatures = [] # sub-sampled

            tfeatures = [] # temp


            """

            for i, f in enumerate(ofeatures):

                tfeatures.append(f)

                if len(tfeatures) == rate[aid] or (i + 1) == len(ofeatures):

                    sfeatures.append(np.array(tfeatures).mean(0))

                    tfeatures = []

            """

            slength = int(len(ofeatures) / rate[aid])
            
            for s in range(slength):

                start = int(max(s - 0.5, 0) * rate[aid]); end = int(min(s + 1.5, slength) * rate[aid])

                sfeatures.append(ofeatures[start : end].mean(0))

            assign(sg, 'features', np.array(sfeatures))
            assign(sg, 'fcomplete', np.ones(1)) # feature complete
            assign(sg, 'complete', np.ones(1)) # feature complete

    return subsampled
                    

def subsample_info(oinfo, psh5, tfps, rate):

    oinfos = load_pickle(oinfo)

    sinfos = dict()

    with h5py.File(psh5, 'r') as sh5:

        ndata = len(list(sh5.keys()))
        
        for i, (aid, v) in enumerate(oinfos.items()):

            if 'av' in aid or 'ep' in aid:
                assert(oinfos[aid].complete)
                assert(int(sh5[aid]['fcomplete'][...]) == 1)
                sfeatures = sh5[aid]['features'][...] # subfeature
                sinfo = copy.deepcopy(oinfos[aid])
                extra = vars(sinfo)
                extra['boundary'] = boundary_mapping(get_boundary(sfeatures, fps = tfps, method = 'kts'), sinfo.nframes, rate[aid])
                print(extra['boundary'])
                extra['positions'] = get_positions(sinfo.nframes, sr = rate[aid]) # v
                extra['fpsegment'] = get_fpsegment(extra['boundary']) # v
                extra['nsamples'] = len(sfeatures)
                extra['subsampled'] = True
                extra['score'] = get_score(**extra)
                extra['summary'] = get_summary(**extra)
                extra['rate'] = rate[aid]
                if 'history' not in extra:
                    extra['history'] = [rate[aid]]
                else:
                    extra['history'].append(rate[aid])
                extra['tfps'] = tfps
                extra['complete'] = True
                
                for k, v in extra.items():
                    setattr(sinfo, k, v)
                sinfos[aid] = sinfo
                
                print('[*] Info Sub-Sampling Completed {:d} %'.format(int(float(i+1) / ndata * 100)))
        
        sinfosfn = '{}_fps_{}.info'.format(oinfo.split('.')[0], tfps) # sub-sampled infos filename

        save_pickle(sinfos, sinfosfn)

        print('[*] Sub-Sampled Info is save to : {}'.format(sinfosfn))
    
