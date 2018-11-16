import os
import argparse
from data.downloader import *
from data.utils import *
from data.danmaku import *
from utils import *

key = '03fc8eb101b091fb'

parser = argparse.ArgumentParser(description = 'Download Video From Bilibili')

parser.add_argument('-d', type = str, help = 'dataset')

parser.add_argument('-o', type = str, default = 'dataset', help = 'output directory')

parser.add_argument('-f', type = str, default = 'mp4', help = 'format')

parser.add_argument('-c', type = str, default = '', help = 'country')

parser.add_argument('-i', action = 'store_true', default = False, help = 'ignore download')

args = parser.parse_args()

cookie = dict()

cookie['DedeUserID'] = '347368229'

cookie['DedeUserID__ckMd5'] = '6e02ca142544e64c'

cookie['sid'] = 'c89920fg'

cookie['SESSDATA'] = '1d13f39c%2C1544798860%2C551f250d'

aids, attr = download_list(os.path.join('list', args.d + '.txt'), os.path.join(args.o, args.d, 'video'), **cookie, ignore = args.i)

infos = dict()

for aid in aids:

    extra = dict()

    if 'ep' in aid:

        epid = aid

        aid = attr['aid']

        fn = os.path.join(args.o, args.d, 'video', '{}.{}'.format(aid, args.f))

        page = int(epid[2:]) - int(attr['base'][2:]) + 1

        info = GetVideoInfo(aid.strip('av'), key, 1)
    
    else:

        fn = os.path.join(args.o, args.d, 'video', '{}.{}'.format(aid, args.f))
        
        info = GetVideoInfo(aid.strip('av'), key)

    extra['danmaku'] = request_danmaku(cid = info.cid)

    if 'country' in attr:
        extra['country'] = attr['country']
        extra['complete'] = False
    else:
        capture = get_capture(fn)
        extra['duration'] = get_duration(capture = capture)
        extra['duration'] = get_duration(capture = capture)
        extra['nframes'] = get_nframes(capture = capture)
        extra['fps'] = get_fps(capture = capture)
        extra['boundary'] = get_boundary(fn, capture, extra['nframes'], 'hecate')
        extra['positions'] = get_positions(extra['nframes'])
        extra['fpsegment'] = get_fpsegment(extra['boundary'])
        extra['score'] = get_score(**extra)
        extra['summary'] = get_summary(**extra)
        extra['complete'] = True

    for k, v in extra.items():
    
        setattr(info, k, v)
    
    infos[aid] = info
    
save_pickle(infos, '{}.info'.format(args.d))

