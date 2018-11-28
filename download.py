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

parser.add_argument('-q', type = int, default = 0, help = 'quality')

parser.add_argument('-i', action = 'store_true', default = False, help = 'ignore download')

args = parser.parse_args()

cookie = dict()

cookie['DedeUserID'] = '347368229'

cookie['DedeUserID__ckMd5'] = '6e02ca142544e64c'

cookie['sid'] = 'ii8ca1k2'

cookie['SESSDATA'] = '1d13f39c%2C1544246349%2Cc62b611b'

avbases, json = download_list(os.path.join('list', args.d + '.json'), os.path.join(args.o, args.d, 'video'), **cookie, ignore = args.i, quality = args.q, debug = True)

print('[*] Video Download Finished')

filename = os.path.join(args.o, args.d, '{}.info'.format(args.d))

infos = load_pickle(filename) if os.path.isfile(filename) else dict()

item = 0

for avid, eps in avbases.items():
    for ep in eps:
        extra = dict()
        if ep in infos:
            if infos[ep].complete:
                continue
        else:
            print('-' * 30)
            if 'ep' in ep:
                fn = os.path.join(args.o, args.d, 'video', '{}.{}'.format(ep, args.f))
                page = int(ep[2:]) - int(eps[0][2:]) + 1
                print('[*] Requesting Danmaku For Episodes {} of {}'.format(page, avid))
                info = GetVideoInfo(avid.strip('av'), key, page)
                #page += 1
            else:
                fn = os.path.join(args.o, args.d, 'video', '{}.{}'.format(aid, args.f))
                info = GetVideoInfo(aid.strip('av'), key)
            extra['danmaku'], extra['content'] = request_danmaku(cid = info.cid)
            if 'country' in json:
                extra['country'] = json['country']
                extra['complete'] = False
            else:
                absolute = int(json[avid]['absolute'].split('-')[0].strip(' ')) + page - 1
                print('[*] Absolute Episodes : ', absolute)
                print('[*] Number of Danmaku : ', len(extra['danmaku']))
                print('[*] Filename : ', fn)
                print('[*] Danmaku Content : ', extra['content'][0 : 20])
                capture = get_capture(fn)
                extra['duration'] = get_duration(capture = capture)
                extra['nframes'] = get_nframes(capture = capture)
                extra['fps'] = get_fps(capture = capture)
                extra['boundary'] = None #get_boundary(fn, capture, extra['nframes'], 'hecate')
                extra['positions'] = None #get_positions(extra['nframes'])
                extra['fpsegment'] = None #get_fpsegment(extra['boundary'])
                extra['score'] = None #get_score(**extra)
                extra['summary'] = None #get_summary(**extra)
                extra['complete'] = True
            for k, v in extra.items():
                setattr(info, k, v)

            if json['type'] == 'dependent' :
                
                infos[ep] = info
            
            else:
           
                infos[aid] = info
            
            item += 1

            if item % 10 == 0:

                save_pickle(infos, filename)

                print('[*] Info CheckPoint is save to ', filename)

save_pickle(infos, filename)

print('[*] Info is save to ', filename)

