import os
import shlex, subprocess
from .utils import load_json

# https://github.com/kamikat/bilibili-get

def download(aid, directory = '.', DedeUserID = None, DedeUserID__ckMd5 = None, sid = None, SESSDATA = None, ignore = False, quality = 0, debug = False):

    if 'av' in aid:

        command = 'bilibili-get -q {} -f mp4 -o {} https://www.bilibili.com/video/{}'.format(quality, os.path.join(directory, 'av%(aid)s.%(ext)s'), aid)

    elif 'ep' in aid:

        command = 'bilibili-get -q {} -f mp4 -o {} https://www.bilibili.com/bangumi/play/{}'.format(quality, os.path.join(directory, 'ep%(episode_id)s.%(ext)s'), aid)

    location = command.find('http')

    print('[*] Download ID : {}'.format(aid))
    
    if DedeUserID is not None:

        print('[*] Download With Cookie') 
        
        cookie = " -C 'DedeUserID={}; DedeUserID__ckMd5={}; sid={}; SESSDATA={};' ".format(DedeUserID, DedeUserID__ckMd5, sid, SESSDATA)
        
        command = command[:location] + cookie + command[location:]

    if debug:

        print('[*] Command : ', command)
        
    command = shlex.split(command)

    if not ignore:

        subprocess.call(command)


def next_ep(av, offset):

    t = av[:2]

    ids = av[2:]

    return t + str(int(ids) + offset)

def read_list(fn):

    json = load_json(fn)

    avbase = dict()

    epbase = []

    for av in json['av-base']:

        epids = []

        for e in range(av['episodes']):

            epids.append(next_ep(av['ep-id'], e))
        
        if 'exclude' in av:

            for ex in av['exclude']:

                epids.remove(ex)

                epids.append(next_ep(epids[-1], 1))

        avbase[av['av-id']] = epids

    return avbase, json

def read_list_v(fn):

    attribute = dict()

    with open(fn) as f:

        raw_lines = f.readlines()

        lines = []

        klass = dict()

        c = None

        for line in raw_lines:
                
            if '\n' == line.strip(' '):

                continue
            
            elif '#' in line:

                c = line.strip('#')

                if 'country' in c:
                    
                    attribute['country'] = c.split(':')[-1].strip(' ').strip('\n')

                elif 'episodes' in c:
                    
                    attribute['episodes'] = int(c.split(':')[-1].strip(' ').strip('\n'))

                elif 'exclude' in c:

                    attribute['exclude'] = []

                elif 'av-base' in c:

                    if 'av-base' not in attribute:

                        attribute['av-base'] = []

                    #base = c.split(':')[-1

                    #attribute['av-base'].append(base)

                else:

                    klass[c] = []

                    klass[c].append(line[:-1])
            
            else:

                if 'episodes' in attribute:

                    if 'ep' in line:

                        c = attribute['episodes'] if attribute['episodes'] > 1 else 1

                        if 'exclude' not in attribute:

                            if ' - ' in line:

                                start, end = line.split('-')

                                start = start.strip(' \n')

                                end = end.strip(' \n')

                                c = int(end[2:]) - int(start[2:]) + 1

                                for _c in range(c):

                                    lines.append('ep' + str(int(start[2:]) + _c))
                            else:

                                attribute['base'] = line[:-1]

                                for _c in range(c):

                                    lines.append('ep' + str(int(line[:-1][2:]) + _c))

                        else:

                            epid = line[:-1]

                            last_epid = 'ep' + str(int(lines[-1][2:]) + 1)

                            lines.remove(epid)

                            lines.append(last_epid)

                    elif 'av' in line:

                        attribute['aid'] = line[:-1]
                else:
                    
                    lines.append(line[:-1])

    return lines, klass, attribute

def download_list(txt, directory, ext = '.mp4', **cookie):

    avbases, json = read_list(txt)

    if not os.path.isdir(directory):

        os.makedirs(directory)

    if json['type'] == 'dependent':

        total = sum([len(v) for k, v in avbases.items()])
        
        print('[*] Download From Dataset Json, Total Videos : {}'.format(total))

        for k, v in avbases.items():

            for ep in v:

                if not os.path.isfile(os.path.join(directory, ep + ext)):

                    download(ep, directory, **cookie)


    # check if directory contain all videos if using episodesa

    if json['type'] == 'dependent':

        pass

        #count = len(os.listdir(directory))

        #assert(count >= attr['episodes'])

    return avbases, json
    
if __name__ == '__main__':

    ids = download_list('list.txt', 'Danmaku/video')

    print(len(ids))
    
    print(ids)

