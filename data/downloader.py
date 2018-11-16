import os
import shlex, subprocess

# https://github.com/kamikat/bilibili-get

def download(aid, directory = '.', DedeUserID = None, DedeUserID__ckMd5 = None, sid = None, SESSDATA = None, ignore = False):

    if 'av' in aid:

        command = 'bilibili-get -f mp4 -o {} https://www.bilibili.com/video/{}'.format(os.path.join(directory, 'av%(aid)s.%(ext)s'), aid)

    elif 'ep' in aid:

        command = 'bilibili-get -f mp4 -o {} https://www.bilibili.com/bangumi/play/{}'.format(os.path.join(directory, 'ep%(episode_id)s.%(ext)s'), aid)


    location = command.find('http')

    if DedeUserID is not None:

        print('[*] Download With Cookie') 
        
        cookie = " -C 'DedeUserID={}; DedeUserID__ckMd5={}; sid={}; SESSDATA={};' ".format(DedeUserID, DedeUserID__ckMd5, sid, SESSDATA)
        
        command = command[:location] + cookie + command[location:]
        
    command = shlex.split(command)

    if not ignore:

        subprocess.call(command)

def read_list(fn):

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

                else:

                    klass[c] = []

                    klass[c].append(line[:-1])
            
            else:

                if 'episodes' in attribute:

                    if 'ep' in line:

                        c = attribute['episodes'] if attribute['episodes'] > 1 else 1

                        attribute['base'] = line[:-1]

                        for _c in range(c):

                            lines.append('ep' + str(int(line[:-1][2:]) + _c))


                    elif 'av' in line:

                        attribute['aid'] = line[:-1]
                else:
                    
                    lines.append(line[:-1])

    return lines, klass, attribute

def download_list(txt, directory, ext = '.mp4', **cookie):

    ids, kids, attr = read_list(txt)

    if not os.path.isdir(directory):

        os.makedirs(directory)

    for v in ids:

        if not os.path.isfile(os.path.join(directory, v + ext)):   

            download(v, directory, **cookie)

    return ids, attr
    
if __name__ == '__main__':

    ids = download_list('list.txt', 'Danmaku/video')

    print(len(ids))
    
    print(ids)

