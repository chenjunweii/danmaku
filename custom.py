from bilibili.bilibili import *
from data.downloader import *

key = '03fc8eb101b091fb'

#o = GetVideoOfUploader(4856007, pagesize = 20, page = 2)

#print('length video list : ', len(o))
#番剧 日本已完结, 全12话AV28423563
#v2 = GetVideoInfo('26753891', key)
#v2 = GetVideoInfo('28423563', key)
v2 = GetVideoInfo('8075425', key)

print(v2.episode)
print(v2.page)
print(v2.spid)

raise

download('')

print(v2)

d = ParseDanmuku(v2.cid)

for _d in d:

    print (_d.content)

    print ('time video : ', _d.t_video)


