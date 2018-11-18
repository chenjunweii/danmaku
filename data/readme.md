# subsampling.py

這裡面的 KTS 是用 sub-sampling 後的 feature 下去切的，所以還原的時候還要以原來的 rate 還原



透過不同程度的 sub-sampling 來做 KTS ，我們可以切出不同程度的 shot boundary

- Local Shots
- Global Shots

最後我可以爲了增加總結的分散程度，可以從每個廣域的 Shots 中的抽取一個 Local Shot 來作爲 Global Shot 才不會，最後變成 Global Shots 全部集中在某一區



#subsample_info

```python
extra['positions'] = get_positions(sinfo.nframes, sr = rate[aid])                 
extra['fpsegment'] = get_fpsegment(extra['boundary'])                             
extra['nsample'] = len(sfeatures)                                                 
extra['score'] = get_score(danmaku = sinfo.danmaku, 
                           fps = sinfo.fps,
                           nframes = extra['nsample'])                                                                                    
extra['summary'] = get_summary(**extra)    
```

danmku : 原始長度

boundary 也是原始的顆度，因爲用 bounary mapping

get_score 的參數　nframes ＝ nsample 是因爲被 sub-sample 了

extra['score'] : 使用 subsampled 後的長度，訓練用所以長度是 subsampled 的

summary ：還是原始長度，必須使用參數 nframes，其實是驗證用的，所以必須是原始長度，如果有層層式結構可能就不會拿來用了

sfpsegment 也是原始顆度，因爲 sboundary 也是原始顆度

把上面的名字改成



sub



sub - boundary



sub 指得是經過  subsampling，所以 sub-boundary 是經過　 subsampling 再取 kts 的 boundary



ssummary 是原始長度，但是用 sub-boundary 來生成的 summary





# sub sampling 取法

雖然 rate 是　30 ，但我們可能會用更大的 kernel 區做才會有重疊的部分，不然會切的不好



rate = 30，但我們取 60

15  + 30 + 15

第一個 15 是前面的 frame

30 是本來的

最後一個 15 是後面的 frame