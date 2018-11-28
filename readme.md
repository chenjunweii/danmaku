# Donwload List 的使用方法

把需要的 av 號或是 ep 號 寫在 list/dataset.txt

個人上傳的話基本上用 av 號就可以了

如果是有限制地區，記得加上 # country : cn

再來是 bangumi 的部分，有些影片是一連串只有一個 av 號的，不同 ep 號

有 ep 號可以用 cookie 配上 bilibili-get 下載高清影片

但是卻無法下載每一個 episodes 的彈幕，要下載彈幕的話

可以用 api GetVideoInfo(aid, key, page)

aid 就是剛剛說到的，只有單獨一個 av 號

要怎麽指定哪一集呢，這個 api 有一個參數 page，就是用來指定這個參數的

所以透過 GetVideoInfo(aid, key, page = 2)

我們就能獲得第二集的 info，藉此獲得第二集的 cid

---

## 專輯

有些 ep 號是連續的所以只需要知道第一集的 ep 號就能推算後面的號碼

記得要加上 

```shell
# episodes : 12
ep123456
```

12 代表 12 集

此時我們只需要填上第一集的 ep 號就可以了

---

##斷點

如果有跳過的地方可以加入寫成以下形式

```shell
# exclude

ep123456

ep234567

```

###但是必須加在 txt 的尾端

---

##分段的專輯


如果一個影集分成兩段可以寫成 (要空格)

```shell
ep123456 - ep234567

ep567896 - ep567901
```



但是總共的集數加起來必須和 episodes 一樣

---

如果要下載 影片和彈幕可以用 download.py 裡面可以指定 cookie

彈幕和影片資訊會存成 pickle 檔

---

train key, test key

詳情可以看 data/key.py

透過 split_key.py 來切，切完會存在 data.info 裡

---

#Think

- 單純使用彈幕的問題
- 用 hecate 的問題
- KTS 層層式切割
- Generate Sub sampling feature
- 保留原始版本
- 加入聲音，用聲音切 kts 可能會更準
- 音頻分離，兩個人的聲音分別取出，讓他知道有兩個人
- Object Detection 把物體切割下來，編碼成特徵塞到 Feature 裡
  - 如果兩個畫面有類似同樣的人事物，應該是同一個 shot
  - 可以用兩層的方式，第一層 object detection -> crop -> resize -> Feature Extraction

---

順序是 先用 download.py 下載

因爲原始長度太長，有些部分就不做了，預設爲 None

之後用 train.py 或說 data.py 裡的 generate 來做 feature extraction, boundary 的地方也可以跳過

最後再用 test_subsample 來生成 1fps 的 training data

****

# Generate

只用來做 Feature Extraction

之後還得用 Sub-sampling

要進行第一次的 Generate : 要把 tfps 設成 -1

並且 force check 要打開。

做完 subsampling 後要開始訓練的時候再把 tfps 設成 1



如果使用 train.py 出現如下錯誤

兩種 sequence 長度不同

[123124]

[1]

非常有可能是因爲 tfps 沒設定成 1，讀到原始長度版本的 info 檔

# Data List

Data List 的部分也更新成可以讀取 Json 檔了

之後全部都會轉換到 Json 版本

```json
"type" : "dependent"
```

當是專輯的時候，就設定爲 dependent

可以分段

```json
{
    "dataset" : "oolong",
    "type" : "dependent",
    "av-base" : [ 

        {
            "absolute" : "0 - 50",
            "av-id" : "av846108",
            "ep-id" : "ep51558",
            "episodes" : 50, 
            "exclude" : []
        },  

        {   
            "absolute" : "51 - 100",
            "av-id" : "av849382",
            "ep-id" : "ep51608",
            "episodes" : 50
        },  

        {   
            "absolute" : "101 - 200",
            "av-id" : "av849844",
            "ep-id" : "ep51659",                                                                                                                                                                                    
            "episodes" : 100 
        },  
    
        {   
            "absolute" : "201 - 300",
            "av-id" : "av857187",
            "ep-id" : "ep51759",
            "episodes" : 100 
        }            
    ]  
}
```

