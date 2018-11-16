哆啦 A 夢

使用方法

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

============

有些 ep 號是連續的所以只需要知道第一集的 ep 號就能推算後面的號碼

記得要加上

# episodes : 12 

12 代表 12 集

此時我們只需要填上第一集的 ep 號就可以了
