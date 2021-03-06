## 將圖片製作成記憶體物件資料集
> 將數字1~9的圖片做成資料集，並且輸入到tensorflow的靜態圖中，然後在將資料從靜態圖中輸出，並顯示。

在讀取資料時，建議不要一次讀取所有資料，建議使用queue方式讀取，可以使用兩個執行緒去處理，一個從queue取出資料，另一個用於讀取檔案並放入快取，如下圖所示:

![image](https://wx1.sinaimg.cn/large/b62e365dly1fh6gm6z2d0g21jb0drws7.gif)


### 產生queue中的批次資料
1. tf.train.slice_input_producer產生queue，如果有帶參數num_epochs，表示將數據的圖片全部讀取過num_epochs次，例如全部圖片有3張，  
   num_epochs=2，就會產生2x3張圖片，如果num_epochs=N，表示不限循環次數，直到其他終止條件達成才會停止。除此之外，如果有帶num_epochs，必須要同時tf.global_variables_initializer()，tf.local_variables_initializer()
2. 將queue裡的圖片進行處理
3. tf.train.batch將處理後的圖片做成批次數據

### 輸出結果
![](output/output.PNG)