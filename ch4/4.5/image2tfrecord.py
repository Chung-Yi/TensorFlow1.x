import os
import tensorflow as tf
from PIL import Image
from sklearn.utils import shuffle
import numpy as np
from tqdm import tqdm

DataPath = "man_woman"


def main():
    
    if not os.path.isfile("mydata.tfrecords"):
        (filenames, labels), _ = load_sample(DataPath)
        makeTFRec(filenames, labels)

    
    TFRecordfilenames = ["mydata.tfrecords"]
    image, label = read_and_decode(TFRecordfilenames, flag='test')
    RunTFSession(image, label)

def load_sample(sample_dir, shuffleflag=True):
    print("loading sample dataset...")
    lfilenames = []
    labelsnames = []
    for (dirpath, dirnames, filenames) in os.walk(sample_dir):
        # print(dirpath)

        for  filename in filenames:
            filename_path = os.sep.join([dirpath, filename])
            lfilenames.append(filename_path)
            labelsnames.append(dirpath.split('\\')[-1])
    
    lab = list(sorted(set(labelsnames)))
    labdict = dict(zip(lab, list(range(len(lab)))))
    

    labels = [labdict[i] for i in labelsnames]
    
    if shuffleflag:
        return shuffle(np.asarray(lfilenames), np.asarray(labels)), np.asarray(lab)
    else:
        return (np.asarray(lfilenames), np.asarray(labels)), np.asarray(lab)


def makeTFRec(filenames, labels):
    writer = tf.python_io.TFRecordWriter("mydata.tfrecords")
    for i in tqdm(range(0, len(labels))):
        img = Image.open(filenames[i])
        img = img.resize((256, 256))
        img_raw = img.tobytes()
        
        example = tf.train.Example(features=tf.train.Features(feature={
            "label":tf.train.Feature(int64_list=tf.train.Int64List(value=[labels[i]])),
            "img_raw":tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        }))
        writer.write(example.SerializeToString())
    
    writer.close()

def read_and_decode(filenames, flag="train", batch_size=3):
    #根據檔名產生一個佇列

    if flag == "train":
        #亂數操作，並循環讀取
        filename_queue = tf.train.string_input_producer(filenames)
    else:
        filename_queue = tf.train.string_input_producer(filenames, num_epochs=1, shuffle=False)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue) #傳回檔案名稱跟檔案
    features = tf.parse_single_example(serialized_example, features={
        "label":tf.FixedLenFeature([], tf.int64),
        "img_raw": tf.FixedLenFeature([], tf.string),
    })

    #tf.decode_raw可以將字串解析成影像對應的像素陣列
    image = tf.decode_raw(features["img_raw"], tf.uint8)
    image = tf.reshape(image, [256, 256, 3])
  

    label = tf.cast(features["label"], tf.int32)
    
    if flag == "train":
        image = tf.cast(image, tf.float32) * (1./255) - 0.5
        img_batch, label_batch = tf.train.batch([image, label], batch_size=batch_size, capacity=20)
    
    return image, label

def RunTFSession(image, label):
    saveimgpath = "show\\"
    if tf.gfile.Exists(saveimgpath):
        tf.gfile.DeleteRecursively(saveimgpath) #如果存在saveimgpath則將其刪除
    tf.gfile.MakeDirs(saveimgpath) #建立saveimgpath

    with tf.Session() as sess:
        #初始化變數(沒有這行會出錯)
        sess.run(tf.local_variables_initializer())

        #建立協調器及啟動多執行緒
        coord = tf.train.Coordinator() 
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        myset = set([])
        try:
            i = 0
            while True:
                if coord.should_stop():
                    break
                example, examplelab = sess.run([image, label])
            
                #取出image, label
                examplelab = str(examplelab)
                print(examplelab)
                if examplelab not in myset:
                    myset.add(examplelab)
                    tf.gfile.MakeDirs(saveimgpath + examplelab)
                
                img = Image.fromarray(example, "RGB") #轉換Image格式
                img.save(saveimgpath+examplelab+"/"+str(i)+"_Label_" + ".jpg")
                print(i)
                i += 1

        except tf.errors.OutOfRangeError:
            print("Done Test -- epoch limit reached")

        finally:
            coord.request_stop()

        coord.join(threads)
        print("stop()")


if __name__ == "__main__":
    main()