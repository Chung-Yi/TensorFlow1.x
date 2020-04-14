import tensorflow as tf
import os
import glob
import numpy as np
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
from show import showresult, showimg



path = "mnist_digits_images"
# images = "mnist_digits_images\**\*.bmp"

def main():
    (image, label), labelsnames = load_sample(path)
    print(len(image), image[:2], len(label), label[:2])
    print(labelsnames[label[:2]], labelsnames)


    batch_size = 16

    images_batches, labels_batches = get_batches(image, label, 28, 28, 1, batch_size)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            for step in np.arange(10):
                if coord.should_stop():
                    break
                images, label = sess.run([images_batches, labels_batches])

                showimg(step, label, images, batch_size)
                print(label)
        
        except tf.errors.OutOfRangeError:
            print("Done")
        
        finally:
            coord.request_stop()
        
        coord.join(threads)



def load_sample(sample_dir):
    
    print("loading sample dataset...")
    
    lfilenames = []
    labelsnames = []
    i = 0
    for (dirpath, dirnames, filenames) in os.walk(sample_dir):
        # print(dirpath, dirnames, filenames) 
       
        # print(dirpath, dirnames)
        for filename in filenames:
            filename_path = os.sep.join((dirpath, filename)) #os.sep根據你所處的平台，自動採用相應的分隔符號
            lfilenames.append(filename_path)
            labelsnames.append(dirpath.split('\\')[-1])
    # print(labelsnames)
    lab = list(sorted(set(labelsnames)))
    labdict = dict(zip(lab, list(range(len(lab)))))
    # print(labdict)
    
    labels = [labdict[i] for i in labelsnames]
    return shuffle(np.asarray(lfilenames), np.asarray(labels)), np.asarray(lab)



def get_batches(image, label, resize_w, resize_h, channels, batch_size):
    queue = tf.train.slice_input_producer([image, label])
    label = queue[1]

    image_c = tf.read_file(queue[0])
    
    image = tf.image.decode_bmp(image_c, channels)

    image = tf.image.resize_image_with_crop_or_pad(image, resize_w, resize_h)

    image = tf.image.per_image_standardization(image)
    image_batch, label_batch = tf.train.batch([image, label], batch_size=batch_size, num_threads=64)
    
    print(label_batch.shape)
    images_batch = tf.cast(image_batch, tf.float32)
    labels_batch = tf.reshape(label_batch, [batch_size])
    print(labels_batch.shape)
    
    return images_batch, labels_batch


if __name__ == "__main__":
    main()