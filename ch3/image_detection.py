import sys
import tensorflow as tf
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


net_path = r'slim'
checkepoint_file = 'pnasnet-5_large_2017_12_13\model.ckpt'

if net_path not in sys.path:
    sys.path.insert(0, net_path)
else:
    print('already add slim')

from datasets import imagenet
from nets.nasnet import pnasnet



def main():
    slim = tf.contrib.slim

    tf.reset_default_graph()
    image_size = pnasnet.build_pnasnet_large.default_image_size  # 331

    labels = imagenet.create_readable_names_for_imagenet_labels()
    
    

    sample_images = ['hy.jpg', 'mac.jpg', 'filename3.jpg', '72.jpg', 'ps.jpg']

    input_imgs = tf.placeholder(tf.float32, [None, image_size, image_size, 3])
    x1 = 2 * (input_imgs / 255.0) - 1.0
    

    arg_scope = pnasnet.pnasnet_large_arg_scope()

    with slim.arg_scope(arg_scope):
        logit, end_points = pnasnet.build_pnasnet_large(
            x1, num_classes=1001, is_training=False)
        print(end_points)
        prob = end_points['Predictions']
        y = tf.argmax(prob, axis=1)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, checkepoint_file)
        

        def preimg(img):
            ch = 3
            print(img.mode)
           
            if img.mode == 'RGBA':
                ch = 4

            imgnp = np.asarray(img.resize((image_size, image_size)),
                            dtype=np.float32).reshape(image_size, image_size, ch)
            return imgnp[:, :, :3]
        

        batchImg = [preimg(Image.open(imgfilename)) for imgfilename in sample_images]
        orgImg = [Image.open(imgfilename) for imgfilename in sample_images]

        yv, img_norm = sess.run([y, x1], feed_dict={input_imgs:batchImg})

        # print(yv, np.shape(yv))


        def showresult(yy, img_norm, img_org):
            plt.figure()
            p1 = plt.subplot(121)
            p2 = plt.subplot(122)

            p1.imshow(img_org)
            p1.axis('off')
            p1.set_title("organization image")

            p2.imshow((img_norm * 255).astype(np.uint8))
            p2.axis('off')
            p2.set_title("input image")

            plt.show()
            # print(yy)
            # print(labels[yy])
        
        for yy, img1, img2 in zip(yv, batchImg, orgImg):
            showresult(yy, img1, img2)


def getone(onestr):
    return onestr.replace(',', ' ')


if __name__ == "__main__":
    main()
