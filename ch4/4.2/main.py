import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle



# generating data in memory
def GenerateData(training_epochs, batchsize = 100):
    for i in range(training_epochs):
        train_X = np.linspace(-1, 1, batchsize) # 100 float numbers between -1 ~ 1
        train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.3
        yield shuffle(train_X, train_Y), i



def main():
    
    # define two tensor
    Xinput = tf.placeholder("float", (None))
    Yinput = tf.placeholder("float", (None))
    
   
    # get output data from generator
    
    training_epochs = 20
    with tf.Session() as sess:
           for (x, y), epoch in GenerateData(training_epochs):
               xv, yv = sess.run([Xinput, Yinput], feed_dict={Xinput: x, Yinput: y})
               print(epoch, "| x.shape:", np.shape(xv), "| x[:3]:", xv[:3])
               print(epoch, "| y.shape:", np.shape(yv), "| y[:3]:", yv[:3])

    train_data = list(GenerateData(1))[0]
    plt.plot(train_data[0][0], train_data[0][1], 'ro', label='Original data')
    plt.legend()
    plt.show()
    
    



if __name__ == "__main__":
    main()