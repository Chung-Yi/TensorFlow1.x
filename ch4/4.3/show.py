
import numpy as np
from matplotlib import pyplot as plt

def showresult(subplot, title, thisimg):
    p = plt.subplot(subplot)
    p.axis('off')
    p.imshow(np.reshape(thisimg, (28, 28)))
    p.set_title(title)


def showimg(index, label, img, ntop):
    plt.figure(figsize=(20, 10))
    plt.axis('off')
    ntop = min(ntop, 9)
    print(index)

    for i in range(ntop):
        showresult(100+10*ntop+1+i, label[i], img[i])
    plt.show()
