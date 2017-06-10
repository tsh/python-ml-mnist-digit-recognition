from mnist import MNIST
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

mndata = MNIST('data')

images, labels = mndata.load_training()

images = images[:5000]
labels = labels[:5000]

def threshold(array):
    for i, val in enumerate(array):
        if val > 0:
            array[i] = 1



train_images, test_images,train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)
for image in train_images:
    threshold(image)


img=np.array(train_images[1])
plt.hist(img)
img=img.reshape((28,28))
plt.show()
# plt.imshow(img, cmap='gray')
# plt.show()