from mnist import MNIST
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm

mndata = MNIST('data')

images, labels = mndata.load_training()

images = images[:5000]
labels = labels[:5000]

# arr = np.array(images[1700]).reshape((28, 28))
# fig = plt.figure()
# plt.imshow(arr, cmap='gray')
# plt.show()

train_images, test_images,train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)
img=np.array(train_images[1])
img=img.reshape((28,28))

clf = svm.SVC()
clf.fit(train_images, train_labels)
print(clf.score(test_images,test_labels))

