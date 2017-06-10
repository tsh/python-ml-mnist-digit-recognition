from statistics import mean
from mnist import MNIST
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from itertools import chain
from sklearn import svm
from sklearn.model_selection import train_test_split

mndata = MNIST('data')

images, labels = mndata.load_training()

images = images
labels = labels

def threshold(array):
    for i, val in enumerate(array):
        if val > 127:
            array[i] = 1
        else:
            array[i] = 0


train_images, test_images, train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)
for image in chain(train_images, test_images):
    threshold(image)


img=np.array(train_images[1])


clf = svm.SVC()
clf.fit(train_images, train_labels)
print(clf.score(test_images, test_labels))

i = Image.open('test.png')
iar = np.array(i)
np.delete(iar, [3], axis=1)
balanced = []
for row in iar:
    for pix in row:
        balanced.append(mean(pix))

threshold(balanced)
nbalanced = np.array(balanced)

# img=balanced.reshape((28,28))
# plt.imshow(np.array(test_images[2]).reshape((28,28)), cmap='gray')
# plt.show()
results=clf.predict(balanced)
print(results)