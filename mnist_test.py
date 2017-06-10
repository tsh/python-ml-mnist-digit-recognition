from mnist import MNIST
import numpy as np
import matplotlib.pyplot as plt

mndata = MNIST('data')

images, labels = mndata.load_training()


arr = np.array(images[700]).reshape((28, 28))
print (labels[700], arr)

fig = plt.figure()
plt.imshow(arr)
plt.show()