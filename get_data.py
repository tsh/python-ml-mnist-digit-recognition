import asyncio
import gzip
import os
from itertools import chain

import requests
from mnist import MNIST
from sklearn import svm
from sklearn.model_selection import train_test_split


def get_mnist():
    mnist_urls = ['http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
                  'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
                  'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
                  'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz']

    mnist_names = []
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    async def download(url):
        fname = url.split('/')[-1]
        dest = os.path.join(data_dir, fname)
        mnist_names.append(dest)

        if os.path.isfile(dest):
            return

        r = requests.get(url, stream=True)
        with open(dest, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)


    ioloop = asyncio.get_event_loop()
    tasks = [ioloop.create_task(download(url)) for url in mnist_urls]
    wait_tasks = asyncio.wait(tasks)
    ioloop.run_until_complete(wait_tasks)
    ioloop.close()


    # TODO: fix gzip unpacking
    for gname in mnist_names:
        out_name = gname.split('.')[0]
        res_file = os.path.join(data_dir, out_name)
        with gzip.open(gname, 'rb') as input:
            with gzip.open(res_file, 'wb') as output:
                content = input.read()
                output.write(content)


def prep_model():
    mndata = MNIST('data')
    images, labels = mndata.load_training()

    def threshold(array):
        for i, val in enumerate(array):
            if val > 127:
                array[i] = 1
            else:
                array[i] = 0

    images = images[:1000]
    labels = labels[:1000]

    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)
    for image in chain(train_images, test_images):
        threshold(image)

    clf = svm.SVC(probability=True)
    clf.fit(train_images, train_labels)
    print(clf.score(test_images, test_labels))


if __name__ == '__main__':
    # get_mnist()
    prep_model()