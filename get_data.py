import asyncio
import gzip
import os
from itertools import chain

import requests
from mnist import MNIST
import sklearn
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from utils import threshold


DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))
CLASSIFIERS_DIR = os.path.abspath(os.path.abspath(os.path.join(os.path.dirname(__file__), 'classifiers')))


def get_mnist():
    """
    Download and unpack MNIST data.
    :return: 
    """
    mnist_urls = ['http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
                  'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
                  'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
                  'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz']

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    async def download(url):
        fname = url.split('/')[-1]
        dest = os.path.join(DATA_DIR, fname)

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

    for gzipped in os.listdir(DATA_DIR):
        gf = gzip.GzipFile(os.path.join(DATA_DIR, gzipped))
        out_name = os.path.join(DATA_DIR, os.path.basename(gzipped).split('.')[0])
        with open(out_name, 'wb') as out:
            out.write(gf.read())
        gf.close()


def prep_model():
    """
    Build and train models and save them for future use. 
    """
    mndata = MNIST('data')
    images, labels = mndata.load_training()

    images = images[:2000]
    labels = labels[:2000]

    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)
    for image in chain(train_images, test_images):
        threshold(image)

    clf = svm.SVC(probability=True)
    clf.fit(train_images, train_labels)
    print(clf.score(test_images, test_labels))
    if not os.path.exists(CLASSIFIERS_DIR):
        os.makedirs(CLASSIFIERS_DIR)
    joblib.dump(clf, os.path.join(CLASSIFIERS_DIR, 'svc.pkl'))
    print(sklearn.__version__)


if __name__ == '__main__':
    get_mnist()
    prep_model()