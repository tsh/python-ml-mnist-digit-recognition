import os

from mnist import MNIST
import sklearn
from sklearn import svm
from sklearn.externals import joblib

from utils import threshold


class BaseClassifier(object):
    CLASSIFIERS_DIR = os.path.abspath(os.path.abspath(os.path.join(os.path.dirname(__file__), 'classifiers')))

    def get_training(self):
        mndata = MNIST('data')
        images, labels = mndata.load_training()
        return images, labels

    def dump(self, clf):
        if not os.path.exists(self.CLASSIFIERS_DIR):
            os.makedirs(self.CLASSIFIERS_DIR)
        joblib.dump(clf, os.path.join(self.CLASSIFIERS_DIR, self.filename()))

    @property
    def classifier(self):
        return joblib.load(os.path.join(self.CLASSIFIERS_DIR, self.filename()))


class SVCClassifier(BaseClassifier):
    def train(self):
        images, labels = self.get_training()
        map(threshold, images)
        self._clf = svm.SVC(probability=True)
        self._clf.fit(images, labels)
        self.dump()
    
    def filename(self):
        return 'svc.pkl'
