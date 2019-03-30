import fasttext
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
# from tempfile import NamedTemporaryFile
import re
import codecs

'''
X: list of sentences
y: list of labels
data example: Example: __label__moviereview The film's two major strengths come down to the two most important ingredients - cast and story.
'''
global LABEL
LABEL = {'not met':'N', 'met':'M'}

def writelist(f, data_li):
    for e in data_li: f.write(e.strip() + '\n')
    return str(f.name)

class FasttextClassifier(BaseEstimator, TransformerMixin):
	'''use fasttext for text classification
	'''
	def __init__(self, lr=0.01, epoch=5, dim=100, ws=10):
		self.lr = lr
		self.epoch = epoch
		self.dim = dim
		self.ws = ws
		# self.set_params(*kwargs)

	def set_params(self, params):
		for k, v in params.items(): exec("self.{} = {}".format(k, v))
		return self

	def get_params(self, deep=True):
		return dict(lr=self.lr, epoch=self.epoch)

	def _preprocess(self, X, y):
		''' put X and y in a file as the example
		'''
		if y != None:
			new_y_li = ['__label__{}'.format(LABEL[e]) for e in y ]
			return X, new_y_li
		else: return X

	def train(self,X, skgmodel, cbowmodel):
		'''
		X is a list of utf-8 text
		'''
		skgmodel = fasttext.skipgram(X, skgmodel)
		cbowmoel = fasttext.cbow(X, cbowmodel)
		return self

	def fit(self, X, y, model_name):
		train_data = ['{} {}'.format(f,e) for e,f in zip(*self._preprocess(X, y))]
		train_f = codecs.open('train.txt', 'w')
		t_fname = writelist(train_f, train_data)
		self.classifier = fasttext.supervised(t_fname, model_name, label_prefix ='__label__', loss="ns", dim=self.dim, ws=self.ws, lr=self.lr, epoch=self.epoch, thread=12, silent=False)
		train_f.close()
		del train_data
		return self

	def load_model(self, model_directory_path):
		print(model_directory_path)
		self.classifier = fasttext.load_model(model_directory_path,label_prefix='__label__')
		return self

	def test(self, X, y):
		test_data = ['{} {}'.format(f,e) for e, f in zip(*self._preprocess(X, y))]
		test_f = codecs.open('test.txt', 'w')
		t_fname = writelist(test_f, test_data)
		test_result = self.classifier.test(t_fname)
		print('test sample:', len(test_data))
		print('Precision:', test_result.precision)
		print('Recall', test_result.recall)
		print('No. of examples:', test_result.nexamples)
		test_f.close()
		del test_data
		return 0

	def predict(self, X_test, y=None):
		y_pred = self.classifier.predict(X_test)
		return y_pred

	def predict_prob(self,X_test, K=4):
		y_prebpro = self.classifier.predict_proba(X_test, k=4)
		return y_prebpro


		
