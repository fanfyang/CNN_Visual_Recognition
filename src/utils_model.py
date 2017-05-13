import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt 
import os, time
from scipy import ndimage
from scipy.misc import imresize, imsave

def fetch_data(path = '../data/img', resize = (224,224,3)):
	images = list()
	labels = list()
	categories = [item for item in os.listdir(path) if item[0] != '.']
	for i in range(len(categories)):
		category = categories[i]
		for item in os.listdir(os.path.join(path, category)):
			if item.endswith('.jpg'):
				image = ndimage.imread(os.path.join(path, category, item))
				image_resized = imresize(image, resize)
				images.append(image_resized)
				labels.append(i)
			break
	return (np.array(images), np.array(labels), categories)

def prepare_predict_data(file_path, channel_mean = np.array([ 203.89836428,  191.68313589,  180.50212764]), resize = (224,224,3)):
	num = len(file_path)
	images = list()
	for path in file_path:
		image = ndimage.imread(path)
		image_resized = imresize(image, resize).astype(np.float32)
		images.append(image_resized - channel_mean)
	return np.array(images, dtype = np.float32)

class Config_Pic:
	def __init__(self, height = 224, width = 224, channels = 3):
		self.height = height
		self.width = width
		self.channels = channels

class Config:
	def __init__(self, lr = 0.025, l2 = 0.0005, batch_size = 30, num_epoch = 20, dropout = 0.5, num_classes = 1000):
		self.lr = lr
		self.l2 = l2
		self.batch_size = batch_size
		self.num_epoch = num_epoch
		self.dropout = dropout
		self.num_classes = num_classes

class model(object):
	def __init__(self):
		self._parameters = None
		self._score = None
		self._loss = None
		self._pred = None
		self._train_op = None

	def load_parameters(self, sess, path):
		parameters = np.load(path)
		for key in parameters:
			sess.run(self._parameters[key].assign(parameters[key]))

	def save_parameters(self, sess, path):
		exec('np.savez(\'%s\''%(path) + ',' + ','.join(['%s = sess.run(self._parameters[\'%s\'])' %(key, key) for key in self._parameters]) + ')')

	# You might want to re-define this function for your model
	def predict(self, sess, image_path):
		images = prepare_predict_data(image_path, self._channel_mean)
		preds, = sess.run([self._pred], {self._input_placeholder:images, self._dropout_placeholder:1.})
		return preds

	# You might want to re-define this function for your model
	def run_epoch(self, sess):
		pass

	# You might want to re-define this function for your model
	def train(self, sess):
		pass
		# for i in xrange(self._config.num_epoch):
		# 	start = time.time()
		# 	print 'Epoch %d: ' %i
		# 	self.run_epoch(sess)
		# 	print 'Elapse Time: {}\n'.format(time.time() - start)

	# You might want to re-define this function for your model
	def error(self, sess, X, y):
		feed_dict = {self._input_placeholder: X, self._dropout_placeholder:1.0}
		pred = sess.run(self._pred, feed_dict)
		num = X.shape[0]
		num_correct = np.sum(pred == y)
		return 1 - num_correct * 1.0 / num