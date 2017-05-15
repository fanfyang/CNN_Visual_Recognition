import numpy as np
import tensorflow as tf
#import matplotlib.pyplot as plt 
import os, sys, time
from scipy import ndimage
from scipy.misc import imresize, imsave

def prepare_train_data(num_per_cate = 50000, path = '../data', dtype = '.jpg'):
	categories = [item for item in os.listdir(os.path.join(path, 'img')) if item[0] != '.' and item != 'Gift Cards Store' and item != 'Office Products']
	num_cate = []
	with open(os.path.join(path, 'images.txt'), 'w') as f:
		for i in range(len(categories)):
			category = categories[i]
			images = [item for item in os.listdir(os.path.join(path, 'img', category)) if item.endswith(dtype)]
			np.random.shuffle(images)
			for j in range(min(num_per_cate,len(images))):
				f.write(images[j].rstrip(dtype)+'\n')
			num_cate.append(min(num_per_cate,len(images)))
	with open(os.path.join(path, 'categories.txt'), 'w') as f:
		for i in range(len(categories)):
			f.write(categories[i]+'\t%d\n'%(num_cate[i]))

def fetch_data(path = '../data', resize = (224,224,3), file = False, dtype = '.jpg'):
	images = list()
	labels = list()
	if file == True:
		with open(os.path.join(path,'categories.txt'), 'r') as f:
			temp = f.readlines()
			num_cate = []
			categories = []
			for item in temp:
				cate, num = item.rstrip('\n').split('\t')
				num_cate.append(int(num))
				categories.append(cate)
		labels = np.concatenate([[i] * num_cate[i] for i in range(len(categories))])
		with open(os.path.join(path,'images.txt'), 'r') as f:
			for i in range(len(categories)):
				category = categories[i]
				for j in range(num_cate[i]):
					image = ndimage.imread(os.path.join(path, 'img', category, f.readline().rstrip('\n')+dtype))
					image_resized = imresize(image, resize)
					images.append(image_resized)
		return (np.array(images), labels, categories)
	else:
		categories = [item for item in os.listdir(os.path.join(path,'img')) if item[0] != '.']
		for i in range(len(categories)):
			category = categories[i]
			for item in os.listdir(os.path.join(path, 'img', category)):
				if item.endswith('.jpg'):
					image = ndimage.imread(os.path.join(path, 'img', category, item))
					image_resized = imresize(image, resize)
					images.append(image_resized)
					labels.append(i)
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
		self._config = None
		self._class_names = None
		self._parameters = None
		self._score = None
		self._loss = None
		self._pred = None
		self._train_op = None

	def load_parameters(self, sess, path, rand_init = []):
		parameters = np.load(path)
		for key in parameters:
			if key not in rand_init:
				sess.run(self._parameters[key].assign(parameters[key]))

	def save_parameters(self, sess, path):
		if not os.path.exists(path):
			os.makedirs(path)
		exec('np.savez(\'%s\''%(path+'para.npz') + ',' + ','.join(['%s = sess.run(self._parameters[\'%s\'])' %(key, key) for key in self._parameters]) + ')')

	# You might want to re-define this function for your model
	def predict(self, sess, image_path):
		images = prepare_predict_data(image_path, self._channel_mean)
		preds, = sess.run([self._pred], {self._input_placeholder:images, self._dropout_placeholder:1.})
		return preds

	def predict_label(self, sess, image_path):
		preds = self.predict(sess, image_path)
		return [self._class_names[i] for i in preds]

	# You might want to re-define this function for your model
	def run_epoch(self, sess, X, y, shuffle = True, batch_per_print = 100):
		start = time.time()
		num_batches = X.shape[0] // self._config.batch_size
		idx = np.arange(X.shape[0])
		if shuffle:
			np.random.shuffle(idx)

		len_eq = 20
		batch_per_eq = (num_batches+len_eq-1)//len_eq

		total_loss = []
		accu = []

		for i in xrange(num_batches):
			batch_idx = idx[i*self._config.batch_size:(i+1)*self._config.batch_size]
			X_batch = X[batch_idx]
			y_batch = y[batch_idx]
			feed_dict = {self._input_placeholder:X_batch, self._label_placeholder:y_batch, self._dropout_placeholder:self._config.dropout}
			loss, pred, _ = sess.run([self._loss, self._pred, self._train_op], feed_dict)
			total_loss.append(loss)
			accu.append(1.*np.sum(pred==y_batch)/self._config.batch_size)
			if (i+1)/batch_per_print*batch_per_print == i+1:
				num_eq = (i+1)//batch_per_eq
				sys.stdout.write('\r '+str(i+1)+' / '+str(num_batches)+' [' + '='*num_eq + ' '*(len_eq - num_eq) + '] - %0.2fs - loss: %0.4f - acc: %0.4f  '%(float(time.time()-start),float(np.mean(total_loss)),float(np.mean(accu))))
				sys.stdout.flush()
		sys.stdout.write('\r '+str(num_batches)+' / '+str(num_batches)+' [' + '='*len_eq + '] - %0.2fs - loss: %0.4f - acc: %0.4f  \n'%(float(time.time()-start),float(np.mean(total_loss)),float(np.mean(accu))))
		sys.stdout.flush()

	# You might want to re-define this function for your model
	def train(self, sess, X_train, y_train, X_val, y_val):
		for i in xrange(self._config.num_epoch):
			print('Epoch %d / %d'%(i+1,self._config.num_epoch))
			self.run_epoch(sess, X_train, y_train)
			print('train acc: %0.4f; val acc: %0.4f \n' % (1-self.error(sess, X_train, y_train),1-self.error(sess, X_val, y_val)))

	# You might want to re-define this function for your model
	def error(self, sess, X, y):
		feed_dict = {self._input_placeholder: X, self._dropout_placeholder:1.0}
		pred = sess.run(self._pred, feed_dict)
		num = X.shape[0]
		num_correct = np.sum(pred == y)
		return 1 - num_correct * 1.0 / num