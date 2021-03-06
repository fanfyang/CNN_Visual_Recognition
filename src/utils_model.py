import numpy as np
import tensorflow as tf
import os, sys, time, argparse
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

def fetch_data(path = '../data', resize = (224,224,3), file = False, dtype = '.jpg', cate_file = 'categories.txt', image_file = 'images.txt', filenames = False, shuffle = True):
	images = list()
	labels = list()
	files = list()
	if file == True:
		with open(os.path.join(path, cate_file), 'r') as f:
			temp = f.readlines()
			num_cate = []
			categories = []
			for item in temp:
				cate, num = item.rstrip('\n').split('\t')
				num_cate.append(int(num))
				categories.append(cate)
		num_fail = [0] * len(categories)
		with open(os.path.join(path, image_file), 'r') as f:
			for i in range(len(categories)):
				category = categories[i]
				for j in range(num_cate[i]):
					try:
						file_name = f.readline().rstrip('\n')
						image = ndimage.imread(os.path.join(path, 'img', category, file_name+dtype))
						if len(image.shape) != 3 and image.shape[3] != 3:
							print('dimension not agree:' + file_name + '; category:' + category + '\n')
							continue
						image_resized = imresize(image, resize)
						images.append(image_resized)
						files.append(file_name)
					except:
						num_fail[i] += 1
						print('fail:' + file_name + '; category:' + category + '; %d\n' %(j))
		labels = np.concatenate([[i] * (num_cate[i]-num_fail[i]) for i in range(len(categories))])
		idx = np.arange(len(labels))
		if shuffle:
			np.random.shuffle(idx)
		if filenames:
			files = [files[i] for i in idx]
			return (np.array(images,dtype=np.float32)[idx], labels[idx], categories, files)
		else:
			return (np.array(images,dtype=np.float32)[idx], labels[idx], categories)
	else:
		categories = [item for item in os.listdir(os.path.join(path,'img')) if item[0] != '.']
		for i in range(len(categories)):
			category = categories[i]
			for item in os.listdir(os.path.join(path, 'img', category)):
				if item.endswith('.jpg'):
					image = ndimage.imread(os.path.join(path, 'img', category, item))
					if len(image.shape) != 3 and image.shape[3] != 3:
						continue
					image_resized = imresize(image, resize)
					images.append(image_resized)
					labels.append(i)
		return (np.array(images,dtype=np.float32), np.array(labels), categories)

def fetch_data_2(path = '../data', dtype = '.jpg', cate_file = 'categories.txt', image_file = 'images.txt', shuffle = True):
	images = list()
	labels = list()
	with open(os.path.join(path, cate_file), 'r') as f:
		temp = f.readlines()
		num_cate = []
		categories = []
		for item in temp:
			cate, num = item.rstrip('\n').split('\t')
			num_cate.append(int(num))
			categories.append(cate)
	with open(os.path.join(path, image_file), 'r') as f:
		for i in range(len(categories)):
			category = categories[i]
			for j in range(num_cate[i]):
				images.append(f.readline().rstrip('\n'))
	labels = np.concatenate([[i] * num_cate[i] for i in range(len(categories))])
	if shuffle:
		idx = np.arange(len(labels))
		np.random.shuffle(idx)
		images = [images[i] for i in idx]
		labels = [labels[i] for i in idx]
	return (images, labels, categories)

def data_generator(images, labels, categories, batch_size, resize = (224,224,3), shuffle = True, dtype = '.jpg'):
	N = len(labels)
	if shuffle:
		idxs = np.arange(len(labels))
		np.random.shuffle(idxs)
	num_batches = (N+batch_size-1) // batch_size
	i = 0
	while True:
		i += 1
		images_data = list()
		labels_data = list()
		if i != num_batches:
			for j in range(batch_size):
				idx = idxs[(i-1)*batch_size+j]
				try:
					image = ndimage.imread(os.path.join('../data/img', categories[labels[idx]], images[idx]+dtype))
					if len(image.shape) != 3 and image.shape[3] != 3:
						continue
					image_resized = imresize(image, resize)
					images_data.append(image_resized)
					labels_data.append(labels[idx])
				except:
					continue
		else:
			for j in range(N-(num_batches-1)*batch_size):
				idx = idxs[(num_batches-1)*batch_size+j]
				try:
					image = ndimage.imread(os.path.join('../data/img', categories[labels[idx]], images[idx]+dtype))
					if len(image.shape) != 3 and image.shape[3] != 3:
						continue
					image_resized = imresize(image, resize)
					images_data.append(image_resized)
					labels_data.append(labels[idx])
				except:
					continue
			i = 0
			if shuffle:
				np.random.shuffle(idxs)
		yield (np.array(images_data,dtype=np.float32), np.array(labels_data))

def prepare_predict_data(file_path, channel_mean = np.array([ 203.89836428,  191.68313589,  180.50212764]), resize = (224,224,3)):
	num = len(file_path)
	images = list()
	for path in file_path:
		image = ndimage.imread(path)
		image_resized = imresize(image, resize).astype(np.float32)
		images.append(image_resized - channel_mean)
	return np.array(images, dtype = np.float32)

def parse_argument(args):
	para = {}
	para['lr'] = float(args.lr) if args.lr != None else 0.001
	para['decay_rate'] = float(args.dr) if args.dr != None else 0.9
	para['decay_steps'] = int(args.ds) if args.ds != None else 700
	para['l2'] = float(args.l2) if args.l2 != None else 0.0005
	para['batch_size'] = int(args.bs) if args.bs != None else 30
	para['num_epoch'] = int(args.ne) if args.ne != None else 20
	para['dropout'] = float(args.d) if args.d != None else 0.5
	para['num_classes'] = int(args.nc) if args.nc != None else 1000
	para['use_batch_norm'] = False if args.bn != None and args.bn == 'F' else True
	return para


class Config_Pic:
	def __init__(self, height = 224, width = 224, channels = 3):
		self.height = height
		self.width = width
		self.channels = channels

class Config:
	def __init__(self, lr = 0.001, decay_rate = 0.9, decay_steps = 700, l2 = 0.0005, batch_size = 30, num_epoch = 20, dropout = 0.5, num_classes = 1000, use_batch_norm = True):
		self.lr = lr
		self.decay_rate = decay_rate
		self.decay_steps = decay_steps
		self.l2 = l2
		self.batch_size = batch_size
		self.num_epoch = num_epoch
		self.dropout = dropout
		self.num_classes = num_classes
		self.use_batch_norm = use_batch_norm

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

	def save_parameters(self, sess, path, version):
		if not os.path.exists(path):
			os.makedirs(path)
		exec('np.savez(\'%s\''%(path+'para_' + version + '.npz') + ',' + ','.join(['%s = sess.run(self._parameters[\'%s\'])' %(key, key) for key in self._parameters]) + ')')

	# You might want to re-define this function for your model
	def predict(self, sess, image_path):
		images = prepare_predict_data(image_path, self._channel_mean)
		preds, features = sess.run([self._pred, self._feature], {self._input_placeholder:images, self._dropout_placeholder:1., self._is_training_placeholder:False})
		return preds, features

	def predict_label(self, sess, image_path):
		preds, = self.predict(sess, image_path)
		return [self._class_names[i] for i in preds]

	# You might want to re-define this function for your model
	def run_epoch(self, sess, X, y, shuffle = True, batch_per_print = 2):
		start = time.time()
		num_batches = X.shape[0] // self._config.batch_size
		idx = np.arange(X.shape[0])
		if shuffle:
			np.random.shuffle(idx)

		len_eq = 20
		batch_per_eq = (num_batches+len_eq-1)//len_eq

		total_loss = []
		accu = []

		for i in range(num_batches):
			batch_idx = idx[i*self._config.batch_size:(i+1)*self._config.batch_size]
			X_batch = X[batch_idx]
			y_batch = y[batch_idx]
			feed_dict = {self._input_placeholder:X_batch, self._label_placeholder:y_batch, self._dropout_placeholder:self._config.dropout, self._is_training_placeholder:True}
			loss, pred, _ = sess.run([self._loss, self._pred, self._train_op], feed_dict)
			# print('loss: ' + str(loss))
			total_loss.append(loss)
			accu.append(1.*np.sum(pred==y_batch)/self._config.batch_size)
			if (i+1)//batch_per_print*batch_per_print == i+1:
				num_eq = (i+1)//batch_per_eq
				sys.stdout.write('\r '+str(i+1)+' / '+str(num_batches)+' [' + '='*num_eq + ' '*(len_eq - num_eq) + '] - %0.2fs - loss: %0.4f - acc: %0.4f  '%(float(time.time()-start),float(np.mean(total_loss)),float(np.mean(accu))))
				sys.stdout.flush()
		batch_idx = idx[num_batches*self._config.batch_size:]
		X_batch = X[batch_idx]
		y_batch = y[batch_idx]
		feed_dict = {self._input_placeholder:X_batch, self._label_placeholder:y_batch, self._dropout_placeholder:self._config.dropout, self._is_training_placeholder:True}
		loss, pred, _ = sess.run([self._loss, self._pred, self._train_op], feed_dict)
		total_loss.append(loss)
		accu.append(1.*np.sum(pred==y_batch)/len(batch_idx))
		sys.stdout.write('\r '+str(num_batches)+' / '+str(num_batches)+' [' + '='*len_eq + '] - %0.2fs - loss: %0.4f - acc: %0.4f  \n'%(float(time.time()-start),float(np.mean(total_loss)),float(np.mean(accu))))
		sys.stdout.flush()

	# You might want to re-define this function for your model
	def train(self, sess, X_train, y_train, X_val, y_val, X_test = None, y_test = None, version = 'v', model = 'vgg'):
		val_acc_current_best = 0.0
		for i in range(self._config.num_epoch):
			print('Epoch %d / %d'%(i+1,self._config.num_epoch))
			self.run_epoch(sess, X_train, y_train)
			train_acc = 1-self.error(sess, X_train, y_train)
			val_acc = 1-self.error(sess, X_val, y_val)
			if X_test is not None:
				test_acc = 1-self.error(sess, X_test, y_test)
			print('train acc: %0.4f; val acc: %0.4f; test acc: %0.4f \n' % (train_acc, val_acc, test_acc))
			if val_acc > val_acc_current_best:
				val_acc_current_best = val_acc
				self.save_parameters(sess, '../model/' + model + '/',version)

	def train_2(self, sess, X_train, y_train, X_val, y_val, categories, resize = (224,224,3), shuffle = True, dtype = '.jpg', batch_per_print = 2):
		g_train = data_generator(X_train, y_train, categories, self._config.batch_size, resize, shuffle = shuffle, dtype = dtype)
		g_val = data_generator(X_val, y_val, categories, self._config.batch_size, resize, shuffle = shuffle, dtype = dtype)
		
		N_train = len(y_train)
		N_val = len(y_val)
		train_batches = (N_train+self._config.batch_size-1) // self._config.batch_size
		val_batches = (N_val+self._config.batch_size-1) // self._config.batch_size

		len_eq = 20
		batch_per_eq = (train_batches+len_eq-1)//len_eq

		for i in range(self._config.num_epoch):
			print('Epoch %d / %d'%(i+1,self._config.num_epoch))
			start = time.time()
			total_loss = []
			accu = []

			for j in range(train_batches):
				X_batch, y_batch = next(g_train)
				feed_dict = {self._input_placeholder:X_batch-self._channel_mean, self._label_placeholder:y_batch, self._dropout_placeholder:self._config.dropout, self._is_training_placeholder:True}
				loss, pred, _ = sess.run([self._loss, self._pred, self._train_op], feed_dict)
				total_loss.append(loss)
				accu.append(1.*np.sum(pred==y_batch)/self._config.batch_size)
				if (j+1)//batch_per_print*batch_per_print == j+1:
					num_eq = (j+1)//batch_per_eq
					sys.stdout.write('\r '+str(j+1)+' / '+str(train_batches)+' [' + '='*num_eq + ' '*(len_eq - num_eq) + '] - %0.2fs - loss: %0.4f - acc: %0.4f  '%(float(time.time()-start),float(np.mean(total_loss)),float(np.mean(accu))))
					sys.stdout.flush()
			sys.stdout.write('\r '+str(train_batches)+' / '+str(train_batches)+' [' + '='*len_eq + '] - %0.2fs - loss: %0.4f - acc: %0.4f  \n'%(float(time.time()-start),float(np.mean(total_loss)),float(np.mean(accu))))
			sys.stdout.flush()

			num_correct_train = 0
			num_correct_val = 0
			for _ in range(train_batches):
				X_batch, y_batch = next(g_train)
				feed_dict = {self._input_placeholder: X_batch-self._channel_mean, self._dropout_placeholder:1.0, self._is_training_placeholder:False}
				pred = sess.run(self._pred, feed_dict)
				num_correct_train += np.sum(pred == y_batch)
			for _ in range(val_batches):
				X_batch, y_batch = next(g_val)
				feed_dict = {self._input_placeholder: X_batch-self._channel_mean, self._dropout_placeholder:1.0, self._is_training_placeholder:False}
				pred = sess.run(self._pred, feed_dict)
				num_correct_val += np.sum(pred == y_batch)
			print('train acc: %0.4f; val acc: %0.4f \n' % (1.*num_correct_train/N_train, 1.*num_correct_val/N_val))

	# You might want to re-define this function for your model
	def error(self, sess, X, y, is_training = False):
		num_batches = X.shape[0] // self._config.batch_size
		num_correct = 0
		for i in range(num_batches):
			feed_dict = {self._input_placeholder: X[i*self._config.batch_size:(i+1)*self._config.batch_size], self._dropout_placeholder:1.0, self._is_training_placeholder:is_training}
			pred = sess.run(self._pred, feed_dict)
			label = y[i*self._config.batch_size:(i+1)*self._config.batch_size]
			num_correct += np.sum(pred == label)
		feed_dict = {self._input_placeholder: X[num_batches*self._config.batch_size:], self._dropout_placeholder:1.0, self._is_training_placeholder:is_training}
		pred = sess.run(self._pred, feed_dict)
		label = y[num_batches*self._config.batch_size:]
		num_correct += np.sum(pred == label)
		return 1 - num_correct * 1.0 / X.shape[0]

	def extract_feature(self, sess, X, is_training = False, version = None):
		num_batches = X.shape[0] // self._config.batch_size
		features = list()
		for i in range(num_batches):
			feed_dict = {self._input_placeholder: X[i*self._config.batch_size:(i+1)*self._config.batch_size], self._dropout_placeholder:1.0, self._is_training_placeholder:is_training}
			feature = sess.run(self._feature, feed_dict)
			features.append(feature)
		feed_dict = {self._input_placeholder: X[num_batches*self._config.batch_size:], self._dropout_placeholder:1.0, self._is_training_placeholder:is_training}
		feature = sess.run(self._feature, feed_dict)
		features.append(feature)
		features = np.concatenate(features, axis = 0)
		if version is not None:
			np.savez('../model/vgg/feature_' + version + '.npz', feature = features)
		return features