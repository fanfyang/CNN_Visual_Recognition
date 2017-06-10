from utils_model import *
import tensorflow as tf

class Config_NN:
	def __init__(self, lr = 0.001, decay_rate = 0.9, decay_steps = 700, l2 = 0.0005, input_dim = 8192, batch_size = 128, num_epoch = 20, dropout = 0.5):
		self.lr = lr
		self.decay_rate = decay_rate
		self.decay_steps = decay_steps
		self.l2 = l2
		self.input_dim = input_dim
		self.batch_size = batch_size
		self.num_epoch = num_epoch
		self.dropout = dropout

class model_nn(model):
	def __init__(self, config):
		self._config = config
		self._input_placeholder = tf.placeholder(dtype = tf.float32, shape = (None, self._config.input_dim))
		self._output_placeholder = tf.placeholder(dtype = tf.float32, shape = (None,))
		self._parameters = dict()

		# with tf.variable_scope('nn/fc1') as scope:
		# 	Wfc1 = tf.get_variable('W',[self._config.input_dim,512], trainable = True, initializer = tf.contrib.layers.xavier_initializer())
		# 	bfc1 = tf.get_variable('b',[512], trainable = True, initializer = tf.contrib.layers.xavier_initializer())
		# 	fc1 = tf.nn.relu(tf.matmul(self._input_placeholder, Wfc1) + bfc1, name = 'fc1')
		# 	self._parameters['fc1_W'] = Wfc1
		# 	self._parameters['fc1_b'] = bfc1
		# 	tf.add_to_collection('Reg', tf.reduce_sum(tf.square(Wfc1)))

		# with tf.variable_scope('nn/fc2') as scope:
		# 	Wfc2 = tf.get_variable('W',[512,32], trainable = True, initializer = tf.contrib.layers.xavier_initializer())
		# 	bfc2 = tf.get_variable('b',[32], trainable = True, initializer = tf.contrib.layers.xavier_initializer())
		# 	fc2 = tf.nn.relu(tf.matmul(fc1, Wfc2) + bfc2, name = 'fc2')
		# 	self._parameters['fc2_W'] = Wfc2
		# 	self._parameters['fc2_b'] = bfc2
		# 	tf.add_to_collection('Reg', tf.reduce_sum(tf.square(Wfc2)))

		with tf.variable_scope('nn/fc3') as scope:
			Wfc3 = tf.get_variable('W',[self._config.input_dim,1], trainable = True, initializer = tf.contrib.layers.xavier_initializer())
			bfc3 = tf.get_variable('b',[1], trainable = True, initializer = tf.contrib.layers.xavier_initializer())
			fc3 = tf.matmul(self._input_placeholder, Wfc3) + bfc3
			self._parameters['fc3_W'] = Wfc3
			self._parameters['fc3_b'] = bfc3
			tf.add_to_collection('Reg', tf.reduce_sum(tf.square(Wfc3)))

		with tf.variable_scope('nn/fc4') as scope:
			Wfc4 = tf.get_variable('W',[1,1], trainable = True, initializer = tf.contrib.layers.xavier_initializer())
			bfc4 = tf.get_variable('b',[1], trainable = True, initializer = tf.contrib.layers.xavier_initializer())
			self._score = tf.matmul(fc3, Wfc4) + bfc4
			self._parameters['fc4_W'] = Wfc4
			self._parameters['fc4_b'] = bfc4
			tf.add_to_collection('Reg', tf.reduce_sum(tf.square(Wfc4)))

		self._pred = tf.squeeze(self._score)
		self._loss = tf.reduce_mean(tf.square(self._pred - self._output_placeholder))
		for reg in tf.get_collection("Reg"):
			self._loss += 0.5 * self._config.l2 * reg

		# self._global_step = tf.Variable(0, trainable = False)
		# lr = tf.train.exponential_decay(self._config.lr, decay_rate = self._config.decay_rate, global_step = self._global_step, decay_steps = self._config.decay_steps)

		optimizer = tf.train.AdamOptimizer(self._config.lr)
		self._train_op = optimizer.minimize(self._loss)

	def run_epoch(self, sess, X, y, shuffle = True, batch_per_print = 2):
		start = time.time()
		num_batches = X.shape[0] // self._config.batch_size
		idx = np.arange(X.shape[0])
		if shuffle:
			np.random.shuffle(idx)

		len_eq = 20
		batch_per_eq = (num_batches+len_eq-1)//len_eq

		total_loss = []

		for i in range(num_batches):
			batch_idx = idx[i*self._config.batch_size:(i+1)*self._config.batch_size]
			X_batch = X[batch_idx]
			y_batch = y[batch_idx]
			feed_dict = {self._input_placeholder:X_batch, self._output_placeholder:y_batch}
			loss, _ = sess.run([self._loss, self._train_op], feed_dict)
			total_loss.append(loss)
			if (i+1)//batch_per_print*batch_per_print == i+1:
				num_eq = (i+1)//batch_per_eq
				sys.stdout.write('\r '+str(i+1)+' / '+str(num_batches)+' [' + '='*num_eq + ' '*(len_eq - num_eq) + '] - %0.2fs - loss: %0.4f  '%(float(time.time()-start),float(np.mean(total_loss))))
				sys.stdout.flush()
		batch_idx = idx[num_batches*self._config.batch_size:]
		X_batch = X[batch_idx]
		y_batch = y[batch_idx]
		feed_dict = {self._input_placeholder:X_batch, self._output_placeholder:y_batch}
		loss, _ = sess.run([self._loss, self._train_op], feed_dict)
		total_loss.append(loss)
		sys.stdout.write('\r '+str(num_batches)+' / '+str(num_batches)+' [' + '='*len_eq + '] - %0.2fs - loss: %0.4f  \n'%(float(time.time()-start),float(np.mean(total_loss))))
		sys.stdout.flush()

	def train(self, sess, X_train, y_train, X_val, y_val, X_test, y_test, version = 'v'):
		val_error_current_best = float('Inf')
		for i in range(self._config.num_epoch):
			print('Epoch %d / %d'%(i+1,self._config.num_epoch))
			self.run_epoch(sess, X_train, y_train)
			train_error = self.error(sess, X_train, y_train)
			val_error = self.error(sess, X_val, y_val)
			test_error = self.error(sess, X_test, y_test)
			print('train error: %0.4f; val error: %0.4f; test error: %0.4f \n' % (train_error, val_error, test_error))
			if val_error < val_error_current_best:
				val_error_current_best = val_error
				self.save_parameters(sess, '../model/similarity/',version)

	def error(self, sess, X, y):
		num_batches = X.shape[0] // self._config.batch_size
		e = 0.0
		for i in range(num_batches):
			feed_dict = {self._input_placeholder: X[i*self._config.batch_size:(i+1)*self._config.batch_size]}
			pred = sess.run(self._pred, feed_dict)
			similarity = y[i*self._config.batch_size:(i+1)*self._config.batch_size]
			e += np.sum((pred - similarity) ** 2)
		feed_dict = {self._input_placeholder: X[num_batches*self._config.batch_size:]}
		pred = sess.run(self._pred, feed_dict)
		similarity = y[num_batches*self._config.batch_size:]
		e += np.sum((pred - similarity) ** 2)
		return np.sqrt(e / X.shape[0])
