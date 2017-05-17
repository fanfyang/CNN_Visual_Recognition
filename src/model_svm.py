from utils_model import *

class svm(model):
	def __init__(self, config):
		# self._config = Config(batch_size = 10, num_classes = 2, l2 = 0, lr = 0.001, num_epoch = 5)
		self._config = config
		self._config_fig = Config_pic()
		self._parameters = dict()

		self._input_placeholder = tf.placeholder(dtype = tf.float32, shape = (None, self._config_fig.width * self._config_fig.height * self._config_fig.channels))
		self._label_placeholder = tf.placeholder(dtype = tf.int32, shape = (None))
		self._dropout_placeholder = tf.placeholder(dtype = tf.float32)

		self._is_training_placeholder = tf.placeholder(dtype = tf.bool)

		W = tf.get_variable('W',[self._config_fig.width * self._config_fig.height * self._config_fig.channels, self._config.num_classes])
		b = tf.get_variable('b',[self._config.num_classes])
		self._score = tf.matmul(tf.reshape(self._input_placeholder,[-1,self._config_fig.width * self._config_fig.height * self._config_fig.channels]),W) + b
		self._parameters['W'] = W
		self._parameters['b'] = b

		self._pred = tf.argmax(self._score, 1)

		self._loss = tf.losses.hinge_loss(labels = tf.one_hot(self._label_placeholder, self._config.num_classes), logits = self._score)
		
		self._loss += 0.5 * self._config.l2 * tf.reduce_sum(tf.square(W))

		optimizer = tf.train.AdamOptimizer(self._config.lr)
		self._train_op = optimizer.minimize(self._loss)

