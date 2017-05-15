from utils_model import *

class model_test(model):
	def __init__(self):
		self._input_placeholder = tf.placeholder(dtype = tf.float32, shape = (None, 10))
		self._label_placeholder = tf.placeholder(dtype = tf.int32, shape = (None))
		self._dropout_placeholder = tf.placeholder(dtype = tf.float32)

		self._config = Config(batch_size = 10, num_classes = 2, l2 = 0, lr = 0.001, num_epoch = 5)

		self._parameters = dict()

		W = tf.get_variable('W',[10,self._config.num_classes])
		b = tf.get_variable('b',[self._config.num_classes])
		self._score = tf.matmul(self._input_placeholder,W) + b
		self._parameters['W'] = W
		self._parameters['b'] = b

		self._pred = tf.argmax(self._score, 1)

		self._loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = tf.one_hot(self._label_placeholder, self._config.num_classes), logits = self._score))
		self._loss += 0.5 * self._config.l2 * tf.reduce_sum(tf.square(W))

		optimizer = tf.train.AdamOptimizer(self._config.lr)
		self._train_op = optimizer.minimize(self._loss)

if __name__ == '__main__':
	test = model_test()
	X = np.random.normal(size = (11000,10))
	A = np.random.normal(size = (10,2))
	temp = np.sum(np.dot(X,A),1)
	mean = np.mean(temp)
	y = temp > mean
	y = y.astype(np.int32)
	X_train = X[:10000]
	X_val = X[10000:]
	y_train = y[:10000]
	y_val = y[10000:]
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		test.train(sess,X_train,y_train,X_val,y_val)
		test.save_parameters(sess, '../model/test/')
