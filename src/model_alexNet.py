from utils_model import *
import tensorflow as tf
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir, 'data/vgg16')))
from imagenet_classes import *

class model_alexNet(model):
	def __init__(self,config):
		self._config_pic = Config_Pic()
		self._config = config

		self._input_placeholder = tf.placeholder(dtype = tf.float32, shape = (None, self._config_pic.height, self._config_pic.width, self._config_pic.channels))
		self._label_placeholder = tf.placeholder(dtype = tf.int32, shape = (None,))
		self._dropout_placeholder = tf.placeholder(dtype = tf.float32)

		self._channel_mean = np.array([203.89836428,  191.68313589,  180.50212764])
		global class_names
		self._class_names = class_names
		self._parameters = dict()

		with tf.variable_scope('alexNet/conv1') as scope:
			Wconv1 = tf.get_variable('W', [11,11,self._config_pic.channels, 96], trainable = True, initializer = tf.contrib.layers.xavier_initializer())
			bconv1 = tf.get_variable('b', [96], trainable = True, initializer = tf.contrib.layers.xavier_initializer())
			conv1 = tf.nn.conv2d(self._input_placeholder, Wconv1, [1,4,4,1], padding = 'VALID') + bconv1
			relu1 = tf.nn.relu(conv1, name = 'relu1')
			self._parameters['conv1_W'] = Wconv1
			self._parameters['conv1_b'] = bconv1
			tf.add_to_collection('Reg', tf.reduce_sum(tf.square(Wconv1)))

		with tf.variable_scope('alexNet/pool1') as scope:
			pool1 = tf.nn.max_pool(relu1, [1,3,3,1], [1,2,2,1], padding = 'VALID', name = 'pool')

		with tf.variable_scope('alexNet/bn1') as scope:
			bn1 = tf.layers.batch_normalization(pool1, trainable = True)			

		with tf.variable_scope('alexNet/conv2') as scope:
			Wconv2 = tf.get_variable('W', [5,5,96,256], trainable = True, initializer = tf.contrib.layers.xavier_initializer())
			bconv2 = tf.get_variable('b', [256], trainable = True, initializer = tf.contrib.layers.xavier_initializer())
			conv2 = tf.nn.conv2d(bn1, Wconv2, [1,1,1,1], padding = 'SAME') + bconv2
			relu2 = tf.nn.relu(conv2, name = 'relu2')
			self._parameters['conv2_W'] = Wconv2
			self._parameters['conv2_b'] = bconv2
			tf.add_to_collection('Reg', tf.reduce_sum(tf.square(Wconv2)))

		with tf.variable_scope('alexNet/pool2') as scope:
			pool2 = tf.nn.max_pool(relu2, [1,3,3,1], [1,2,2,1], padding = 'VALID', name = 'pool')

		with tf.variable_scope('alexNet/bn2') as scope:
			bn1 = tf.layers.batch_normalization(pool2, trainable = True)

		with tf.variable_scope('alexNet/conv3') as scope:
			Wconv3 = tf.get_variable('W', [3,3,256,384], trainable = True, initializer = tf.contrib.layers.xavier_initializer())
			bconv3 = tf.get_variable('b', [384], trainable = True, initializer = tf.contrib.layers.xavier_initializer())
			conv3 = tf.nn.conv2d(bn1, Wconv3, [1,1,1,1], padding = 'SAME') + bconv3
			relu3 = tf.nn.relu(conv3, name = 'relu3')
			self._parameters['conv3_W'] = Wconv3
			self._parameters['conv3_b'] = bconv3
			tf.add_to_collection('Reg', tf.reduce_sum(tf.square(Wconv3)))

		with tf.variable_scope('alexNet/conv4') as scope:
			Wconv4 = tf.get_variable('W',[3,3,384,384], trainable = True, initializer = tf.contrib.layers.xavier_initializer())
			bconv4 = tf.get_variable('b',[384], trainable = True, initializer = tf.contrib.layers.xavier_initializer())
			conv4 = tf.nn.conv2d(relu3, Wconv4, [1,1,1,1], padding = 'SAME') + bconv4
			relu4 = tf.nn.relu(conv4, name = 'relu4')
			self._parameters['conv4_W'] = Wconv4
			self._parameters['conv4_b'] = bconv4
			tf.add_to_collection('Reg', tf.reduce_sum(tf.square(Wconv4)))

		with tf.variable_scope('alexNet/conv5') as scope:
			Wconv5 = tf.get_variable('W',[3,3,384,256], trainable = True, initializer = tf.contrib.layers.xavier_initializer())
			bconv5 = tf.get_variable('b',[256], trainable = True, initializer = tf.contrib.layers.xavier_initializer())
			conv5 = tf.nn.conv2d(relu4, Wconv5, [1,1,1,1], padding = 'SAME') + bconv5
			relu5 = tf.nn.relu(conv5, name = 'relu5')
			self._parameters['conv5_W'] = Wconv5
			self._parameters['conv5_b'] = bconv5
			tf.add_to_collection('Reg', tf.reduce_sum(tf.square(Wconv5)))		

		with tf.variable_scope('alexNet/pool3') as scope:
			pool3 = tf.nn.max_pool(relu5, [1,3,3,1], [1,2,2,1], padding = 'VALID', name = 'pool')			

		with tf.variable_scope('alexNet/fc6') as scope:
			pool3_reshape = tf.reshape(pool3, [-1,25088])
			Wfc6 = tf.get_variable('W',[25088,4096], trainable = True, initializer = tf.contrib.layers.xavier_initializer())
			bfc6 = tf.get_variable('b',[4096], trainable = True, initializer = tf.contrib.layers.xavier_initializer())
			fc6 = tf.nn.dropout(tf.nn.relu(tf.matmul(pool3_reshape, Wfc6) + bfc6), self._dropout_placeholder, name = 'fc6')
			self._parameters['fc6_W'] = Wfc6
			self._parameters['fc6_b'] = bfc6
			tf.add_to_collection('Reg', tf.reduce_sum(tf.square(Wfc6)))

		with tf.variable_scope('alexNet/fc7') as scope:
			Wfc7 = tf.get_variable('W',[4096,4096], trainable = True, initializer = tf.contrib.layers.xavier_initializer())
			bfc7 = tf.get_variable('b',[4096], trainable = True, initializer = tf.contrib.layers.xavier_initializer())
			fc7 = tf.nn.dropout(tf.nn.relu(tf.matmul(fc6, Wfc7) + bfc7), self._dropout_placeholder, name = 'fc7')
			self._parameters['fc7_W'] = Wfc7
			self._parameters['fc7_b'] = bfc7
			tf.add_to_collection('Reg', tf.reduce_sum(tf.square(Wfc7)))

		with tf.variable_scope('alexNet/fc8') as scope:
			Wfc8 = tf.get_variable('W',[4096,self._config.num_classes], trainable = True, initializer = tf.contrib.layers.xavier_initializer())
			bfc8 = tf.get_variable('b',[self._config.num_classes], trainable = True, initializer = tf.contrib.layers.xavier_initializer())
			self._score = tf.matmul(fc7, Wfc8) + bfc8
			self._parameters['fc8_W'] = Wfc8
			self._parameters['fc8_b'] = bfc8
			tf.add_to_collection('Reg', tf.reduce_sum(tf.square(Wfc8)))

		self._pred = tf.argmax(self._score, 1)

		self._loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = tf.one_hot(self._label_placeholder, self._config.num_classes), logits = self._score))
		for reg in tf.get_collection("Reg"):
			self._loss += 0.5 * self._config.l2 * reg

		optimizer = tf.train.AdamOptimizer(self._config.lr)
		self._train_op = optimizer.minimize(self._loss)


if __name__ == '__main__':
	alexNet = model_alexNet()
	with tf.Session() as sess:
		alexNet.load_parameters(sess,'../data/alexNet/alexNet_weights.npz')
		print(alexNet.predict_label(sess,['../data/img/test.jpg']))



