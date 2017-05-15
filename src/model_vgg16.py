from utils_model import *
import tensorflow as tf
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir, 'data/vgg16')))
from imagenet_classes import *

class model_vgg16(model):
	def __init__(self):
		self._config_pic = Config_Pic()
		self._config = Config()

		self._input_placeholder = tf.placeholder(dtype = tf.float32, shape = (None, self._config_pic.height, self._config_pic.width, self._config_pic.channels))
		self._label_placeholder = tf.placeholder(dtype = tf.int32, shape = (None,))
		self._dropout_placeholder = tf.placeholder(dtype = tf.float32)

		self._channel_mean = np.array([ 203.89836428,  191.68313589,  180.50212764])
		global class_names
		self._class_names = class_names
		self._parameters = dict()

		with tf.variable_scope('vgg16/conv1_1') as scope:
			Wconv1_1 = tf.get_variable('W', [3,3,self._config_pic.channels,64], trainable = True)
			bconv1_1 = tf.get_variable('b', [64], trainable = True)
			conv1_1 = tf.nn.conv2d(self._input_placeholder, Wconv1_1, [1,1,1,1], padding = 'SAME') + bconv1_1
			relu1_1 = tf.nn.relu(conv1_1, name = 'relu1_1')
			self._parameters['conv1_1_W'] = Wconv1_1
			self._parameters['conv1_1_b'] = bconv1_1
			tf.add_to_collection('Reg', tf.reduce_sum(tf.square(Wconv1_1)))

		with tf.variable_scope('vgg16/conv1_2') as scope:
			Wconv1_2 = tf.get_variable('W', [3,3,64,64], trainable = True)
			bconv1_2 = tf.get_variable('b', [64], trainable = True)
			conv1_2 = tf.nn.conv2d(relu1_1, Wconv1_2, [1,1,1,1], padding = 'SAME') + bconv1_1
			relu1_2 = tf.nn.relu(conv1_2, name = 'relu1_2')
			self._parameters['conv1_2_W'] = Wconv1_2
			self._parameters['conv1_2_b'] = bconv1_2
			tf.add_to_collection('Reg', tf.reduce_sum(tf.square(Wconv1_2)))

		with tf.variable_scope('vgg16/pool1') as scope:
			pool1 = tf.nn.max_pool(relu1_2, [1,2,2,1], [1,2,2,1], padding = 'VALID', name = 'pool')

		with tf.variable_scope('vgg16/conv2_1') as scope:
			Wconv2_1 = tf.get_variable('W', [3,3,64,128], trainable = True)
			bconv2_1 = tf.get_variable('b', [128], trainable = True)
			conv2_1 = tf.nn.conv2d(pool1, Wconv2_1, [1,1,1,1], padding = 'SAME') + bconv2_1
			relu2_1 = tf.nn.relu(conv2_1, name = 'relu2_1')
			self._parameters['conv2_1_W'] = Wconv2_1
			self._parameters['conv2_1_b'] = bconv2_1
			tf.add_to_collection('Reg', tf.reduce_sum(tf.square(Wconv2_1)))

		with tf.variable_scope('vgg16/conv2_2') as scope:
			Wconv2_2 = tf.get_variable('W', [3,3,128,128], trainable = True)
			bconv2_2 = tf.get_variable('b', [128], trainable = True)
			conv2_2 = tf.nn.conv2d(relu2_1, Wconv2_2, [1,1,1,1], padding = 'SAME') + bconv2_2
			relu2_2 = tf.nn.relu(conv2_2, name = 'relu2_2')
			self._parameters['conv2_2_W'] = Wconv2_2
			self._parameters['conv2_2_b'] = bconv2_2
			tf.add_to_collection('Reg', tf.reduce_sum(tf.square(Wconv2_2)))

		with tf.variable_scope('vgg16/pool2') as scope:
			pool2 = tf.nn.max_pool(relu2_2, [1,2,2,1], [1,2,2,1], padding = 'VALID', name = 'pool')

		with tf.variable_scope('vgg16/conv3_1') as scope:
			Wconv3_1 = tf.get_variable('W',[3,3,128,256], trainable = True)
			bconv3_1 = tf.get_variable('b',[256], trainable = True)
			conv3_1 = tf.nn.conv2d(pool2, Wconv3_1, [1,1,1,1], padding = 'SAME') + bconv3_1
			relu3_1 = tf.nn.relu(conv3_1, name = 'relu3_1')
			self._parameters['conv3_1_W'] = Wconv3_1
			self._parameters['conv3_1_b'] = bconv3_1
			tf.add_to_collection('Reg', tf.reduce_sum(tf.square(Wconv3_1)))

		with tf.variable_scope('vgg16/conv3_2') as scope:
			Wconv3_2 = tf.get_variable('W',[3,3,256,256], trainable = True)
			bconv3_2 = tf.get_variable('b',[256], trainable = True)
			conv3_2 = tf.nn.conv2d(relu3_1, Wconv3_2, [1,1,1,1], padding = 'SAME') + bconv3_2
			relu3_2 = tf.nn.relu(conv3_2, name = 'relu3_2')
			self._parameters['conv3_2_W'] = Wconv3_2
			self._parameters['conv3_2_b'] = bconv3_2
			tf.add_to_collection('Reg', tf.reduce_sum(tf.square(Wconv3_2)))

		with tf.variable_scope('vgg16/conv3_3') as scope:
			Wconv3_3 = tf.get_variable('W',[3,3,256,256], trainable = True)
			bconv3_3 = tf.get_variable('b',[256], trainable = True)
			conv3_3 = tf.nn.conv2d(relu3_2, Wconv3_3, [1,1,1,1], padding = 'SAME') + bconv3_3
			relu3_3 = tf.nn.relu(conv3_3, name = 'relu3_3')
			self._parameters['conv3_3_W'] = Wconv3_3
			self._parameters['conv3_3_b'] = bconv3_3
			tf.add_to_collection('Reg', tf.reduce_sum(tf.square(Wconv3_3)))

		with tf.variable_scope('vgg16/pool3') as scope:
			pool3 = tf.nn.max_pool(relu3_3, [1,2,2,1], [1,2,2,1], padding = 'VALID', name = 'pool')

		with tf.variable_scope('vgg16/conv4_1') as scope:
			Wconv4_1 = tf.get_variable('W',[3,3,256,512], trainable = True)
			bconv4_1 = tf.get_variable('b',[512], trainable = True)
			conv4_1 = tf.nn.conv2d(pool3, Wconv4_1, [1,1,1,1], padding = 'SAME') + bconv4_1
			relu4_1 = tf.nn.relu(conv4_1, name = 'relu4_1')
			self._parameters['conv4_1_W'] = Wconv4_1
			self._parameters['conv4_1_b'] = bconv4_1
			tf.add_to_collection('Reg', tf.reduce_sum(tf.square(Wconv4_1)))

		with tf.variable_scope('vgg16/conv4_2') as scope:
			Wconv4_2 = tf.get_variable('W',[3,3,512,512], trainable = True)
			bconv4_2 = tf.get_variable('b',[512], trainable = True)
			conv4_2 = tf.nn.conv2d(relu4_1, Wconv4_2, [1,1,1,1], padding = 'SAME') + bconv4_2
			relu4_2 = tf.nn.relu(conv4_2, name = 'relu4_2')
			self._parameters['conv4_2_W'] = Wconv4_2
			self._parameters['conv4_2_b'] = bconv4_2
			tf.add_to_collection('Reg', tf.reduce_sum(tf.square(Wconv4_2)))

		with tf.variable_scope('vgg16/conv4_3') as scope:
			Wconv4_3 = tf.get_variable('W',[3,3,512,512], trainable = True)
			bconv4_3 = tf.get_variable('b',[512], trainable = True)
			conv4_3 = tf.nn.conv2d(relu4_2, Wconv4_3, [1,1,1,1], padding = 'SAME') + bconv4_3
			relu4_3 = tf.nn.relu(conv4_3, name = 'relu4_3')
			self._parameters['conv4_3_W'] = Wconv4_3
			self._parameters['conv4_3_b'] = bconv4_3
			tf.add_to_collection('Reg', tf.reduce_sum(tf.square(Wconv4_3)))

		with tf.variable_scope('vgg16/pool4') as scope:
			pool4 = tf.nn.max_pool(relu4_3, [1,2,2,1], [1,2,2,1], padding = 'VALID', name = 'pool')

		with tf.variable_scope('vgg16/conv5_1') as scope:
			Wconv5_1 = tf.get_variable('W',[3,3,512,512], trainable = True)
			bconv5_1 = tf.get_variable('b',[512], trainable = True)
			conv5_1 = tf.nn.conv2d(pool4, Wconv5_1, [1,1,1,1], padding = 'SAME') + bconv5_1
			relu5_1 = tf.nn.relu(conv5_1, name = 'relu5_1')
			self._parameters['conv5_1_W'] = Wconv5_1
			self._parameters['conv5_1_b'] = bconv5_1
			tf.add_to_collection('Reg', tf.reduce_sum(tf.square(Wconv5_1)))

		with tf.variable_scope('vgg16/conv5_2') as scope:
			Wconv5_2 = tf.get_variable('W',[3,3,512,512], trainable = True)
			bconv5_2 = tf.get_variable('b',[512], trainable = True)
			conv5_2 = tf.nn.conv2d(relu5_1, Wconv5_2, [1,1,1,1], padding = 'SAME') + bconv5_2
			relu5_2 = tf.nn.relu(conv5_2, name = 'relu5_2')
			self._parameters['conv5_2_W'] = Wconv5_2
			self._parameters['conv5_2_b'] = bconv5_2
			tf.add_to_collection('Reg', tf.reduce_sum(tf.square(Wconv5_2)))

		with tf.variable_scope('vgg16/conv5_3') as scope:
			Wconv5_3 = tf.get_variable('W',[3,3,512,512], trainable = True)
			bconv5_3 = tf.get_variable('b',[512], trainable = True)
			conv5_3 = tf.nn.conv2d(relu5_2, Wconv5_3, [1,1,1,1], padding = 'SAME') + bconv5_3
			relu5_3 = tf.nn.relu(conv5_3, name = 'relu5_3')
			self._parameters['conv5_3_W'] = Wconv5_3
			self._parameters['conv5_3_b'] = bconv5_3
			tf.add_to_collection('Reg', tf.reduce_sum(tf.square(Wconv5_3)))

		with tf.variable_scope('vgg16/pool5') as scope:
			pool5 = tf.nn.max_pool(relu5_3, [1,2,2,1], [1,2,2,1], padding = 'VALID', name = 'pool')

		with tf.variable_scope('vgg16/fc6') as scope:
			pool5_reshape = tf.reshape(pool5, [-1,25088])
			Wfc6 = tf.get_variable('W',[25088,4096], trainable = True)
			bfc6 = tf.get_variable('b',[4096], trainable = True)
			fc6 = tf.nn.dropout(tf.nn.relu(tf.matmul(pool5_reshape, Wfc6) + bfc6), self._dropout_placeholder, name = 'fc6')
			self._parameters['fc6_W'] = Wfc6
			self._parameters['fc6_b'] = bfc6
			tf.add_to_collection('Reg', tf.reduce_sum(tf.square(Wfc6)))

		with tf.variable_scope('vgg16/fc7') as scope:
			Wfc7 = tf.get_variable('W',[4096,4096], trainable = True)
			bfc7 = tf.get_variable('b',[4096], trainable = True)
			fc7 = tf.nn.dropout(tf.nn.relu(tf.matmul(fc6, Wfc7) + bfc7), self._dropout_placeholder, name = 'fc7')
			self._parameters['fc7_W'] = Wfc7
			self._parameters['fc7_b'] = bfc7
			tf.add_to_collection('Reg', tf.reduce_sum(tf.square(Wfc7)))

		with tf.variable_scope('vgg16/fc8') as scope:
			Wfc8 = tf.get_variable('W',[4096,self._config.num_classes], trainable = True)
			bfc8 = tf.get_variable('b',[self._config.num_classes], trainable = True)
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

class model_vgg16_20(model):
	def __init__(self, config):
		self._config_pic = Config_Pic()
		self._config = config

		self._input_placeholder = tf.placeholder(dtype = tf.float32, shape = (None, self._config_pic.height, self._config_pic.width, self._config_pic.channels))
		self._label_placeholder = tf.placeholder(dtype = tf.int32, shape = (None,))
		self._dropout_placeholder = tf.placeholder(dtype = tf.float32)

		self._channel_mean = np.array([ 203.89836428,  191.68313589,  180.50212764])
		# self._class_names = class_names
		self._parameters = dict()

		with tf.variable_scope('vgg16_20/conv1_1') as scope:
			Wconv1_1 = tf.get_variable('W', [3,3,self._config_pic.channels,64], trainable = False)
			bconv1_1 = tf.get_variable('b', [64], trainable = False)
			conv1_1 = tf.nn.conv2d(self._input_placeholder, Wconv1_1, [1,1,1,1], padding = 'SAME') + bconv1_1
			relu1_1 = tf.nn.relu(conv1_1, name = 'relu1_1')
			self._parameters['conv1_1_W'] = Wconv1_1
			self._parameters['conv1_1_b'] = bconv1_1
			# tf.add_to_collection('Reg', tf.reduce_sum(tf.square(Wconv1_1)))

		with tf.variable_scope('vgg16_20/conv1_2') as scope:
			Wconv1_2 = tf.get_variable('W', [3,3,64,64], trainable = False)
			bconv1_2 = tf.get_variable('b', [64], trainable = False)
			conv1_2 = tf.nn.conv2d(relu1_1, Wconv1_2, [1,1,1,1], padding = 'SAME') + bconv1_1
			relu1_2 = tf.nn.relu(conv1_2, name = 'relu1_2')
			self._parameters['conv1_2_W'] = Wconv1_2
			self._parameters['conv1_2_b'] = bconv1_2
			# tf.add_to_collection('Reg', tf.reduce_sum(tf.square(Wconv1_2)))

		with tf.variable_scope('vgg16_20/pool1') as scope:
			pool1 = tf.nn.max_pool(relu1_2, [1,2,2,1], [1,2,2,1], padding = 'VALID', name = 'pool')

		with tf.variable_scope('vgg16_20/conv2_1') as scope:
			Wconv2_1 = tf.get_variable('W', [3,3,64,128], trainable = False)
			bconv2_1 = tf.get_variable('b', [128], trainable = False)
			conv2_1 = tf.nn.conv2d(pool1, Wconv2_1, [1,1,1,1], padding = 'SAME') + bconv2_1
			relu2_1 = tf.nn.relu(conv2_1, name = 'relu2_1')
			self._parameters['conv2_1_W'] = Wconv2_1
			self._parameters['conv2_1_b'] = bconv2_1
			# tf.add_to_collection('Reg', tf.reduce_sum(tf.square(Wconv2_1)))

		with tf.variable_scope('vgg16_20/conv2_2') as scope:
			Wconv2_2 = tf.get_variable('W', [3,3,128,128], trainable = False)
			bconv2_2 = tf.get_variable('b', [128], trainable = False)
			conv2_2 = tf.nn.conv2d(relu2_1, Wconv2_2, [1,1,1,1], padding = 'SAME') + bconv2_2
			relu2_2 = tf.nn.relu(conv2_2, name = 'relu2_2')
			self._parameters['conv2_2_W'] = Wconv2_2
			self._parameters['conv2_2_b'] = bconv2_2
			# tf.add_to_collection('Reg', tf.reduce_sum(tf.square(Wconv2_2)))

		with tf.variable_scope('vgg16_20/pool2') as scope:
			pool2 = tf.nn.max_pool(relu2_2, [1,2,2,1], [1,2,2,1], padding = 'VALID', name = 'pool')

		with tf.variable_scope('vgg16_20/conv3_1') as scope:
			Wconv3_1 = tf.get_variable('W',[3,3,128,256], trainable = False)
			bconv3_1 = tf.get_variable('b',[256], trainable = False)
			conv3_1 = tf.nn.conv2d(pool2, Wconv3_1, [1,1,1,1], padding = 'SAME') + bconv3_1
			relu3_1 = tf.nn.relu(conv3_1, name = 'relu3_1')
			self._parameters['conv3_1_W'] = Wconv3_1
			self._parameters['conv3_1_b'] = bconv3_1
			# tf.add_to_collection('Reg', tf.reduce_sum(tf.square(Wconv3_1)))

		with tf.variable_scope('vgg16_20/conv3_2') as scope:
			Wconv3_2 = tf.get_variable('W',[3,3,256,256], trainable = False)
			bconv3_2 = tf.get_variable('b',[256], trainable = False)
			conv3_2 = tf.nn.conv2d(relu3_1, Wconv3_2, [1,1,1,1], padding = 'SAME') + bconv3_2
			relu3_2 = tf.nn.relu(conv3_2, name = 'relu3_2')
			self._parameters['conv3_2_W'] = Wconv3_2
			self._parameters['conv3_2_b'] = bconv3_2
			# tf.add_to_collection('Reg', tf.reduce_sum(tf.square(Wconv3_2)))

		with tf.variable_scope('vgg16_20/conv3_3') as scope:
			Wconv3_3 = tf.get_variable('W',[3,3,256,256], trainable = False)
			bconv3_3 = tf.get_variable('b',[256], trainable = False)
			conv3_3 = tf.nn.conv2d(relu3_2, Wconv3_3, [1,1,1,1], padding = 'SAME') + bconv3_3
			relu3_3 = tf.nn.relu(conv3_3, name = 'relu3_3')
			self._parameters['conv3_3_W'] = Wconv3_3
			self._parameters['conv3_3_b'] = bconv3_3
			# tf.add_to_collection('Reg', tf.reduce_sum(tf.square(Wconv3_3)))

		with tf.variable_scope('vgg16_20/pool3') as scope:
			pool3 = tf.nn.max_pool(relu3_3, [1,2,2,1], [1,2,2,1], padding = 'VALID', name = 'pool')

		with tf.variable_scope('vgg16_20/conv4_1') as scope:
			Wconv4_1 = tf.get_variable('W',[3,3,256,512], trainable = False)
			bconv4_1 = tf.get_variable('b',[512], trainable = False)
			conv4_1 = tf.nn.conv2d(pool3, Wconv4_1, [1,1,1,1], padding = 'SAME') + bconv4_1
			relu4_1 = tf.nn.relu(conv4_1, name = 'relu4_1')
			self._parameters['conv4_1_W'] = Wconv4_1
			self._parameters['conv4_1_b'] = bconv4_1
			# tf.add_to_collection('Reg', tf.reduce_sum(tf.square(Wconv4_1)))

		with tf.variable_scope('vgg16_20/conv4_2') as scope:
			Wconv4_2 = tf.get_variable('W',[3,3,512,512], trainable = False)
			bconv4_2 = tf.get_variable('b',[512], trainable = False)
			conv4_2 = tf.nn.conv2d(relu4_1, Wconv4_2, [1,1,1,1], padding = 'SAME') + bconv4_2
			relu4_2 = tf.nn.relu(conv4_2, name = 'relu4_2')
			self._parameters['conv4_2_W'] = Wconv4_2
			self._parameters['conv4_2_b'] = bconv4_2
			# tf.add_to_collection('Reg', tf.reduce_sum(tf.square(Wconv4_2)))

		with tf.variable_scope('vgg16_20/conv4_3') as scope:
			Wconv4_3 = tf.get_variable('W',[3,3,512,512], trainable = False)
			bconv4_3 = tf.get_variable('b',[512], trainable = False)
			conv4_3 = tf.nn.conv2d(relu4_2, Wconv4_3, [1,1,1,1], padding = 'SAME') + bconv4_3
			relu4_3 = tf.nn.relu(conv4_3, name = 'relu4_3')
			self._parameters['conv4_3_W'] = Wconv4_3
			self._parameters['conv4_3_b'] = bconv4_3
			# tf.add_to_collection('Reg', tf.reduce_sum(tf.square(Wconv4_3)))

		with tf.variable_scope('vgg16_20/pool4') as scope:
			pool4 = tf.nn.max_pool(relu4_3, [1,2,2,1], [1,2,2,1], padding = 'VALID', name = 'pool')

		with tf.variable_scope('vgg16_20/conv5_1') as scope:
			Wconv5_1 = tf.get_variable('W',[3,3,512,512], trainable = False)
			bconv5_1 = tf.get_variable('b',[512], trainable = False)
			conv5_1 = tf.nn.conv2d(pool4, Wconv5_1, [1,1,1,1], padding = 'SAME') + bconv5_1
			relu5_1 = tf.nn.relu(conv5_1, name = 'relu5_1')
			self._parameters['conv5_1_W'] = Wconv5_1
			self._parameters['conv5_1_b'] = bconv5_1
			# tf.add_to_collection('Reg', tf.reduce_sum(tf.square(Wconv5_1)))

		with tf.variable_scope('vgg16_20/conv5_2') as scope:
			Wconv5_2 = tf.get_variable('W',[3,3,512,512], trainable = False)
			bconv5_2 = tf.get_variable('b',[512], trainable = False)
			conv5_2 = tf.nn.conv2d(relu5_1, Wconv5_2, [1,1,1,1], padding = 'SAME') + bconv5_2
			relu5_2 = tf.nn.relu(conv5_2, name = 'relu5_2')
			self._parameters['conv5_2_W'] = Wconv5_2
			self._parameters['conv5_2_b'] = bconv5_2
			# tf.add_to_collection('Reg', tf.reduce_sum(tf.square(Wconv5_2)))

		with tf.variable_scope('vgg16_20/conv5_3') as scope:
			Wconv5_3 = tf.get_variable('W',[3,3,512,512], trainable = False)
			bconv5_3 = tf.get_variable('b',[512], trainable = False)
			conv5_3 = tf.nn.conv2d(relu5_2, Wconv5_3, [1,1,1,1], padding = 'SAME') + bconv5_3
			relu5_3 = tf.nn.relu(conv5_3, name = 'relu5_3')
			self._parameters['conv5_3_W'] = Wconv5_3
			self._parameters['conv5_3_b'] = bconv5_3
			# tf.add_to_collection('Reg', tf.reduce_sum(tf.square(Wconv5_3)))

		with tf.variable_scope('vgg16_20/pool5') as scope:
			pool5 = tf.nn.max_pool(relu5_3, [1,2,2,1], [1,2,2,1], padding = 'VALID', name = 'pool')

		with tf.variable_scope('vgg16_20/fc6') as scope:
			pool5_reshape = tf.reshape(pool5, [-1,25088])
			Wfc6 = tf.get_variable('W',[25088,4096], trainable = True)
			bfc6 = tf.get_variable('b',[4096], trainable = True)
			fc6 = tf.nn.dropout(tf.nn.relu(tf.matmul(pool5_reshape, Wfc6) + bfc6), self._dropout_placeholder, name = 'fc6')
			self._parameters['fc6_W'] = Wfc6
			self._parameters['fc6_b'] = bfc6
			tf.add_to_collection('Reg', tf.reduce_sum(tf.square(Wfc6)))

		with tf.variable_scope('vgg16_20/fc7') as scope:
			Wfc7 = tf.get_variable('W',[4096,4096], trainable = True)
			bfc7 = tf.get_variable('b',[4096], trainable = True)
			fc7 = tf.nn.dropout(tf.nn.relu(tf.matmul(fc6, Wfc7) + bfc7), self._dropout_placeholder, name = 'fc7')
			self._parameters['fc7_W'] = Wfc7
			self._parameters['fc7_b'] = bfc7
			tf.add_to_collection('Reg', tf.reduce_sum(tf.square(Wfc7)))

		with tf.variable_scope('vgg16_20/fc8') as scope:
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

		self._global_step = tf.Variable(0, trainable = False)
		lr = tf.train.exponential_decay(self._config.lr, decay_rate = self._config.decay_rate, global_step = self._global_step, decay_steps = self._config.decay_steps)

		optimizer = tf.train.AdamOptimizer(lr)
		self._train_op = optimizer.minimize(self._loss)

if __name__ == '__main__':
	config = Config(num_classes = 20, batch_size = 10, lr = 0.001, l2 = 0.0)
	vgg16 = model_vgg16_20(config)

	x,y,z = fetch_data(file = True)
	X_train = x[:700]
	X_val = x[700:]
	y_train = y[:700]
	y_val = y[700:]

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	vgg.load_parameters(sess,'../data/vgg16/vgg16_weights.npz',rand_init = ['fc8_W', 'fc8_b'])
	vgg.train(sess,X_train,y_train,X_val,y_val)
