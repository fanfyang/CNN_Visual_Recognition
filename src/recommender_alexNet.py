from utils_model import *
import tensorflow as tf
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir, 'data/vgg16')))
from imagenet_classes import *

class recommender_alexNet(model):

	def __init__(self,config):


		def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
			'''From https://github.com/ethereon/caffe-tensorflow
			'''
			c_i = input.get_shape()[-1]
			assert c_i%group==0
			assert c_o%group==0
			convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
			if group==1:
				conv = convolve(input, kernel)
			else:
				input_groups =  tf.split(input, group, 3)   #tf.split(3, group, input)
				kernel_groups = tf.split(kernel, group, 3)  #tf.split(3, group, kernel) 
				output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
				conv = tf.concat(output_groups, 3)          #tf.concat(3, output_groups)
			return  tf.reshape(tf.nn.bias_add(conv, biases), [-1]+conv.get_shape().as_list()[1:])




		self._config_pic = Config_Pic(height = 227, width = 227, channels = 3)
		self._config = config

		self._input_placeholder = tf.placeholder(dtype = tf.float32, shape = (None, self._config_pic.height, self._config_pic.width, self._config_pic.channels))
		self._label_placeholder = tf.placeholder(dtype = tf.int32, shape = (None,))
		self._dropout_placeholder = tf.placeholder(dtype = tf.float32)
		self._is_training_placeholder = tf.placeholder(dtype = tf.bool)

		self._channel_mean = np.array([203.89836428,  191.68313589,  180.50212764])
		global class_names
		self._class_names = class_names
		self._parameters = dict()

		with tf.variable_scope('alexNet/conv1') as scope:
			Wconv1 = tf.get_variable('W', [11,11,self._config_pic.channels, 96], trainable =False)
			bconv1 = tf.get_variable('b', [96], trainable =False)
			conv1 = conv(self._input_placeholder, Wconv1, bconv1, 11, 11, 96, 4, 4, padding='VALID', group=1)
			relu1 = tf.nn.relu(conv1, name = 'relu1')
			self._parameters['conv1_W'] = Wconv1
			self._parameters['conv1_b'] = bconv1
			# tf.add_to_collection('Reg', tf.reduce_sum(tf.square(Wconv1)))

		# with tf.variable_scope('alexNet/lrn1') as scope:
		# 	radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
		# 	lrn1 = tf.nn.local_response_normalization(relu1, 
		# 											depth_radius=radius, 
		# 											alpha=alpha,
		# 											beta=beta,
		# 											bias=bias)			

		with tf.variable_scope('alexNet/pool1') as scope:
			pool1 = tf.nn.max_pool(relu1, [1,3,3,1], [1,2,2,1], padding = 'VALID', name = 'pool')

		with tf.variable_scope('alexNet/conv2') as scope:
			Wconv2 = tf.get_variable('W', [5,5,48,256], trainable =False)
			bconv2 = tf.get_variable('b', [256], trainable =False)
			conv2 = conv(pool1, Wconv2, bconv2, 5, 5, 256, 1, 1, padding='SAME', group=2)
			relu2 = tf.nn.relu(conv2, name = 'relu2')
			self._parameters['conv2_W'] = Wconv2
			self._parameters['conv2_b'] = bconv2
			# tf.add_to_collection('Reg', tf.reduce_sum(tf.square(Wconv2)))

		# with tf.variable_scope('alexNet/lrn2') as scope:
		# 	radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
		# 	lrn2 = tf.nn.local_response_normalization(relu2, 
		# 											depth_radius=radius, 
		# 											alpha=alpha,
		# 											beta=beta,
		# 											bias=bias)

		with tf.variable_scope('alexNet/pool2') as scope:
			pool2 = tf.nn.max_pool(relu2, [1,3,3,1], [1,2,2,1], padding = 'VALID', name = 'pool')

		with tf.variable_scope('alexNet/conv3') as scope:
			Wconv3 = tf.get_variable('W', [3,3,256,384], trainable =False)
			bconv3 = tf.get_variable('b', [384], trainable =False)
			conv3 = conv(pool2, Wconv3, bconv3, 3, 3, 348, 1, 1, padding='SAME', group=1)
			# conv3 = tf.nn.conv2d(bn1, Wconv3, [1,1,1,1], padding = 'SAME') + bconv3
			relu3 = tf.nn.relu(conv3, name = 'relu3')
			self._parameters['conv3_W'] = Wconv3
			self._parameters['conv3_b'] = bconv3
			# tf.add_to_collection('Reg', tf.reduce_sum(tf.square(Wconv3)))

		with tf.variable_scope('alexNet/conv4') as scope:
			Wconv4 = tf.get_variable('W',[3,3,192,384], trainable =False)
			bconv4 = tf.get_variable('b',[384], trainable =False)
			conv4 = conv(relu3, Wconv4, bconv4, 3, 3, 348, 1, 1, padding='SAME', group=2)
			# conv4 = tf.nn.conv2d(relu3, Wconv4, [1,1,1,1], padding = 'SAME') + bconv4
			relu4 = tf.nn.relu(conv4, name = 'relu4')
			self._parameters['conv4_W'] = Wconv4
			self._parameters['conv4_b'] = bconv4
			# tf.add_to_collection('Reg', tf.reduce_sum(tf.square(Wconv4)))

		with tf.variable_scope('alexNet/conv5') as scope:
			Wconv5 = tf.get_variable('W',[3,3,192,256], trainable =False)
			bconv5 = tf.get_variable('b',[256], trainable =False)
			conv5 = conv(relu4, Wconv5, bconv5, 3, 3, 256, 1, 1, padding='SAME', group=2)
			# conv5 = tf.nn.conv2d(relu4, Wconv5, [1,1,1,1], padding = 'SAME') + bconv5
			relu5 = tf.nn.relu(conv5, name = 'relu5')
			self._parameters['conv5_W'] = Wconv5
			self._parameters['conv5_b'] = bconv5
			# tf.add_to_collection('Reg', tf.reduce_sum(tf.square(Wconv5)))		

		with tf.variable_scope('alexNet/pool3') as scope:
			pool3 = tf.nn.max_pool(relu5, [1,3,3,1], [1,2,2,1], padding = 'VALID', name = 'pool')			

		with tf.variable_scope('alexNet/fc6') as scope:
			pool3_reshape = tf.reshape(pool3, [-1,9216])
			# Wfc6 = tf.get_variable('W',[9216,4096], trainable=True, initializer=tf.contrib.layers.xavier_initializer())
			Wfc6 = tf.get_variable('W',[9216,4096], trainable=False)
			bfc6 = tf.get_variable('b',[4096], trainable=False)
			fc6 = tf.nn.dropout(tf.nn.relu(tf.matmul(pool3_reshape, Wfc6) + bfc6), self._dropout_placeholder, name = 'fc6')
			self._parameters['fc6_W'] = Wfc6
			self._parameters['fc6_b'] = bfc6
			# tf.add_to_collection('Reg', tf.reduce_sum(tf.square(Wfc6)))

		with tf.variable_scope('alexNet/fc7') as scope:
			Wfc7 = tf.get_variable('W',[4096,4096], trainable =False)
			bfc7 = tf.get_variable('b',[4096], trainable =False)
			fc7 = tf.nn.dropout(tf.nn.relu(tf.matmul(fc6, Wfc7) + bfc7), self._dropout_placeholder, name = 'fc7')
			self._parameters['fc7_W'] = Wfc7
			self._parameters['fc7_b'] = bfc7
			# tf.add_to_collection('Reg', tf.reduce_sum(tf.square(Wfc7)))
			self._vector = fc7




		# with tf.variable_scope('alexNet/fc8') as scope:
		# 	Wfc8 = tf.get_variable('W',[4096,self._config.num_classes], trainable=False)
		# 	bfc8 = tf.get_variable('b',[self._config.num_classes], trainable=False)
		# 	self._score = tf.matmul(fc7, Wfc8) + bfc8
		# 	self._parameters['fc8_W'] = Wfc8
		# 	self._parameters['fc8_b'] = bfc8
		# 	# tf.add_to_collection('Reg', tf.reduce_sum(tf.square(Wfc8)))

def parse_args():
	parser = argparse.ArgumentParser(description="Run node2vec.")
	parser.add_argument('--weight', nargs='?', default='weight.pyz', help='Input weight path')
	return pasrser.parse_args()


if __name__ == '__main__':

	config = Config()
	r_alex = recommender_alexNet(config)

	args = parse_args()
	weight = args.weight

	sess = tf.Session()
	sess.run()
	r_alex.load_parameters(sess, '../../../FYang/model/Alex/' + weight, rand_init=['fc8'])


	x_base, y_base, z_base = fetch_data(file = True, resize = (227,227,3), cate_file = 'images_10000.txt', image_file = 'images_10000.txt')
	x_base -= r_alex._channel_mean

	feed_dict = {r_alex._input_placeholder:x_base, r_alex._label_placeholder:y_base, r_alex._dropout_placeholder:r_alex._config.dropout, r_alex._is_training_placeholder:False}
	vector_base = sess.run(r_alex._vector, feed_dict=feed_dict)
	print(vector_base.shape)

	# x_test, y_test, z_test = fetch_data(file = True, resize = (227,227,3), cate_file = 'images_small.txt', image_file = 'images_small.txt')
	# x_test -= r_alex._channel_mean	
	
