from model_alexNet import *


def parse_args():
	parser = argparse.ArgumentParser(description="Run node2vec.")
	parser.add_argument('--weight', nargs='?', default='weight.pyz', help='Input weight path')
	return parser.parse_args()


if __name__ == '__main__':

	config = Config(num_classes = 20)
	alex = model_alexNet(config)

	args = parse_args()
	weight = args.weight

	sess = tf.Session()

	alex.load_parameters(sess, '../../../FYang/model/Alex/' + weight)


	x_base, y_base, z_base = fetch_data(file = True, resize = (227,227,3), cate_file = 'categories_10000.txt', image_file = 'images_10000.txt')
	x_base -= alex._channel_mean
	feed_dict = {alex._input_placeholder:x_base, alex._label_placeholder:y_base, alex._dropout_placeholder:alex._config.dropout, alex._is_training_placeholder:False}
	vector_base = sess.run(alex._vector, feed_dict=feed_dict)
	print(vector_base.shape)

	# x_test, y_test, z_test = fetch_data(file = True, resize = (227,227,3), cate_file = 'images_small.txt', image_file = 'images_small.txt')
	# x_test -= r_alex._channel_mean	
	
