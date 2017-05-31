from model_alexNet import *


def parse_args():
	parser = argparse.ArgumentParser(description="Run node2vec.")
	parser.add_argument('--weight', nargs='?', default='weight.pyz', help='Input weight path')
	return parser.parse_args()

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
	features = []

	for i in range(num_batches):
		batch_idx = idx[i*self._config.batch_size:(i+1)*self._config.batch_size]
		X_batch = X[batch_idx]
		y_batch = y[batch_idx]
		feed_dict = {self._input_placeholder:X_batch, self._label_placeholder:y_batch, self._dropout_placeholder:self._config.dropout, self._is_training_placeholder:False}
		vector = sess.run(self._vector, feed_dict)
		# print('loss: ' + str(loss))
		# total_loss.append(loss)
		features.append(vector)
		# accu.append(1.*np.sum(pred==y_batch)/self._config.batch_size)
		if (i+1)//batch_per_print*batch_per_print == i+1:
			num_eq = (i+1)//batch_per_eq
			sys.stdout.write('\r '+str(i+1)+' / '+str(num_batches)+' [' + '='*num_eq + ' '*(len_eq - num_eq) + '] - %0.2fs  '%(float(time.time()-start)))
			sys.stdout.flush()
		if shuffle:
			np.random.shuffle(idx)
	
	sys.stdout.write('\r '+str(num_batches)+' / '+str(num_batches)+' [' + '='*len_eq + '] - %0.2fs  \n'%(float(time.time()-start)))
	sys.stdout.flush()
	return features




if __name__ == '__main__':

	config = Config(num_classes=20)
	alex = model_alexNet(config)

	args = parse_args()
	weight = args.weight

	sess = tf.Session()

	alex.load_parameters(sess, '../../../FYang/model/Alex/' + weight)


	x_base, y_base, z_base = fetch_data(file = True, resize = (227,227,3), cate_file = 'categories_10000.txt', image_file = 'images_10000.txt')
	x_base -= alex._channel_mean
	# feed_dict = {alex._input_placeholder:x_base, alex._label_placeholder:y_base, alex._dropout_placeholder:alex._config.dropout, alex._is_training_placeholder:False}
	# vector_base = sess.run(alex._vector, feed_dict=feed_dict)
	features = run_epoch(self, sess, x_base, y_base, shuffle = False)
	
	# print(vector_base.shape)

	# x_test, y_test, z_test = fetch_data(file = True, resize = (227,227,3), cate_file = 'images_small.txt', image_file = 'images_small.txt')
	# x_test -= r_alex._channel_mean	




	
