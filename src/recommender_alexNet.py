from model_alexNet import *
import operator

def parse_args():
	parser = argparse.ArgumentParser(description="Run alex net recommender.")
	parser.add_argument('--weight', nargs='?', default='weight.pyz', help='Input weight path')
	parser.add_argument('--base', nargs='?', default='10000')
	parser.add_argument('--test', nargs='?', default='small')
	return parser.parse_args()

def run_epoch(model, sess, X, y, shuffle = True, batch_per_print = 2):
	start = time.time()
	num_batches = X.shape[0] // model._config.batch_size
	idx = np.arange(X.shape[0])
	if shuffle:
		np.random.shuffle(idx)

	len_eq = 20
	batch_per_eq = (num_batches+len_eq-1)//len_eq

	total_loss = []
	accu = []
	features = []

	for i in range(num_batches):
		batch_idx = idx[i*model._config.batch_size:(i+1)*model._config.batch_size]
		X_batch = X[batch_idx]
		y_batch = y[batch_idx]
		feed_dict = {model._input_placeholder:X_batch, model._label_placeholder:y_batch, model._dropout_placeholder:model._config.dropout, model._is_training_placeholder:False}
		vector = sess.run(model._vector, feed_dict)
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
	features = np.array(features)
	return features.reshape(-1, features.shape[2])

def cosine(x1, x2):
	return np.dot(x1, x2)/(np.linalg.norm(x1)*np.linalg.norm(x2))


def top_similar_item(x_base, y_base, x_test, y_test, k):
	similarities = {}
	for i in range(x_test.shape[0]):
		s = {}
		for j in range(x_base.shape[0]):
			s[y_base[j]] = cosine(x_base[j], x_test[i])
		similarities[y_test[i]] = sorted(s.items(), key=operator.itemgetter(1), reverse=True)[0:k]
	return similarities






if __name__ == '__main__':

	config = Config(num_classes=20)
	alex = model_alexNet(config)

	args = parse_args()
	weight = args.weight
	cate_base = 'categories_' + args.base + '.txt'
	image_base = 'images_' + args.base + '.txt'
	cate_test = 'categories_' + args.test + '.txt'
	image_test = 'images_' + args.test + '.txt'


	sess = tf.Session()

	alex.load_parameters(sess, '../../../FYang/model/Alex/' + weight)


	x_base, y_base, z_base = fetch_data(file = True, resize = (227,227,3), cate_file = cate_base, image_file = image_base)
	x_base -= alex._channel_mean
	# feed_dict = {alex._input_placeholder:x_base, alex._label_placeholder:y_base, alex._dropout_placeholder:alex._config.dropout, alex._is_training_placeholder:False}
	# vector_base = sess.run(alex._vector, feed_dict=feed_dict)
	features_base = run_epoch(alex, sess, x_base, y_base, shuffle = False)

	# print(features_base.shape)

	x_test, y_test, z_test = fetch_data(file = True, resize = (227,227,3), cate_file = cate_test, image_file = image_test)
	features_test = run_epoch(alex, sess, x_test, y_test, shuffle = False)

	print(features_base.shape, y_base.shape, features_test.shape, y_test.shape)
	similarities = top_similar_item(features_base, y_base, features_test, y_test, 2)
	print(similarities)



	# x_test, y_test, z_test = fetch_data(file = True, resize = (227,227,3), cate_file = 'images_small.txt', image_file = 'images_small.txt')
	# x_test -= r_alex._channel_mean	




	
