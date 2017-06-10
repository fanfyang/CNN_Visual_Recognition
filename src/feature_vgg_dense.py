from model_vgg16_dense import *
from time import time

parser = argparse.ArgumentParser()
parser.add_argument('--lr', help = 'learning rate', nargs='?', default=0.0005)
parser.add_argument('--dr', help = 'decay_rate')
parser.add_argument('--ds', help = 'decay steps')
parser.add_argument('--l2', help = 'regularization', nargs='?', default=0.01)
parser.add_argument('--bs', help = 'batch size', nargs='?', default=16)
parser.add_argument('--ne', help = 'number of epoches', nargs='?', default=2)
parser.add_argument('--d', help = 'droupout', nargs='?', default=0.5)
parser.add_argument('--nc', help = 'number of classes', nargs='?', default=20)
parser.add_argument('--bn', help = 'batch normalization', nargs='?', default=False)
parser.add_argument('--v', help = 'version', nargs='?', default=str(time()))
parser.add_argument('--user', help= 'user name', nargs='?', default='FYang')
args = parser.parse_args()

version = 'lr_' + str(args.lr) + '_ne_' + str(args.ne) + '_d_' + str(args.d) + '_l2_' + str(args.l2)
userName = args.user

para = parse_argument(args)

config = Config(**para)
vgg_dense = model_vgg16_20_dense(config)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
param_name = '../../../'+ userName + '/model/dense/para_' + version + '.npz'
vgg_dense.load_parameters(sess,param_name)

images, labels, categories, files = fetch_data(file = True, cate_file = 'categories_10000.txt', image_file = 'images_10000.txt', filenames = True, shuffle = False)
# images, labels, categories = fetch_data(file = True, cate_file = 'categories_10000.txt', image_file = 'images_10000.txt', filenames = False, shuffle = True)
x = images - vgg._channel_mean

# N = len(labels)
# N_train = N // 10 * 7
# N_val = N // 10 * 9
# X_train = x[:N_train]
# X_val = x[N_train:N_val]
# X_test = x[N_val:]
# y_train = labels[:N_train]
# y_val = labels[N_train:N_val]
# y_test = labels[N_val:]

# print(vgg.error(sess, X_train, y_train))
# print(vgg.error(sess, X_val, y_val))
# print(vgg.error(sess, X_test, y_test))

def extract_feature(model, sess, X, userName, is_training = False, version = None):
	num_batches = X.shape[0] // model._config.batch_size
	features = list()
	for i in range(num_batches):
		feed_dict = {model._input_placeholder: X[i*model._config.batch_size:(i+1)*model._config.batch_size], model._dropout_placeholder:1.0, model._is_training_placeholder:is_training}
		feature = sess.run(model._feature, feed_dict)
		features.append(feature)
	feed_dict = {model._input_placeholder: X[num_batches*model._config.batch_size:], model._dropout_placeholder:1.0, model._is_training_placeholder:is_training}
	feature = sess.run(model._feature, feed_dict)
	features.append(feature)
	features = np.concatenate(features, axis = 0)
	if version is not None:
		np.savez('../../../'+ userName + '/model/dense/feature_' + version + '.npz', feature = features)
	return features



extract_feature(vgg_dense, sess, x, userName, is_training = False, version = version)
print(files[0])
print(categories[labels[0]])
print(files[-1])
print(categories[labels[-1]])
