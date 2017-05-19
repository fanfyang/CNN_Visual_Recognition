from model_vgg16 import *
from model_svm import *
from model_alexNet import *

parser = argparse.ArgumentParser()

parser.add_argument('--v', help = 'version')
parser.add_argument('--m', help = 'model')
parser.add_argument('--nc', help = 'number of classes')
parser.add_argument('--bs', help = 'batch size')
args = parser.parse_args()

version = args.v
model = args.m
num_classes = int(args.nc)
batch_size = int(args.bs)

if model == 'vgg': 
	x,y,z = fetch_data(file = True, cate_file = 'categories_10000.txt', image_file = 'images_10000.txt')
	N = len(y)
	N_val = N // 10 * 9
	X_test = x[N_val:]
	y_test = y[N_val:]

	config = Config(num_classes = num_classes, batch_size = batch_size)
	vgg = model_vgg16_20(config)
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	vgg.load_parameters(sess,'../model/vgg/para_' + version + '.npz')
	print('test accu: %0.4f' % (1-vgg.error(sess,X_test-vgg._channel_mean,y_test)))
elif model == 'svm':
	x,y,z = fetch_data(file = True, cate_file = 'categories_10000.txt', image_file = 'images_10000.txt')
	N = len(y)
	N_val = N // 10 * 9
	X_test = x[N_val:]
	y_test = y[N_val:]
	
	config = Config(num_classes = num_classes, batch_size = batch_size)
	svm = svm(config)
	sess = tf.Session()
	svm.load_parameters(sess,'../model/svm/para_' + version + '.npz')
	print('test accu: %0.4f' % (1-svm.error(sess,X_test-svm._channel_mean,y_test)))
else:
	x,y,z = fetch_data(file = True, resize = (227,227,3), cate_file = 'categories_10000.txt', image_file = 'images_10000.txt')
	N = len(y)
	N_val = N // 10 * 9
	X_test = x[N_val:]
	y_test = y[N_val:]
	config = Config(num_classes = num_classes, batch_size = batch_size)
	alex = model_alexNet(config)
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	alex.load_parameters(sess,'../model/alex/para_' + version + '.npz')
	print('test accu: %0.4f' % (1-alex.error(sess,X_test-alex._channel_mean,y_test)))