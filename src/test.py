from model_vgg16 import *
from model_svm import *
from model_alexNet import *

parser = argparse.ArgumentParser()

parser.add_argument('--v', help = 'version')
parser.add_argument('--m', help = 'model')
args = parser.parse_args()

version = args.v
model = args.m

x,y,z = fetch_data(file = True, cate_file = 'categories_10000.txt', image_file = 'images_10000.txt')
N = len(y)
N_val = N // 10 * 9
X_test = x[N_val:]
y_test = y[N_val:]

if model == 'vgg': 
	config = Config()
	vgg = model_vgg16_20(config)
	sess = tf.Session()
	vgg.load_parameters(sess,'../model/vgg/para_' + version + '.npz')
	print('test accu: %0.4f' % (1-vgg.error(sess,X_test-vgg._channel_mean,y_test)))
elif model == 'svm':
	config = Config()
	svm = svm(config)
	sess = tf.Session()
	svm.load_parameters(sess,'../model/svm/para_' + version + '.npz')
	print('test accu: %0.4f' % (1-svm.error(sess,X_test-vgg._channel_mean,y_test)))
else:
	config = Config()
	alex = model_alexNet(config)
	sess = tf.Session()
	alex.load_parameters(sess,'../model/alex/para_' + version + '.npz')
	print('test accu: %0.4f' % (1-alex.error(sess,X_test-vgg._channel_mean,y_test)))