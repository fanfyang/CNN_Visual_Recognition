from model_vgg16 import *

parser = argparse.ArgumentParser()
parser.add_argument('--lr', help = 'learning rate')
parser.add_argument('--dr', help = 'decay_rate')
parser.add_argument('--ds', help = 'decay steps')
parser.add_argument('--l2', help = 'regularization')
parser.add_argument('--bs', help = 'batch size')
parser.add_argument('--ne', help = 'number of epoches')
parser.add_argument('--d', help = 'droupout')
parser.add_argument('--nc', help = 'number of classes')
parser.add_argument('--bn', help = 'batch normalization')
parser.add_argument('--v', help = 'version')
parser.add_argument('--i', help = 'initialization')
args = parser.parse_args()

version = args.v

para = parse_argument(args)

# Example 1
# x,y,class_names = fetch_data(file = True, cate_file = 'categories_small.txt', image_file = 'images_small.txt')
x,y,class_names = fetch_data(file = True, cate_file = 'categories_10000.txt', image_file = 'images_10000.txt')
N = len(y)
N_train = N // 10 * 7
N_val = N // 10 * 9
X_train = x[:N_train]
y_train = y[:N_train]
X_val = x[N_train:N_val]
y_val = y[N_train:N_val]
X_test = x[N_val:]
y_test = y[N_val:]

config = Config(**para)
vgg = model_vgg16_20(config)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
if args.i is None:
	vgg.load_parameters(sess,'../data/vgg16/vgg16_weights.npz',rand_init = ['fc8_W', 'fc8_b'])
else:
	vgg.load_parameters(sess,'../model/vgg/para_' + args.i + '.npz')
vgg.train(sess,X_train - vgg._channel_mean,y_train,X_val - vgg._channel_mean,y_val,X_test - vgg._channel_mean,y_test,version, model = 'vgg')

# # Example 2
# x,y,z = fetch_data_2(cate_file = 'categories.txt', image_file = 'images.txt')
# N = len(y)
# N_train = N // 10 * 7
# N_val = N // 10 * 9
# X_train = x[:N_train]
# X_val = x[N_train:N_val]
# y_train = y[:N_train]
# y_val = y[N_train:N_val]

# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# vgg.load_parameters(sess,'../data/vgg16/vgg16_weights.npz',rand_init = ['fc8_W', 'fc8_b'])
# vgg.train_2(sess,X_train,y_train,X_val,y_val,z)
