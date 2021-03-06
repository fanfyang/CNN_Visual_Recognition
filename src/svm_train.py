from model_svm import *

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
args = parser.parse_args()

version = args.v

para = parse_argument(args)

config = Config(**para)
# config = Config(num_classes = 20, batch_size = 70, lr = 0.001, l2 = 0.0)
svm = svm(config)

# Example 1
x,y,z = fetch_data(file = True, cate_file = 'categories_10000.txt', image_file = 'images_10000.txt')
# x -= vgg._channel_mean
# X_train = x[:700]
# X_val = x[700:]
# y_train = y[:700]
# y_val = y[700:]

N = len(y)
N_train = N // 10 * 7
N_val = N // 10 * 9
X_train = x[:N_train]
X_val = x[N_train:N_val]
y_train = y[:N_train]
y_val = y[N_train:N_val]

sess = tf.Session()
sess.run(tf.global_variables_initializer())
svm.train(sess,X_train,y_train,X_val,y_val,version, model = 'svm')

# Example 2
#x,y,z = fetch_data_2( cate_file = 'categories.txt', image_file = 'images.txt')
#N = len(y)
#N_train = N // 10 * 7
#N_val = N // 10 * 9
#X_train = x[:N_train]
#X_val = x[N_train:N_val]
#y_train = y[:N_train]
#y_val = y[N_train:N_val]

#sess = tf.Session()
#sess.run(tf.global_variables_initializer())
# svm.load_parameters(sess,'../data/vgg16/vgg16_weights.npz',rand_init = ['fc8_W', 'fc8_b'])
#svm.train_2(sess,X_train,y_train,X_val,y_val,z)