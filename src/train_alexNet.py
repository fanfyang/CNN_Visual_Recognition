from model_alexNet import *
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

# version = args.v
version = 'lr_' + args.lr + '_ne_' + args.ne + '_d_' + args.d + '_l2_' + args.l2

para = parse_argument(args)

config = Config(**para)
# config = Config(num_classes = 20, batch_size = 70, lr = 0.001, l2 = 0.0)
alex = model_alexNet(config)


def alex_train(model, sess, X_train, y_train, X_val, y_val, X_test, y_test, version):
	val_acc_current_best = 0.0
	userName = args.user
	resultName =  '../../../'+ userName + '/model/Alex/' + version + '.res'
	resultWriter = open(resultName, 'w')
	for i in range(model._config.num_epoch):
		resultWriter.write('Epoch %d / %d'%(i+1,model._config.num_epoch))
		model.run_epoch(sess, X_train, y_train)
		train_acc = 1-model.error(sess, X_train, y_train)
		val_acc = 1-model.error(sess, X_val, y_val)
		if X_test is not None:
			test_acc = 1-model.error(sess, X_test, y_test)
		resultWriter.write('train acc: %0.4f; val acc: %0.4f; test acc: %0.4f \n' % (train_acc, val_acc, test_acc))
		if val_acc > val_acc_current_best:
			val_acc_current_best = val_acc
			model.save_parameters(sess, '../../../'+ userName + '/model/Alex/',version)


# Example 1
x,y,z = fetch_data(file = True, resize = (227,227,3), cate_file = 'categories_10000.txt', image_file = 'images_10000.txt')
x -= alex._channel_mean
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
X_test = x[N_val:]
y_test = y[N_val:]
# print('===========', X_train.shape, X_val.shape, y_train.shape, y_val.shape, '===============')
sess = tf.Session()
sess.run(tf.global_variables_initializer())
alex.load_parameters_npy(sess,'../data/alex/bvlc_alexnet.npy',rand_init = ['fc8'])
alex_train(alex,sess,X_train,y_train,X_val,y_val,X_test,y_test,version)





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
# alex.load_parameters_npy(sess,'../data/alex/bvlc_alexnet.npy',rand_init = ['fc8'])
# alex.train_2(sess,X_train,y_train,X_val,y_val,z, resize = (227,227,3))
# alex.save_parameters(sess, '../model/alex_v1/')