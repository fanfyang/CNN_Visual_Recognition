from model_vgg16_dense import *

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
vgg = model_vgg16_20(config)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
vgg.load_parameters(sess,'../model/vgg/para_' + str(version) + '.npz')

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

vgg.extract_feature(sess, x, is_training = False, version = version)
print(files[0])
print(categories[labels[0]])
print(files[-1])
print(categories[labels[-1]])
