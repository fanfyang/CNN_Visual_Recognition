from model_similarity import *

parser = argparse.ArgumentParser()
parser.add_argument('--lr', help = 'learning rate')
parser.add_argument('--dr', help = 'decay_rate')
parser.add_argument('--ds', help = 'decay steps')
parser.add_argument('--l2', help = 'regularization')
parser.add_argument('--bs', help = 'batch size')
parser.add_argument('--ne', help = 'number of epoches')
parser.add_argument('--d', help = 'droupout')
parser.add_argument('--v', help = 'version')
args = parser.parse_args()

version = args.v

def parse_argument_NN(args):
	para = {}
	para['lr'] = float(args.lr) if args.lr != None else 0.001
	para['decay_rate'] = float(args.dr) if args.dr != None else 0.9
	para['decay_steps'] = int(args.ds) if args.ds != None else 700
	para['l2'] = float(args.l2) if args.l2 != None else 0.0005
	para['batch_size'] = int(args.bs) if args.bs != None else 128
	para['num_epoch'] = int(args.ne) if args.ne != None else 20
	para['dropout'] = float(args.d) if args.d != None else 0.5
	return para

def prepare_training_data(path_feature, path_similarity = '../data/similarity/'):
	categories = []
	files = []
	labels = []
	with open('../data/images.txt','r') as f:
		n = int(f.readline().rstrip('\n'))
		for _ in range(n):
			category = f.readline().rstrip('\n')
			categories.append(category)
		while True:
			temp = f.readline().rstrip('\n')
			if temp == '':
				break
			temp_split = temp.split('\t')
			files.append(temp_split[0])
			labels.append(int(temp_split[1]))
	labels = np.array(labels)

	features = np.load(path_feature)['feature']

	scores = dict()
	score_ignore = dict()
	score_ignore['Cell Phones & Accessories'] = 2
	score_ignore['Watches'] = 474
	score_ignore['Home Improvement'] = 495
	for cate in categories:
		scores[cate] = np.load(path_similarity + 'similarity_'+ cate +'.npz')['similarity']

	x = []
	y = []
	skip = 0
	for i in range(len(categories)):
		num_skip = 500 * i - skip
		category = categories[i]
		for k1 in range(499):
			if category in score_ignore:
				if k1 == score_ignore[category]:
					continue
				elif k1 < score_ignore[category]:
					feature_k1 = features[num_skip + k1]
				else:
					feature_k1 = features[num_skip + k1 - 1]
			else:
				feature_k1 = features[num_skip + k1]
			for k2 in range(k1+1,500):
				if category in score_ignore:
					if k2 == score_ignore[category]:
						continue
					elif k2 < score_ignore[category]:
						feature_k2 = features[num_skip + k2]
					else:
						feature_k2 = features[num_skip + k2 - 1]
				else:
					feature_k2 = features[num_skip + k2]
				x.append(np.concatenate((feature_k1, feature_k2)))
				y.append(scores[category][k1,k2])
		if category in score_ignore:
			skip += 1
	return (np.array(x),np.array(y))

path_feature = '../model/vgg/feature_3.npz'

x, y = prepare_training_data(path_feature)
print(x.shape)
print(y.shape)

# N_train = N // 10 * 7
# N_val = N // 10 * 9
# X_train = x[:N_train]
# y_train = y[:N_train]
# X_val = x[N_train:N_val]
# y_val = y[N_train:N_val]
# X_test = x[N_val:]
# y_test = y[N_val:]

# para = parse_argument_NN(args)
# config = Config_NN(**para)
# model_sim = model_nn(config)
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# model_sim.train(sess, X_train, y_train, X_val, y_val, X_test, y_test, version = 'v')
