from utils_recommendation import *
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
args = parser.parse_args()

version = args.v

para = parse_argument(args)

config = Config(**para)
vgg = model_vgg16_20(config)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
vgg.load_parameters(sess,'../model/vgg/para_' + str(version) + '.npz')

categories = []
files = []
labels = []
with open('../images.txt','r') as f:
	n = int(f.readline().rstrip('\n'))
	for _ in range(n):
		category = f.readline().rstrip('\n')
		categories.append(category)
	while True:
		temp = f.readline().rstrip('\n')
		if temp == None:
			break
		temp_split = temp.split('\t')
		try:
			assert(len(temp_split)==2)
		except:
			print(temp)
			print(temp_split)
		files.append(temp_split[0])
		labels.append(int(temp_split[1]))
labels = np.array(labels)

features = np.load('../model/vgg/feature_' + str(version) + '.npz')['feature']

recommendations = recommend(sess, vgg, features, labels, categories, files, ['../data/test.jpg'])
print(recommendations)