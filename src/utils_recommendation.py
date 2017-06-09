from utils_model import *
from utils_database import *

ALLOWED_EXTENSIONS = set(['jpg'])
UPLOAD_FOLDER = '../web/img/temp'

def subsample(features, labels, files, label):
	mask = labels == label
	sample_features = features[mask]
	sample_files = [files[i] for i in range(len(mask)) if mask[i] == True]
	return (sample_features, sample_files)

def feature_normalization(features):
	return features / np.linalg.norm(features, axis = 1)[:,np.newaxis]

def top_similar_cosine(features, feature, num):
	features_normalization = feature_normalization(features)
	dis = np.dot(features_normalization, feature)
	temp = [(dis[i], i) for i in range(len(dis))]
	temp_sort = sorted(temp, key = lambda t:-t[0])
	return [idx for _, idx in temp_sort[:num]]

def top_similar_L2(features, feature, num):
	dis = np.linalg.norm(features - feature, axis = 1)
	temp = [(dis[i], i) for i in range(len(dis))]
	temp_sort = sorted(temp, key = lambda t:t[0])
	return [idx for _, idx in temp_sort[:num]]

def recommend(sess, model, features, labels, categories, files, image_path, num = 9, dis = 'cosine'):
	label_pred, feature_pred = model.predict(sess, image_path)
	recommendation = []
	for i in range(len(label_pred)):
		label = label_pred[i]
		feature = feature_pred[i]
		sample_features, sample_files = subsample(features, labels, files, label)
		if dis == 'cosine':
			idx = top_similar_cosine(sample_features, feature, num)
		else:
			idx = top_similar_L2(sample_features, feature, num)
		temp = (categories[label], [sample_files[i] for i in idx])
		recommendation.append(temp)
	return recommendation

def Jaccard_Similarity(x, y):
	intersection = set(x).intersection(set(y))
	union = set(x).union(set(y))
	return 1.0 * len(intersection) / len(union)

if __name__ == '__main__':
	import nltk
	titles = []
	with open('../data/title_10000.txt','r') as f:
		for line in f:
			temp = line.rstrip('\n')
			if temp == '<missing>':
				titles.append('')
			else:
				titles.append(temp)
	categories = []
	with open('../data/categories_10000.txt','r') as f:
		for line in f:
			categories.append(line.split('\t')[0])
	sims = dict()
	for i in range(len(categories)):
		sim = np.eye(500)
		titles_tokenize = [nltk.word_tokenize(titles[500*i+j]) for j in range(500)]
		for k1 in range(499):
			if len(titles_tokenize[k1]) == 0:
				continue
			for k2 in range(k1+1,500):
				if len(titles_tokenize[k2]) == 0:
					continue
				sim[k1,k2] = sim[k2,k1] = Jaccard_Similarity(titles_tokenize[k1],titles_tokenize[k2])
		sims[categories[i]] = sim
	for key in sims.keys():
		exec('np.savez(\'%s\''%('../data/similarity_'+ key +'.npz') + ',similarity = sims[\'%s\'])' %(key))
