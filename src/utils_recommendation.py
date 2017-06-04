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
	dis = np.dot(features, feature)
	temp = [(dis[i], i) for i in range(len(dis))]
	temp_sort = sorted(temp, key = lambda t:-t[0])
	return [idx for _, idx in temp_sort[:num]]

def recommend(sess, model, features, labels, categories, files, image_path, num = 9):
	label_pred, feature_pred = model.predict(sess, image_path)
	recommendation = []
	for i in range(len(label_pred)):
		label = label_pred[i]
		feature = feature_pred[i]
		sample_features, sample_files = subsample(features, labels, files, label)
		features_normalization = feature_normalization(sample_features)
		idx = top_similar_cosine(features_normalization, feature, num)
		temp = (categories[label], [sample_files[i] for i in idx])
		recommendation.append(temp)
	return recommendation