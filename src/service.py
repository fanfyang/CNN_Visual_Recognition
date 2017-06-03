from model_vgg16 import *
from utils_recommendation import *

from functools import wraps
from flask import Flask,request, redirect, current_app, jsonify

# define jsonp decorator/wrap for cross domain api call
def support_jsonp(f):
    """Wraps JSONified output for JSONP"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        callback = request.args.get('callback', False)
        if callback:
            content = str(callback) + '(' + str(f().data) + ')'
            return current_app.response_class(content, mimetype='application/json')
        else:
            return f(*args, **kwargs)
    return decorated_function

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

features = np.load('../model/vgg/feature_' + str(version) + '.npz')['feature']

app = Flask(__name__)

@app.route("/recommend" , methods = ["GET"])
@support_jsonp
def model_recommend_api():
	img = request.args.get('img','')
	try:
		path = '../data/'+ img + '.jpg'
		category, files = recommend(sess, vgg, features, labels, categories, files, [path])[0]
		response = {'status':0, 'category':category, 'recommendations':files}
	except:
		response = {'status':1}
	return jsonify(response)

app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False, processes=1)