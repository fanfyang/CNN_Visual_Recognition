from model_vgg16 import *

config = Config(num_classes = 20, batch_size = 70, lr = 0.001, l2 = 0.0)
vgg = model_vgg16_20(config)

x,y,z = fetch_data(file = True)
X_train = x[:700]
X_val = x[700:]
y_train = y[:700]
y_val = y[700:]

sess = tf.Session()
sess.run(tf.global_variables_initializer())
vgg.load_parameters(sess,'../data/vgg16/vgg16_weights.npz',rand_init = ['fc8_W', 'fc8_b'])
vgg.train(sess,X_train,y_train,X_val,y_val)