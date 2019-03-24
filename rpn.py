import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from loader import loader
from spatial_transformer import transformer
import pdb
import cv2


training_size = 1600
batch_size = 50
test_bs = 100
max_epochs = 20
max_iters = int(max_epochs*(training_size/batch_size))
shuffle = True

imgs, mask_car, mask_peop, nx_s, ny_s, nw_s, nh_s, valid_mask, labels, out_label_gt, perm = loader("./P&C dataset/", shuffle)

nxtrain = imgs[:training_size]
nytrain = labels[:training_size]
nmtrain = valid_mask[:training_size]

nxstrain = nx_s[:training_size]
nystrain = ny_s[:training_size]
nwstrain = nw_s[:training_size]
nhstrain = nh_s[:training_size]

nxtest = imgs[training_size:]
nytest = labels[training_size:]
nmtest = valid_mask[training_size:]

nxstest = nx_s[training_size:]
nystest = ny_s[training_size:]
nwstest = nw_s[training_size:]
nhstest = nh_s[training_size:]

tf.reset_default_graph()

x1 = tf.placeholder(tf.float32, [None, 128, 128, 3])
y1 = tf.placeholder(tf.float32, [None, 8, 8, 1])
valid_m = tf.placeholder(tf.float32, [None, 8, 8, 1])

x_st = tf.placeholder(tf.float32, [None, 8, 8])
y_st = tf.placeholder(tf.float32, [None, 8, 8])
w_st = tf.placeholder(tf.float32, [None, 8, 8])
h_st = tf.placeholder(tf.float32, [None, 8, 8])

def network(xraw, labels, valid_mask, x_st, y_st, w_st, h_st, batch_size):

	with tf.variable_scope('base_net') as scope:

		# conv1
		conv1 = tf.layers.conv2d(inputs=xraw,filters=8,kernel_size=[3,3],padding='SAME',name='conv1')
		bn1 = tf.layers.batch_normalization(conv1, name='bn1')
		r1 = tf.nn.relu(bn1, name='r1')
		mp1 = tf.layers.max_pooling2d(r1,pool_size=2,strides=2,padding='valid',name='mp1')

		# conv2
		conv2 = tf.layers.conv2d(inputs=mp1,filters=16,kernel_size=[3,3],padding='SAME',name='conv2')
		bn2 = tf.layers.batch_normalization(conv2, name='bn2')
		r2 = tf.nn.relu(bn2, name='r2')
		mp2 = tf.layers.max_pooling2d(r2,pool_size=2,strides=2,padding='valid',name='mp2')

		# conv3
		conv3 = tf.layers.conv2d(inputs=mp2,filters=32,kernel_size=[3,3],padding='SAME',name='conv3')
		bn3 = tf.layers.batch_normalization(conv3, name='bn3')
		r3 = tf.nn.relu(bn3, name='r3')
		mp3 = tf.layers.max_pooling2d(r3,pool_size=2,strides=2,padding='valid',name='mp3')

		# conv4
		conv4 = tf.layers.conv2d(inputs=mp3,filters=64,kernel_size=[3,3],padding='SAME',name='conv4')
		bn4 = tf.layers.batch_normalization(conv4, name='bn4')
		r4 = tf.nn.relu(bn4, name='r4')
		mp4 = tf.layers.max_pooling2d(r4,pool_size=2,strides=2,padding='valid',name='mp4')

		# conv5
		conv5 = tf.layers.conv2d(inputs=mp4,filters=128,kernel_size=[3,3],padding='SAME',name='conv5')
		bn5 = tf.layers.batch_normalization(conv5, name='bn5')
		r5 = tf.nn.relu(bn5, name='r5')

	with tf.variable_scope('rpn') as scope2:

		# intermiddiate
		convinter = tf.layers.conv2d(inputs=r5,filters=128,kernel_size=[3,3],padding='SAME',name='convinter')
		bninter = tf.layers.batch_normalization(convinter, name='bninter')
		rinter = tf.nn.relu(bninter, name='rinter')

		with tf.variable_scope('cls') as scope3:

			# cls
			clsx = tf.layers.conv2d(inputs=rinter,filters=1,kernel_size=[1,1],padding='SAME',name='clsx')

			loss_cl = tf.nn.sigmoid_cross_entropy_with_logits(logits=clsx, labels=tf.cast(labels, tf.float32))
			tf_loss = tf.reduce_mean(tf.boolean_mask(loss_cl, valid_mask))

			sig_clsx = tf.sigmoid(clsx)
			acc = tf.cast(tf.equal(tf.cast(tf.greater(sig_clsx, 0.5), tf.float32), labels), tf.float32)
			accuracy_tr = tf.reduce_mean(tf.boolean_mask(acc, valid_mask))

		with tf.variable_scope('reg'):
			
			# reg
			regx = tf.layers.conv2d(inputs=rinter,filters=4,kernel_size=[1,1],bias_initializer=tf.constant_initializer([4,4,8,8]),padding='SAME',name='regx')

			def smooth_l1(x):
				ab = tf.abs(x)
				return tf.where(ab<1, 0.5 * tf.square(x), ab - 0.5)

			w_a = float(48)
			t_x = (regx[:, :, :, 0] - x_st) / w_a
			t_y = (regx[:, :, :, 1] - y_st) / w_a
			t_w = tf.log((regx[:, :, :, 2] / (w_st + 1e-5)) + 1e-5)
			t_h = tf.log((regx[:, :, :, 3] / (h_st + 1e-5)) + 1e-5)

			t_x = tf.where(tf.is_nan(t_x), tf.zeros_like(t_x), t_x)
			t_y = tf.where(tf.is_nan(t_y), tf.zeros_like(t_y), t_y)
			t_w = tf.where(tf.is_nan(t_w), tf.zeros_like(t_w), t_w)
			t_h = tf.where(tf.is_nan(t_h), tf.zeros_like(t_h), t_h)
			t_x = tf.where(tf.is_inf(t_x), tf.zeros_like(t_x), t_x)
			t_y = tf.where(tf.is_inf(t_y), tf.zeros_like(t_y), t_y)
			t_w = tf.where(tf.is_inf(t_w), tf.zeros_like(t_w), t_w)
			t_h = tf.where(tf.is_inf(t_h), tf.zeros_like(t_h), t_h)
			loss_reg = (smooth_l1(t_x) + smooth_l1(t_y) + smooth_l1(t_w) + smooth_l1(t_h))/4.0

			loss_reg = tf.boolean_mask(loss_reg, valid_m[:,:,:,0])
			loss_reg = tf.reduce_mean(loss_reg)

	return clsx, tf_loss, accuracy_tr, sig_clsx, loss_reg, regx

clsx, tf_loss, accuracy_tr, sig_clx, loss_reg, regx = network(x1, y1, valid_m, x_st, y_st, w_st, h_st, batch_size)

optimizer = tf.train.AdamOptimizer(1e-3)
train_op = optimizer.minimize(tf_loss + loss_reg)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

losses_cl = []
losses_re = []
accuracies = []

for epoch in range(max_epochs):

	tr_acc = 0.0
	tr_loss = 0.0
	tr_loss_re = 0.0
	for i in range(0, nxtrain.shape[0], batch_size):
		nxtr, nytr, nmtr = nxtrain[i:i+batch_size], nytrain[i:i+batch_size], nmtrain[i:i+batch_size]

		nxstr, nystr, nwstr, nhstr = nxstrain[i:i+batch_size], nystrain[i:i+batch_size], nwstrain[i:i+batch_size], nhstrain[i:i+batch_size]

		_, loss, accuracy, loss_r, cls_feat, reg_feat = sess.run([train_op, tf_loss, accuracy_tr, loss_reg, sig_clx, regx], 
																feed_dict={x1: nxtr, y1: nytr, valid_m: nmtr,
																		x_st: nxstr, y_st: nystr,
																		w_st: nwstr, h_st: nhstr})
		tr_acc += accuracy
		tr_loss += loss.mean()
		tr_loss_re += loss_r.mean()
		if(i%(training_size-batch_size)==0 and i!=0):
			losses_cl.append(tr_loss/float(training_size/batch_size))
			losses_re.append(tr_loss_re/float(training_size/batch_size))
			print("Iteration:", int(epoch*(max_iters/max_epochs) + (i/batch_size) + 1), "/", max_iters, 
				"|| Accuracy:", tr_acc/float(training_size/batch_size), "|| Loss_CL:", tr_loss/float(training_size/batch_size),
				"|| Loss_RE:", tr_loss_re/float(training_size/batch_size))

	t_acc = 0.0
	t_loss = 0.0
	for i in range(0, nxtest.shape[0], test_bs):
		nxte, nyte, nmte = nxtest[i:i+test_bs], nytest[i:i+test_bs], nmtest[i:i+test_bs]
		nxste, nyste, nwste, nhste = nxstest[i:i+test_bs], nystest[i:i+test_bs], nwstest[i:i+test_bs], nhstest[i:i+test_bs]

		test_accuracy, loss_r_te = sess.run([accuracy_tr, loss_reg], feed_dict={x1: nxte, y1: nyte, valid_m: nmte,
														x_st: nxste, y_st: nyste,
														w_st: nwste, h_st: nhste}) 
		t_acc += test_accuracy
		t_loss += loss_r_te.mean()

	print("Test accuracy:", t_acc/float(nxtest.shape[0]/test_bs), "Test Regression Loss:", t_loss/float(nxtest.shape[0]/test_bs))

# np.save("./losses_cl.npy",losses_cl)
np.save("./losses_re.npy",losses_re)
# plt.plot(np.arange(len(losses_cl)), losses_cl)
# plt.xlabel("#Epochs")
# plt.ylabel("Classification Training Loss")
# plt.show()

# plt.plot(np.arange(len(losses_re)), losses_re)
# plt.xlabel("#Epochs")
# plt.ylabel("Regression Training Loss")
# plt.show()

