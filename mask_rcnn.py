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
batch_size = 20
max_epochs = 20
max_iters = int(max_epochs*(training_size/batch_size))
shuffle = True

imgs, mask_car, mask_peop, nx_s, ny_s, nw_s, nh_s, valid_mask, labels, out_label_gt, perm = loader("./P&C dataset/", shuffle)

mask_car = np.load("gdrive/My Drive/CIS680/HW2B/car_masks.npy")
mask_peop = np.load("gdrive/My Drive/CIS680/HW2B/ppl_masks.npy")
nxtrain = imgs[:training_size]
nytrain = labels[:training_size]
nmtrain = valid_mask[:training_size]
notrain = out_label_gt[:training_size]

nxstrain = nx_s[:training_size]
nystrain = ny_s[:training_size]
nwstrain = nw_s[:training_size]
nhstrain = nh_s[:training_size]

nxtest = imgs[training_size:]
nytest = labels[training_size:]
nmtest = valid_mask[training_size:]
notest = out_label_gt[training_size:]

nxstest = nx_s[training_size:]
nystest = ny_s[training_size:]
nwstest = nw_s[training_size:]
nhstest = nh_s[training_size:]

tf.reset_default_graph()

x1 = tf.placeholder(tf.float32, [None, 128, 128, 3])
y1 = tf.placeholder(tf.float32, [None, 8, 8, 1])
valid_m = tf.placeholder(tf.float32, [None, 8, 8, 1])
out_label = tf.placeholder(tf.float32, [None, 8, 8, 1])

x_st = tf.placeholder(tf.float32, [None, 8, 8])
y_st = tf.placeholder(tf.float32, [None, 8, 8])
w_st = tf.placeholder(tf.float32, [None, 8, 8])
h_st = tf.placeholder(tf.float32, [None, 8, 8])

def network(xraw, labels, valid_mask, out_label, x_st, y_st, w_st, h_st):

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
		
		sptin = tf.reshape(r5, [batch_size, r5.shape[1], r5.shape[2], r5.shape[3]])

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
			amax = tf.argmax(tf.reshape(sig_clsx, [tf.shape(sig_clsx)[0], -1, 1]), axis=1, output_type=tf.int32)
			acc = tf.round(sig_clsx)
			acc1 = tf.cast(tf.equal(tf.cast(acc, tf.float32), labels), tf.float32)
			accuracy_tr = tf.reduce_mean(tf.boolean_mask(acc1, valid_mask))

		with tf.variable_scope('reg') as scope4:
			
			# reg
			regx = tf.layers.conv2d(inputs=rinter,filters=4,kernel_size=[1,1],bias_initializer=tf.constant_initializer([4,4,8,8]),padding='SAME',name='regx')
			def smooth_l1(x):
				ab = tf.abs(x)
				return tf.where(ab<1, 0.5 * tf.square(x), ab - 0.5)

			regx_oc = tf.reshape(regx, [tf.shape(regx)[0], -1, 4])
			idx = tf.concat([tf.expand_dims(tf.range(tf.shape(regx)[0], dtype=tf.int32), 1), amax[:, 0:1]], axis=1)
			regx_oc = tf.gather_nd(regx_oc, idx)

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

	# Faster RCNN
	theta = tf.transpose(
		tf.convert_to_tensor(
			[regx_oc[:, 2]*16 / 128.0, 
			tf.zeros(shape=[tf.shape(regx_oc)[0]], dtype=tf.float32), 
			(regx_oc[:, 0]*16 - 64) / 64.0,
			tf.zeros(shape=[tf.shape(regx_oc)[0]], dtype=tf.float32), 
			regx_oc[:, 3]*16 / 128.0, 
			(regx_oc[:, 1]*16 - 64) / 64.0]
		)
	)

	cropped_images = transformer(U=xraw, theta=theta, out_size=(22, 22), name='sp1')
	sptx = transformer(U=sptin, theta=theta, out_size=(4, 4), name='sp2')
	print(sptx)

	# conv6
	conv6 = tf.layers.conv2d(inputs=sptx,filters=128,kernel_size=[3,3],padding='SAME',name='conv6')
	bn6 = tf.layers.batch_normalization(conv6, name='bn6')
	r6 = tf.nn.relu(bn6, name='r6')

	# conv7
	conv7 = tf.layers.conv2d(inputs=r6,filters=128,kernel_size=[3,3],padding='SAME',name='conv7')
	bn7 = tf.layers.batch_normalization(conv7, name='bn7')
	r7 = tf.nn.relu(bn7, name='r7')

	# fc1
	fc1 = tf.layers.dense(tf.reshape(r7, [-1, r7.shape[1]*r7.shape[2]*r7.shape[3]]),2,name='fc1')

	out_label = tf.reshape(out_label, [tf.shape(out_label)[0], -1, 1])
	label_cls = tf.gather_nd(out_label, tf.concat([tf.expand_dims(tf.range(tf.shape(out_label)[0], dtype=tf.int32), 1), amax[:, 0:1]], axis=1))
	print(label_cls)
	label_cls = tf.cast(label_cls, tf.int32)

	# Softmax
	with tf.variable_scope('sm') as scope5:
		loss_cls_final = tf.nn.softmax_cross_entropy_with_logits_v2(logits=fc1, labels=tf.one_hot(label_cls, 2))
		loss_cls_final = tf.reduce_mean(loss_cls_final)

		y_hat = tf.cast(tf.argmax(tf.nn.softmax(fc1), axis=1), tf.int32)
		correct_prediction = tf.cast(tf.equal(y_hat, label_cls), tf.float32)
		accuracy_final = tf.reduce_mean(correct_prediction)
		# accuracy_final = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(tf.nn.softmax(fc1), axis=1, output_type=tf.int32), label_cls)))

	return clsx, tf_loss, accuracy_tr, sig_clsx, loss_reg, regx, cropped_images, accuracy_final, loss_cls_final, sptx, label_cls


clsx, tf_loss, accuracy_tr, sig_clx, loss_reg, regx, cropped_images, accuracy_final, loss_cls_final, output_roi, label_roi = network(x1, y1, valid_m, out_label, x_st, y_st, w_st, h_st)

optimizer = tf.train.AdamOptimizer(8e-4)
train_op = optimizer.minimize(tf_loss + loss_reg + loss_cls_final)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

losses_rpn_cl = []
losses_rpn_re = []
losses_cl_final = []

training_roi_list = []
training_roi_label_list = []

testing_roi_list = []
testing_roi_label_list = []

for epoch in range(max_epochs):

	tr_loss = 0.0
	tr_loss_re = 0.0
	tr_loss_final = 0.0
	for i in range(0, nxtrain.shape[0], batch_size):
		nxtr, nytr, nmtr, notr = nxtrain[i:i+batch_size], nytrain[i:i+batch_size], nmtrain[i:i+batch_size], notrain[i:i+batch_size]
		nxstr, nystr, nwstr, nhstr = nxstrain[i:i+batch_size], nystrain[i:i+batch_size], nwstrain[i:i+batch_size], nhstrain[i:i+batch_size]

		_, loss, loss_r, twop1, loss_final, roi, roi_label = sess.run(
			[train_op, tf_loss, loss_reg, cropped_images, loss_cls_final, output_roi, label_roi], 
																feed_dict={x1: nxtr, y1: nytr, valid_m: nmtr, out_label: notr,
																			x_st: nxstr, y_st: nystr,
																			w_st: nwstr, h_st: nhstr})

		if(epoch%(max_epochs-1) == 0):
			training_roi_list.append(roi[0])
			training_roi_label_list.append(roi_label[0])
		tr_loss += loss.mean()
		tr_loss_re += loss_r.mean()
		tr_loss_final += loss_final.mean()
		if(i%(training_size-batch_size)==0 and i!=0):
			losses_rpn_cl.append(tr_loss/float(training_size/batch_size))
			losses_rpn_re.append(tr_loss_re/float(training_size/batch_size))
			losses_cl_final.append(tr_loss_final/float(training_size/batch_size))
			print("Iteration:", int(epoch*(max_iters/max_epochs) + (i/batch_size) + 1), "/", max_iters, 
				"Loss_RPN_CL:", tr_loss/float(training_size/batch_size),
				"|| Loss_RPN_RE:", tr_loss_re/float(training_size/batch_size),
				"|| Loss_CL_Final:", tr_loss_final/float(training_size/batch_size),)

	te_acc_final = 0.0
	for i in range(0, nxtest.shape[0], batch_size):
		nxte, nyte, nmte, note = nxtest[i:i+batch_size], nytest[i:i+batch_size], nmtest[i:i+batch_size], notest[i:i+batch_size]
		nxste, nyste, nwste, nhste = nxstest[i:i+batch_size], nystest[i:i+batch_size], nwstest[i:i+batch_size], nhstest[i:i+batch_size]

		twop1_test, te_final, roi, roi_label = sess.run([cropped_images, accuracy_final, output_roi, label_roi], feed_dict={x1: nxte, y1: nyte, valid_m: nmte, out_label: note,
															x_st: nxste, y_st: nyste,
															w_st: nwste, h_st: nhste}) 

		if(epoch%(max_epochs-1) == 0):
			testing_roi_list.append(roi[0])
			testing_roi_label_list.append(roi_label[0])

		te_acc_final += te_final

	print("Test Accuracy Final:", te_acc_final/float(nxtest.shape[0]/batch_size))

roi_features = np.vstack((np.array(training_roi_list), np.array(testing_roi_list)))
roi_labels = np.vstack((np.array(training_roi_label_list), np.array(testing_roi_label_list)))

# Mask R-CNN

tf.reset_default_graph()

training_roi_features = roi_features[:training_size]
testing_roi_features = roi_features[training_size:]

training_roi_labels = roi_labels[:training_size]
testing_roi_labels = roi_labels[training_size:]

training_car_masks = mask_car[:training_size]
training_ppl_masks = mask_peop[:training_size]
testing_car_masks = mask_car[training_size:]
testing_ppl_masks = mask_peop[training_size:]

batch_size = 1

x1 = tf.placeholder("float", [None, 4, 4, 128])
y1 = tf.placeholder("float", [None])
y2 = tf.placeholder("float", [None])
y3 = tf.placeholder("float", [None, 1])

def mask_rcnn(x1, y1, y2, y3):
	
	m1 = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d(x1, filters=128, kernel_size=[3,3], padding="SAME", kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1), name="conv11"), name="bn11"), name="relu11")
	m2 = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d(m1, filters=128, kernel_size=[3,3], padding="SAME", kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1), name="conv12"), name="bn12"), name="relu12")
	m3 = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d(m2, filters=1, kernel_size=[1,1], padding="SAME", kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1), name="conv13"), name="bn13"), name="relu13")

	# upsample to 22x22 masks
	up1 = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d_transpose(m3, filters=1, kernel_size=[3,3], strides=2, padding="valid", kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1), name="up14"), name="bn14"), name="relu14")
	up2 = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d_transpose(up1, filters=1, kernel_size=[3,3], strides=2, padding="valid", kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1), name="up15"), name="bn15"), name="relu15")
	up3 = tf.layers.conv2d_transpose(up2, filters=1, kernel_size=[4,4], strides=1, padding="valid", kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1), name="up16")

	# loss 
	loss_op = tf.where(y3 == 1., tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y1, logits=up3)), tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y2, logits=up3)))

	# binary mask prediction
	y_hat = tf.cast(tf.round(tf.nn.sigmoid(up3)), tf.float32)

	# accuracy of mask prediction
	correct_prediction = tf.where(y3 == 1., tf.cast(tf.equal(y_hat, y1), tf.float32), tf.cast(tf.equal(y_hat, y2), tf.float32))

	accuracy_operation = tf.reduce_mean(correct_prediction)

	return loss_op, accuracy_operation

tf_loss, accuracy_op = mask_rcnn(x1, y1, y2, y3)

optimizer = tf.train.AdamOptimizer(1e-3)
train_op = optimizer.minimize(tf_loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

training_loss_list = []
testing_acc_list = []

for epoch in range(max_epochs):

	loss_tr = 0.0
	acc_ts = 0.0

	for i in range(0, nxtrain.shape[0], batch_size):

		train_roi, train_roi_label, train_car_mask, train_ppl_mask = training_roi_features[i:i+batch_size], training_roi_labels[i:i+batch_size], training_car_masks[i:i+batch_size], training_ppl_masks[i:i+batch_size]
		train_car_mask = train_car_mask[:, :, :, np.newaxis]
		train_ppl_mask = train_ppl_mask[:, :, :, np.newaxis]
		
		_, loss = sess.run([train_op, tf_loss], feed_dict={x1: train_roi, y1: train_car_mask, y2: train_ppl_mask, y3: train_roi_label})
		
		loss_tr += loss
		
	loss_tr = loss_tr/training_size
	
	training_loss_list.append(loss_tr)
	print("Epoch: ", epoch+1, ", Training Loss: ", loss_tr)

	for j in range(0, nxtest.shape[0], batch_size):
            
		test_roi, test_roi_label, test_car_mask, test_ppl_mask = testing_roi_features[j:j+batch_size], testing_roi_labels[j:j+batch_size], testing_car_masks[j:j+batch_size], testing_ppl_masks[j:j+batch_size]
		
		test_car_mask = test_car_mask[:, :, :, np.newaxis]
		test_ppl_mask = test_ppl_mask[:, :, :, np.newaxis]
		
		acc = sess.run([accuracy_op], feed_dict={x1: test_roi, y1: test_car_mask, y2: test_ppl_mask, y3: test_roi_label})

		acc_ts += acc[0]
	
	acc_ts = acc_ts/nxtest.shape[0]
	
	testing_acc_list.append(acc_ts)
	if(epoch == (max_epochs-1)):
		print("Final Testing Accuracy: ", acc_ts)

np.save("./mask_tr_losses.npy", training_loss_list)
# plt.plot(len(training_loss_list), training_loss_list)
# plt.xlabel("#Epochs")
# plt.ylabel("Mask R-CNNTraining loss")
# plt.show()
