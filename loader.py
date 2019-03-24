from PIL import Image
import numpy as np

	
def intersection_over_union(boxA, boxB):
    
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou  
    
def sliding_window(image, stepSize, windowSize):
	
	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


def loader(path, shuffle):
	img = np.zeros((2000,128,128,3), dtype=np.float32)
	mc = []
	mp = []

	for i in range(2000):
		img[i] = np.array(Image.open(path+"/img/{0:06d}.jpg".format(i)), dtype=np.float32)
		mc.append(np.array(Image.open(path+"/mask/car/{0:06d}.png".format(i)), dtype=np.float32))
		mp.append(np.array(Image.open(path+"/mask/people/{0:06d}.png".format(i)), dtype=np.float32))

	mc = np.array(mc)
	mp = np.array(mp)

	lc = np.loadtxt(path+"/label_car.txt", delimiter=',', dtype=np.float32)
	lp = np.loadtxt(path+"/label_people.txt", delimiter=',', dtype=np.float32)

	# print(lc[0], "here",lp[0])

	nimg = np.empty(img.shape, dtype=img.dtype)
	nmc = np.empty(mc.shape, dtype=mc.dtype)
	nmp = np.empty(mp.shape, dtype=mp.dtype)
	nlc = np.empty(lc.shape, dtype=lc.dtype)
	nlp = np.empty(lp.shape, dtype=lp.dtype)
	nl = np.empty(lp.shape, dtype=lp.dtype)

	if(shuffle==True):
		np.random.seed(None)
		perm = np.random.permutation(2000)
		for oidx,nidx in enumerate(perm):
			nimg[oidx] = img[nidx]
			nmc[oidx] = mc[nidx]
			nmp[oidx] = mp[nidx]
			nlc[oidx] = lc[nidx]
			nlp[oidx] = lp[nidx]
	else:
		perm = np.arange(2000)
		nimg = img
		nmc = mc
		nmp = mp
		nlc = lc
		nlp = lp

	# print(nlc[0],lc[perm[0]], perm[0],lp[perm[0]],nlp[0])

	bc = nlc/16.0
	x_star_c = bc[:,0] + (bc[:,2]/2.0 )
	y_star_c = bc[:,1] + (bc[:,3]/2.0 )
	w_star_c = bc[:,2]
	h_star_c = bc[:,3] 

	bp = nlp/16.0
	x_star_p = bp[:,0] + (bp[:,2]/2.0 )
	y_star_p = bp[:,1] + (bp[:,3]/2.0 )
	w_star_p = bp[:,2]
	h_star_p = bp[:,3] 

	#star = np.zeros([2000,4])
	#star[:,0] = x_star
	#star[:,1] = y_star
	#star[:,2] = b[:,2]
	#star[:,3] = b[:,3]

	size_of_window = 3 # 48/16 = 3

	mask = np.zeros([10,10])
	label = np.zeros([10,10])
	mask_all = np.zeros([2000,8,8,1])
	label_all = np.zeros([2000,8,8,1])
	x_star = np.zeros([2000,8,8], dtype=np.float32)
	y_star = np.zeros([2000,8,8], dtype=np.float32)
	w_star = np.zeros([2000,8,8], dtype=np.float32)
	h_star = np.zeros([2000,8,8], dtype=np.float32)
	out_label = np.zeros([2000,8,8,1], dtype=np.float32)

	for i in range(2000):
		mask = np.zeros([10,10])
		label = np.zeros([10,10])

		for x in range(0,8):
			for y in range(0,8):
				predicted_box = [x,y,x+size_of_window-1,y+size_of_window-1]

				gt_car = nlc[i]
				new_gt_car = gt_car/16
				new_gt_car[2] = new_gt_car[2]+new_gt_car[0]
				new_gt_car[3] = new_gt_car[3]+new_gt_car[1]
				new_gt_car = new_gt_car+1

				gt_peep = nlp[i]
				new_gt_peep = gt_peep/16
				new_gt_peep[2] = new_gt_peep[2]+new_gt_peep[0]
				new_gt_peep[3] = new_gt_peep[3]+new_gt_peep[1]
				new_gt_peep = new_gt_peep+1
				half_window = size_of_window/2

				iou_car = intersection_over_union(new_gt_car, predicted_box)
				iou_peep = intersection_over_union(new_gt_peep, predicted_box)

				iou = max(iou_car,iou_peep)
				if(iou==iou_car):
					x_star[i,y,x] = x_star_c[i]
					y_star[i,y,x] = y_star_c[i]
					w_star[i,y,x] = w_star_c[i]
					h_star[i,y,x] = h_star_c[i]
					out_label[i,y,x,0] = 0
				else:
					x_star[i,y,x] = x_star_p[i]
					y_star[i,y,x] = y_star_p[i]
					w_star[i,y,x] = w_star_p[i]
					h_star[i,y,x] = h_star_p[i]
					out_label[i,y,x,0] = 1

				if iou > 0.5 or iou<0.1:
					x_coord = x+1
					y_coord = y+1
					mask[y_coord,x_coord] = 1
				# if iou <= 0.5 and iou>=0.1:
				# 	x_coord = x+1
				# 	y_coord = y+1
				# 	label[y_coord,x_coord] = -1
				if iou>0.5:
					x_coord = x+1
					y_coord = y+1
					label[y_coord,x_coord] = 1

		mask_all[i,:,:,0] = mask[1:9,1:9]
		label_all[i,:,:,0] = label[1:9,1:9]

		# if(i==0):
		# 	print(x_star[0],y_star[0],w_star[0],h_star[0],nlc[0],nlp[0])

	return nimg, nmc, nmp, x_star, y_star, w_star, h_star, mask_all, label_all, out_label, perm
