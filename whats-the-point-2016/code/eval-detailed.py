import numpy as np
from numpy import *
from numpy.linalg import *
import PIL
from PIL import Image
import skimage
import scipy
import sys
import os
import matplotlib.pyplot as plot
import time
import copy

BASE_FOLDER = '/imagenetdb3/olga/data/VOCdevkit/VOC2012/'
IMAGES_DIR = BASE_FOLDER + 'JPEGImages/'
GT_DIR = BASE_FOLDER + 'SegmentationClass/'
VAL_FILE = '/imagenetdb3/olga/data/segm_lmdb/pascal_2012v.txt'
#VAL_FILE = BASE_FOLDER + 'ImageSets/Segmentation/val_2011.txt'
MODELS_DIR = '/imagenetdb3/abearman/caffe/models/clean-fcn-32s-pascal/'

DIR = '2012/image-level-labels-constraint'
MODEL = 'image-level-labels-constraint_step6400.caffemodel'

PROTOTEXT_FILE = MODELS_DIR + '/deploy.prototxt'
CAFFE_MODEL = MODELS_DIR + DIR + '/iters/' + MODEL
OUTPUT_DIR = MODELS_DIR + DIR

SAVE_FOLDER = '';

def main():
	sys.path.append('/imagenetdb3/abearman/caffe/python/')
	print "Importing caffe ..."
	import caffe

	val_file = open(VAL_FILE)
	ids = [line[:-1] for line in val_file]
	#from random import shuffle
	#shuffle(ids)
	
	overall_acc = 0
	accuracies = []

	# Load net
	net = caffe.Net(PROTOTEXT_FILE, CAFFE_MODEL, caffe.TEST)
	caffe.set_mode_gpu()
	caffe.set_device(1)

	num_intersection = {}; 
	num_union = {}; 
	current_iou = {}

	settings = ['regular','addbg','fixbg','removeclasses','fixboth'];  #,'removebg'
	#settings = ['addbg','regular'];
	for s in settings:
		num_intersection[s] = [0.0]*21;
		num_union[s] = [0.0]*21

	start_t = time.time();
	time_segm = 0;
	time_other = 0;
	for i in range(len(ids)):
		if (time.time() - start_t) > 10:
			print "iter:", i,"/",len(ids),"time_segm:",time_segm,"time_other:",time_other
			for s in settings:
				print '---'
				print '     ',s
				print current_iou[s]
				print "Mean: ", mean(current_iou[s])
			print ""
			print ""
			start_t = time.time()
			
		tt = time.time();
		image_path = IMAGES_DIR + ids[i] + '.jpg'
		gt_path = GT_DIR + ids[i] + '.png'
		
		obj_path = ''
		if i == 0:
			print 'obj_path:',obj_path

		predicted_segmentation = segment(image_path, net,obj_path)	
		time_segm += time.time()-tt;

		tt = time.time()
		for s in settings:
			bAddBg = False
			bRemoveBg = False
			bRemoveClasses = False

			if s == 'regular':
				pass
			elif s == 'addbg':
				bAddBg = True
			elif s == 'removebg':
				bRemoveBg = True
			elif s == 'fixbg':
				bAddBg = True
				bRemoveBg = True
			elif s == 'removeclasses':
				bRemoveClasses = True
			elif s == 'fixboth':
				bAddBg = True
				bRemoveBg = True
				bRemoveClasses = True
			else:
				raise Exception('unknown type')

			(ni,nu) = computeIOU(predicted_segmentation,gt_path,bAddBg,bRemoveBg,bRemoveClasses);
			num_intersection[s] = np.add(num_intersection[s],ni)
			num_union[s] = np.add(num_union[s],nu)
			current_iou[s] = [float(num_intersection[s][i])/num_union[s][i] if num_union[s][i] > 0 else 0 for i in range(len(num_union[s]))];

		time_other += time.time()-tt
		
	for s in settings:
		print "Mean iou per class: ", repr(current_iou[s])
		print "Mean iou: ", mean(current_iou[s])

def computeIOU(predicted_init, gt_path,bAddBg,bRemoveBg,bRemoveClasses): 
	# Count number of class appearances
	#class_appearances = [0] * 21
	gt_image = Image.open(gt_path)
	gt_ = np.array(gt_image, dtype=np.float32)

	predicted = copy.deepcopy(predicted_init)
	if bAddBg:
		ind = (predicted > 0) & (gt_ == 0)
		predicted[ind] = 0
	if bRemoveBg:
		ind = (predicted == 0) & (gt_ > 0)
		predicted[ind] = gt_[ind]
	if bRemoveClasses:
		existent = np.unique(gt_)
		toRemove = np.setdiff1d(np.unique(predicted),existent)
		for r in toRemove:
			ind = predicted == r
			predicted[ind] = gt_[ind]

	im_height = gt_.shape[0] # num rows
	im_width = gt_.shape[1]  # num cols
	N = im_height * im_width
	gt_linear = np.resize(gt_, (N,))	

	# Compute IOU
	num_int = [0.0]*21
	num_un = [0.0]*21
	for class_ in range(len(num_un)):
		(ni,nu) = jaccard(predicted, gt_, class_)
		num_int[class_] = ni;
		num_un[class_] = nu;
	return (num_int,num_un)

# Function: iou
# --------------
# Computes the Jaccard similarity of two flattened images for a class
def jaccard(pred, gt, class_):
	p = (pred == class_);
	g = (gt == class_);
	gm = (gt == 255);

	intersection = sum(p & g);
	union = sum(p | g) - sum(p & gm);
	return (intersection,union)

def segment(image_path, net,obj_path):
	# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
	im = Image.open(image_path)
	(width, height) = im.size

	in_ = np.array(im, dtype=np.float32)
	orig_image = in_
	in_ = in_[:,:,::-1]
	in_ -= np.array((104.00698793, 116.66876762, 122.67891434))
	in_ = in_.transpose((2, 0, 1))

	net.blobs['data'].reshape(1, *in_.shape)
	net.blobs['data'].data[...] = in_

	# run net and take argmax for prediction
	net.forward()

	if len(obj_path) == 0:
		out = net.blobs['upscore'].data[0].argmax(axis=0)
	else:
		obj = Image.open(obj_path)
		obj = np.array(obj,dtype=np.float32)/255;

		prob = np.exp(net.blobs['upscore'].data[0])
		prob = prob[1:,:,:] #no background
		Z = prob.sum(axis = 0)
		prob_c = prob / Z * obj # prob(class | not background) * P(not background)

		(a,b) = obj.shape
		obj = obj.reshape((1,a,b))
		pp = np.concatenate((1-obj,prob_c))
		out = pp.argmax(axis = 0)

	#import scipy.io
	#scipy.io.savemat('out.mat', {'out': out})

	if len(SAVE_FOLDER) > 0:
		if not os.path.exists(SAVE_FOLDER):
			os.makedirs(SAVE_FOLDER)
	#labelled_img = skimage.color.label2rgb(out)
		#labelled_img = scipy.misc.toimage(out)
		out = np.uint8(out)
		labelled_img = Image.fromarray(out,mode='L');
		plot.imshow(labelled_img);

		basename = os.path.splitext(os.path.basename(image_path))[0]
	#plot.imshow(labelled_img)
	#im.save(basename + '_in.png')
		labelled_img.save(SAVE_FOLDER + '/' + basename + '.png')
	
	return out

if __name__ == "__main__":
	main()
