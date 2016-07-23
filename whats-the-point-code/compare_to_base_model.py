import numpy as np
from numpy import *
from numpy.linalg import *
import PIL
from PIL import Image
import skimage
import scipy
import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
from matplotlib.colors import ListedColormap
import matplotlib
from pylab import *
import csv

PASCAL_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'dining table', 'dog', 'horse', 'motorbike', 'person', 'potted plant', 'sheep', 'sofa', 'train', 'tv monitor']

IMAGES_DIR = '/imagenetdb3/olga/data/VOCdevkit/VOC2012/JPEGImages/'
GT_DIR = '/imagenetdb3/olga/data/VOCdevkit/VOC2012/SegmentationClass/'
VAL_FILE_NAME = '/imagenetdb3/olga/data/segm_lmdb/pascal_2012v.txt'
VAL_FILE = open(VAL_FILE_NAME)
IDS = [line[:-1] for line in VAL_FILE]

MODELS_PARENT_DIR = '/imagenetdb3/abearman/caffe/models/fcn-32s-pascal/'

BASE_MODEL_DIR = '2012/fully-supervised/'
BASE_MODEL_NAME = 'fully-supervised_step87000.caffemodel'
OUR_MODEL_DIR = '2012/real1-click1-cls-con-obj/'
OUR_MODEL_NAME = 'real1-click1-cls-con-obj_step51200.caffemodel'

NUM_CLASSES = 21

sys.path.append('/imagenetdb3/abearman/caffe/python/')
print "Importing caffe ..."
import caffe

def main(argv):
	#evaluate_model(BASE_MODEL_DIR, BASE_MODEL_NAME)
	#evaluate_model(OUR_MODEL_DIR, OUR_MODEL_NAME)
	confusion_matrix = build_confusion_matrix()

def build_confusion_matrix():
	base_file_name = 'mean_ious_' + BASE_MODEL_NAME + '.csv'
	ours_file_name = 'mean_ious_' + OUR_MODEL_NAME + '.csv'

	mean_ious_base = []
	mean_ious_ours = []

	confusion_matrix = [(-1, -1), (-1, 2), (2, -1), (2, 2)] 
	image_ids = [''] * 4
	# (base good, ours good), (base good, ours bad), (base bad, ours good), (base bad, ours bad)] 

	with open(base_file_name, 'rb') as base_csvfile:
		for row in csv.reader(base_csvfile):
			for item in row:
				mean_ious_base.append(float(item))	
			break

	with open(ours_file_name, 'rb') as ours_csvfile:
		for row in csv.reader(ours_csvfile): 
			for item in row:
				mean_ious_ours.append(float(item))	
			break

	assert len(mean_ious_base) == len(mean_ious_ours)
	for i in range(len(mean_ious_base)):
		base = mean_ious_base[i]
		ours = mean_ious_ours[i]

		# base good, ours good
		if (base + ours) > (confusion_matrix[0][0] + confusion_matrix[0][1]):
			confusion_matrix[0] = (base, ours)
			image_ids[0] = IDS[i]	
	
		# base good, ours bad
		if (base - ours) > (confusion_matrix[1][0] - confusion_matrix[1][1]) and IDS[i] != '2011_001534':
			confusion_matrix[1] = (base, ours)
			image_ids[1] = IDS[i]

		# base bad, ours good
		if (ours - base) > (confusion_matrix[2][1] - confusion_matrix[2][0]) and IDS[i] != '2011_001793':
			confusion_matrix[2] = (base, ours)
			image_ids[2] = IDS[i]

		# base bad, ours bad
		if (base + ours) < (confusion_matrix[3][0] + confusion_matrix[3][1]) and IDS[i] != '2010_004951':
			confusion_matrix[3] = (base, ours) 
			image_ids[3] = IDS[i]

	print confusion_matrix
	print image_ids
	return confusion_matrix

def evaluate_model(MODEL_DIR, MODEL_NAME):
	PROTOTEXT_FILE = MODELS_PARENT_DIR + MODEL_DIR + 'deploy.prototxt'
	CAFFE_MODEL = MODELS_PARENT_DIR + MODEL_DIR + 'iters/' + MODEL_NAME
	OUTPUT_DIR = MODELS_PARENT_DIR + MODEL_DIR

	# Load net
	net = caffe.Net(PROTOTEXT_FILE, CAFFE_MODEL, caffe.TEST)
	caffe.set_mode_gpu()
	caffe.set_device(3)

	mean_ious_for_model = []

	start_t = time.time();
	time_segm = 0;
	time_other = 0;
	for i in range(len(IDS)):
		if (time.time() - start_t) > 10:
			print "iter:", i, "/", len(IDS), "time_segm:", time_segm, "time_other:", time_other
			start_t = time.time()
			
		tt = time.time();
		image_path = IMAGES_DIR + IDS[i] + '.jpg'
		gt_path = GT_DIR + IDS[i] + '.png'
		predicted_segmentation = segment(image_path, net)	
		time_segm += time.time()-tt;

		tt = time.time()
		iou_for_image = computeIOU(predicted_segmentation, gt_path)
		mean_iou_for_image = mean([iou for iou in iou_for_image if iou != -1])
		mean_ious_for_model.append(mean_iou_for_image)
		print "IOUs for image: " + IDS[i] + ": " + str(iou_for_image)
		print "Mean IOU for image: " + IDS[i] + ": " + str(mean_iou_for_image)
		print ""

		time_other += time.time()-tt

	csv_file = open('mean_ious_' + MODEL_NAME + '.csv', 'wb')
	wr = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
	wr.writerow(mean_ious_for_model)

def computeIOU(predicted, gt_path): 
	num_intersection = [0.0] * NUM_CLASSES
	num_union = [0.0] * NUM_CLASSES

	gt_image = Image.open(gt_path)
	gt_ = np.array(gt_image, dtype=np.float32)

	im_height = gt_.shape[0] # num rows
	im_width = gt_.shape[1]  # num cols
	N = im_height * im_width
	gt_linear = np.resize(gt_, (N,))	

	# Compute IOU
	predicted_linear = np.resize(predicted, (N,))
	for class_ in range(len(num_union)):
		(ni, nu) = jaccard(predicted, gt_, class_)
		num_intersection[class_] = ni;
		num_union[class_] = nu;

	ious = [-1] * 21
	for i in range(len(num_union)):
		iou_for_class = -1  # just initializing 
		if num_union[i] > 0:
			iou_for_class = float(num_intersection[i]) / num_union[i]  
		# set it to -1 if the class does not appear in the image, otherwise leave it as 0.0
		if i not in gt_:
			iou_for_class = -1
		ious[i] = iou_for_class 
	return ious

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

def segment(image_path, net):
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
	out = net.blobs['upscore'].data[0].argmax(axis=0)
	return out


if __name__ == "__main__":
	main(sys.argv)
