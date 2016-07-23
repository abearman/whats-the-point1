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
import compute_mIOU

IMAGES_DIR = GT_DIR = VAL_DIR = '' 
IMAGES_DIR = '/imagenetdb3/olga/data/VOCdevkit/VOC2012/JPEGImages/'
BBOX_DIR = '/imagenetdb3/olga/data/PASCAL_bbox_masks/'
GT_DIR = '/imagenetdb3/olga/data/VOCdevkit/VOC2012/SegmentationClass/'
GT_DIR1 = '/imagenetdb3/olga/data/PASCAL_SBD/dataset/cls_plus_VOC/'
TRAIN_FILE = '/imagenetdb3/olga/data/segm_lmdb/pascal_2012t_SBDtv_minus_2012v.txt' 
VAL_FILE =	'/imagenetdb3/olga/data/segm_lmdb/pascal_2012v.txt'

NUM_CLASSES = 21 


def main(argv):
	#train_file = open(TRAIN_FILE)
	val_file = open(VAL_FILE)
	ids = [line[:-1] for line in val_file]

	ni0 = [0.0] * NUM_CLASSES
	nu0 = [0.0] * NUM_CLASSES
	ni1 = [0.0] * NUM_CLASSES
	nu1 = [0.0] * NUM_CLASSES

	counter = 1
	for i in range(len(ids)):
		print counter
		counter += 1	

		gt_path = GT_DIR + ids[i] + '.png'
		if not os.path.isfile(gt_path): gt_path = GT_DIR1 + ids[i] + '.png'
		gt_arr = np.array(Image.open(gt_path))
	
		bbox_mask_path = BBOX_DIR + ids[i] + '.png'
		bbox_mask_arr = np.array(Image.open(bbox_mask_path))
		#Image.fromarray(bbox_mask_arr).save(ids[i] + "_orig.png")

		# Method 0: just the bbox mask compared to the ground truth
		(ni0, nu0) = compute_mIOU.computeIOU(bbox_mask_arr, gt_arr, ni0, nu0) 

		# Method 1: shrink 
		bbox_seg_arr = method1(bbox_mask_arr, a=0.9)		
		#Image.fromarray(bbox_seg_arr).save(ids[i]+ ".png")
		(ni1, nu1) = compute_mIOU.computeIOU(bbox_seg_arr, gt_arr, ni1, nu1)
	
	print "method 0 mIOU: " + str(calculate_iou_from_class_counts(ni0, nu0))	
	print "method 1 mIOU: " + str(calculate_iou_from_class_counts(ni1, nu1))


def method1(bbox_mask_arr, a=0.5):
	bbox_seg_arr = np.copy(bbox_mask_arr)
	for k in range(1, NUM_CLASSES):
		idx = np.argwhere(bbox_mask_arr == k)

		if idx.size > 0:
			rows = idx[:,0]
			cols = idx[:,1]

			min_row = min(rows)
			max_row = max(rows)
			min_col = min(cols)
			max_col = max(cols)
	
			orig_row_length = max_row - min_row
			resized_row_length = orig_row_length * sqrt(a)
			row_length_change = orig_row_length - resized_row_length
			new_min_row = min_row + (row_length_change / 2)
			new_max_row = max_row - (row_length_change / 2)

			orig_col_length = max_col - min_col
			resized_col_length = orig_col_length * sqrt(a)
			col_length_change = orig_col_length - resized_col_length
			new_min_col = min_col + (col_length_change / 2)
			new_max_col = max_col - (col_length_change / 2)

			bbox_seg_arr[min_row:max_row+1, min_col:new_min_col+1] = 255
			bbox_seg_arr[min_row:max_row+1, new_max_col:max_col+1] = 255
			bbox_seg_arr[min_row:new_min_row+1, min_col:max_col+1] = 255
			bbox_seg_arr[new_max_row:max_row+1, min_col:max_col+1] = 255

	return bbox_seg_arr

def calculate_iou_from_class_counts(num_intersection, num_union):
	class_ious = [0.0] * NUM_CLASSES
	for k in range(NUM_CLASSES):
		if num_union[k] > 0:
			class_ious[k] = float(num_intersection[k] / num_union[k])
		else:
			class_ious[k] = 0.0
	return sum(class_ious) / NUM_CLASSES	# mean IOU


if __name__ == "__main__":
	main(sys.argv)
