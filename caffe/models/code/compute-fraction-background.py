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

IMAGES_DIR = GT_DIR = VAL_DIR = '' 
IMAGES_DIR = '/imagenetdb3/olga/data/VOCdevkit/VOC2012/JPEGImages/'
GT_DIR = '/imagenetdb3/olga/data/VOCdevkit/VOC2012/SegmentationClass/'
GT_DIR1 = '/imagenetdb3/olga/data/PASCAL_SBD/dataset/cls_plus_VOC/'
TRAIN_FILE = '/imagenetdb3/olga/data/segm_lmdb/pascal_2012t_SBDtv_minus_2012v.txt' 

print IMAGES_DIR
print GT_DIR
print TRAIN_FILE

NUM_CLASSES = 21 

def main(argv):
	train_file = open(TRAIN_FILE)
	ids = [line[:-1] for line in train_file]

	num_pix_background = 0
	num_pix_total = 0
	
	start_t = time.time();
	time_segm = 0;
	time_other = 0;
	for i in range(len(ids)):
		if (time.time() - start_t) > 10:
			print "iter:", i,"/",len(ids),"time_segm:",time_segm,"time_other:",time_other
			print "Current fraction background: ", 1.0 * num_pix_background / num_pix_total 
			print ""
			start_t = time.time()
			
		tt = time.time();
		image_path = IMAGES_DIR + ids[i] + '.jpg'
		gt_path = GT_DIR + ids[i] + '.png'
		if not os.path.isfile(gt_path): gt_path = GT_DIR1 + ids[i] + '.png'
		gt = Image.open(gt_path)
		gt_arr = np.asarray(gt)	

		num_pix_total += (gt_arr.shape[0] * gt_arr.shape[1]) # Width x height
		num_pix_background += len(gt_arr[gt_arr == 0])

		time_segm += time.time()-tt;

		tt = time.time()
		time_other += time.time()-tt
		
	print "Fraction background: ", 1.0 * num_pix_background / num_pix_total 

if __name__ == "__main__":
	main(sys.argv)
