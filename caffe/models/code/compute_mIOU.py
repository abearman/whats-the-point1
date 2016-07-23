import numpy as np
from numpy import *
from numpy.linalg import *
import PIL
from PIL import Image
import skimage
import scipy
import sys
import os
from pylab import *


def computeIOU(predicted_arr, gt_arr, num_intersection, num_union):
	im_height = gt_arr.shape[0] # num rows
	im_width = gt_arr.shape[1]	# num cols

	# Add on intersection and union
	for class_ in range(len(num_union)):
		(ni, nu) = jaccard(predicted_arr, gt_arr, class_)
		num_intersection[class_] += ni
		num_union[class_] += nu

	return (num_intersection, num_union)


# Function: iou
# --------------
# Computes the Jaccard similarity of two flattened images for a class
def jaccard(pred, gt, class_):
	p = (pred == class_);
	g = (gt == class_);
	gm = (gt == 255);

	intersection = sum(p & g);
	union = sum(p | g) - sum(p & gm);
	return (intersection, union)


