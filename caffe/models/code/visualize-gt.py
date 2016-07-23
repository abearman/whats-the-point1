import numpy as np
import skimage
import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pylab
from skimage.draw import circle
import scipy
import PIL
from PIL import Image, ImageDraw 
from PIL import ImageFont, ImageOps 
from matplotlib.colors import ListedColormap
import matplotlib
from pylab import *
import Image
import constants 

DATA_DIR = '/imagenetdb3/olga/data/'
VOC_DIR = DATA_DIR + 'VOCdevkit/VOC2012/' 
IMAGES_DIR = VOC_DIR + 'JPEGImages/'
GT_DIR = VOC_DIR + 'SegmentationClass/'
VAL_FILE = '/imagenetdb3/olga/data/segm_lmdb/pascal_2012v.txt'
OUTPUT_DIR = '/imagenetdb3/abearman/caffe/models/clean-fcn-32s-pascal/2012/gt/'

def main(argv):
	val_file = open(VAL_FILE)
	ids = [line[:-1] for line in val_file]

	for i in range(len(ids)):
		IMG_NAME = ids[i] #'2007_000129'  
		img_path = IMAGES_DIR + IMG_NAME + '.jpg'
		gt_path = GT_DIR + IMG_NAME + '.png'

		# Original image
		orig_img = Image.open(img_path)
		img_arr = np.asarray(orig_img)

		# Ground truth image
		gt = Image.open(gt_path)
		gt_arr = np.copy(np.asarray(gt))
		gt_arr[gt_arr == 255] = 0 # Make unlabelled be background

		labels = np.unique(gt_arr).tolist()
		print labels
		#classes = (np.array(PASCAL_CLASSES)[labels]).tolist()
		#print classes
		#colors = (np.array(COLOR_SCHEME_NAMES)[labels]).tolist()
		#print colors
	
		output = np.zeros((gt_arr.shape[0], gt_arr.shape[1], 3))
		for k in labels:
			i, j = np.where(gt_arr == k)
			output[i, j] = hex_to_rgb(constants.COLOR_SCHEME_HEX[k])	
		output = output.astype(np.uint8)

		#Image.fromarray(output).save('blah.png')
		Image.fromarray(output).save(OUTPUT_DIR + '/' + IMG_NAME + '.png')

		#cb = colorbar_index(ncolors=len(classes), cmap=cmap, shrink=0.5, labels=classes)
		#cb.ax.tick_params(labelsize=12)

def hex_to_rgb(value):
		value = value.lstrip('#')
		lv = len(value)
		return tuple(int(value[i:i+lv/3], 16) for i in range(0, lv, lv/3))

# Convenience functions for working with colour ramps and bars
def colorbar_index(ncolors, cmap, labels=None, **kwargs):
	"""
	This is a convenience function to stop you making off-by-one errors
	Takes a standard colour ramp, and discretizes it,
	then draws a colour bar with correctly aligned labels
	"""
	cmap = cmap_discretize(cmap, ncolors)
	mappable = cm.ScalarMappable(cmap=cmap)
	mappable.set_array([])
	mappable.set_clim(-0.5, ncolors+0.5)
	colorbar = plt.colorbar(mappable, **kwargs)
	colorbar.set_ticks(np.linspace(0, ncolors, ncolors))
	colorbar.set_ticklabels(range(ncolors))
	if labels:
			colorbar.set_ticklabels(labels)
	return colorbar

def cmap_discretize(cmap, N):
	"""
	Return a discrete colormap from the continuous colormap cmap.
			cmap: colormap instance, eg. cm.jet. 
			N: number of colors.

		Example
				x = resize(arange(100), (5,100))
				djet = cmap_discretize(cm.jet, 5)
				imshow(x, cmap=djet)

	"""
	if type(cmap) == str:
			cmap = get_cmap(cmap)
	colors_i = np.concatenate((np.linspace(0, 1., N), (0., 0., 0., 0.)))
	colors_rgba = cmap(colors_i)
	indices = np.linspace(0, 1., N + 1)
	cdict = {}
	for ki, key in enumerate(('red', 'green', 'blue')):
			cdict[key] = [(indices[i], colors_rgba[i - 1, ki], colors_rgba[i, ki]) for i in xrange(N + 1)]
	return matplotlib.colors.LinearSegmentedColormap(cmap.name + "_%d" % N, cdict, 1024)

if __name__ == "__main__":
	main(sys.argv)
