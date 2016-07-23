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
from matplotlib.colors import colorConverter
import constants 

DATA_DIR = '/imagenetdb3/olga/data/'
VOC_DIR = DATA_DIR + 'VOCdevkit/VOC2012/' 
IMAGES_DIR = VOC_DIR + 'JPEGImages/'
CLICKS_DIR = DATA_DIR + 'PASCAL_AMT_clicks/real1/'

def rgb2gray(rgb):
	r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
	gray = (0.2989 * r) + (0.5870 * g) + (0.1140 * b)
	return np.asarray(np.dstack((gray, gray, gray)), dtype=np.uint8)

def main(argv):
	#IMG_NAME = '2007_000129' # bikes
	#IMG_NAME = '2009_000426' #	buses
	#IMG_NAME = '2007_000346' # girl and bottle
	#IMG_NAME = '2007_009841' # horse and boy 

	#IMG_NAME = '2009_001433' # 2 people on horses
	#IMG_NAME = '2009_003071' # little boy at table
	#IMG_NAME = '2009_002732' # Girl with baby and bottle
	IMG_NAME = '2010_000241' # bird

	img_path = IMAGES_DIR + IMG_NAME + '.jpg'
	clicks_path = CLICKS_DIR + IMG_NAME + '.png'

	# Original image
	orig_img = Image.open(img_path)
	img_arr = np.asarray(orig_img)

	# Clicks
	points_arr = np.asarray(Image.open(clicks_path))
	clicks_im = np.copy(img_arr)
	clicks_im *= 0.7 # 0.8. 0.9, 0.7, 0.7
	clicks_im = Image.fromarray(clicks_im.astype(np.uint8))
	draw = ImageDraw.Draw(clicks_im)
	sz = 8
	sz_b = 12
	for r in range(img_arr.shape[0]):
		for c in range(img_arr.shape[1]):
			k = points_arr[r,c]
			if k != 255:
				draw.ellipse((c-sz_b, r-sz_b, c+sz_b, r+sz_b), fill=constants.COLOR_SCHEME_HEX[0])
				draw.ellipse((c-sz, r-sz, c+sz, r+sz), fill=constants.COLOR_SCHEME_HEX[k])
	# outline ='white
	clicks_im.save('clicks-viz.png')

def hex_to_rgb(value):
	value = value.lstrip('#')
	lv = len(value)
	return tuple(int(value[i:i+lv/3], 16) for i in range(0, lv, lv/3))

if __name__ == "__main__":
	main(sys.argv)
