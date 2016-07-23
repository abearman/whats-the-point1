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

DATA_DIR = '/imagenetdb3/olga/data/'
VOC_DIR = DATA_DIR + 'VOCdevkit/VOC2012/' 
IMAGES_DIR = VOC_DIR + 'JPEGImages/'
GT_DIR = VOC_DIR + 'SegmentationClass/'
OBJECTNESS_DIR = DATA_DIR + 'PASCAL_objectness/'
CLICKS_DIR = DATA_DIR + 'PASCAL_AMT_clicks/real1/'

COLOR_SCHEME_NAMES      = ['black', 'tomato', 'orange', 'yellow', 'green', 'blue', 'purple', 'magenta', 'cyan',
                                'darkblue', 'darkgreen', 'palevioletred', 'maroon', 'gold', 'salmon',
                                'firebrick', 'orchid', 'greenyellow', 'lavender', 'mistyrose', 'darkturquoise']

COLOR_SCHEME_HEX = ['#000000', '#FF6347', '#FFA500', '#FFFF00', '#00FF00', '#0000FF', '#A020F0', '#FF00FF', '#00FFFF',
'#00008B', '#006400', '#DB7093', '#B03060', '#FFD700', '#FA8072',
'#B22222', '#DA70D6', '#ADFF2F', '#E6E6FA', '#FFE4B5', '#00CED1']

PASCAL_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'dog', 'chair', 'cow', 'dining table', 'cat', 'horse', 'motorbike', 'person', 'potted plant', 'sheep', 'sofa', 'train', 'tv monitor']

def rgb2gray(rgb):
	r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
	gray = (0.2989 * r) + (0.5870 * g) + (0.1140 * b)
	return np.asarray(np.dstack((gray, gray, gray)), dtype=np.uint8)

def main(argv):
	#IMG_NAME = '2007_000129' # bikes
	#IMG_NAME = '2007_000042' #	train
	#IMG_NAME = '2007_000346' # girl and bottle
	#IMG_NAME = '2007_009841' # horse and boy 
	IMG_NAME = '2007_000925' # 2 sheep

	img_path = IMAGES_DIR + IMG_NAME + '.jpg'
	gt_path = GT_DIR + IMG_NAME + '.png'
	obj_path = OBJECTNESS_DIR + IMG_NAME + '.png'
	clicks_path = CLICKS_DIR + IMG_NAME + '.png'

	# Original image
	orig_img = Image.open(img_path)
	img_arr = np.asarray(orig_img)
	Image.fromarray(img_arr).save('figures/original_img.png')

	# Labels
	gt = Image.open(gt_path)
	gt_arr = np.asarray(gt)

	labels_img = np.ones(img_arr.shape)
	labels_img.fill(200)
	labels_img = Image.fromarray(labels_img.astype(np.uint8))
	labels = np.unique(gt_arr).tolist()
	labels.remove(255)
	labels.remove(0)
	print labels
	classes = np.array(PASCAL_CLASSES)[labels]
	print classes
	num_classes = len(classes)
	draw = ImageDraw.Draw(labels_img)
	fontsize = 60
	font_path = "/usr/share/fonts/truetype/freefont/FreeSans.ttf"
	
	im_height = (labels_img.size)[1]
	coefficient = 2 if num_classes > 1 else 4 
	#y_offset = im_height - (2*fontsize * num_classes)
	y_offset = 10
	font = ImageFont.truetype(font_path, 60)
	for class_, k in zip(classes, labels):
		draw.text((20, y_offset), class_, hex_to_rgb(COLOR_SCHEME_HEX[k]), font=font) 
		y_offset += fontsize
	#labels_img = ImageOps.expand(labels_img, border=3, fill='black')
	labels_img.save('figures/labels.png')

	# Clicks
	points_arr = np.asarray(Image.open(clicks_path))
	clicks_im = np.zeros(img_arr.shape)
	clicks_im.fill(200)
	clicks_im = Image.fromarray(clicks_im.astype(np.uint8))
	draw = ImageDraw.Draw(clicks_im)
	sz = 8
	for r in range(img_arr.shape[0]):
		for c in range(img_arr.shape[1]):
			k = points_arr[r,c]
			if k != 255:
				draw.ellipse((c-sz, r-sz, c+sz, r+sz), fill=COLOR_SCHEME_NAMES[k], outline ='white')
	clicks_im.save('figures/clicks.png')
	
	# Objectness
	obj = Image.open(obj_path)
	obj_arr = np.asarray(obj)
	Image.fromarray(obj_arr).save('figures/objectness.png')

	# Fully supervised 
	fs_arr = np.zeros((gt_arr.shape[0], gt_arr.shape[1], 3))
	for r in range(img_arr.shape[0]):
		for c in range(img_arr.shape[1]):
			k = gt_arr[r,c]
			hex = 0
			if k == 255:
				hex = hex_to_rgb(COLOR_SCHEME_HEX[0])
			else:
				hex = hex_to_rgb(COLOR_SCHEME_HEX[k])				
			fs_arr[r,c,:] = hex
	fs_arr = fs_arr.astype(np.uint8)
	Image.fromarray(fs_arr).save('figures/GT.png')

def hex_to_rgb(value):
	value = value.lstrip('#')
	lv = len(value)
	return tuple(int(value[i:i+lv/3], 16) for i in range(0, lv, lv/3))

if __name__ == "__main__":
	main(sys.argv)
