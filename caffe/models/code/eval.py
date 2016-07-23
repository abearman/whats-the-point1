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
import constants


is_pascal = True
is_test = False
is_numbers = True
is_visualize = True 
is_color = True


IMAGES_DIR = GT_DIR = VAL_DIR = '' 
if is_pascal:
	if is_test:
		IMAGES_DIR = '/imagenetdb3/olga/data/VOCdevkit/VOC2012/JPEGImagesTest/'
		VAL_FILE = '/imagenetdb3/olga/data/VOCdevkit/VOC2012/ImageSets/Segmentation/test.txt'
	else:
		IMAGES_DIR = '/imagenetdb3/olga/data/VOCdevkit/VOC2012/JPEGImages/'
		VAL_FILE = '/imagenetdb3/olga/data/segm_lmdb/pascal_2012v.txt'
	GT_DIR = '/imagenetdb3/olga/data/VOCdevkit/VOC2012/SegmentationClass/'
else:
	IMAGES_DIR = '/imagenetdb3/olga/data/SIFTflow/Images/spatial_envelope_256x256_static_8outdoorcategories/'
	GT_DIR = '/imagenetdb3/olga/data/SIFTflow/SegmentationClass/spatial_envelope_256x256_static_8outdoorcategories/'
	VAL_FILE = '/imagenetdb3/olga/data/SIFTflow/val.txt'

print IMAGES_DIR
print GT_DIR
print VAL_FILE

MODELS_DIR = '/imagenetdb3/abearman/caffe/models/fcn-32s-pascal/'

DIR = '2012/image-level-labels-proportional-half/'
MODEL = 'image-level-labels-proportional-half_step6000.caffemodel'
#DIR = '2012/bbox-cls-con-proportional/'
#MODEL = 'bbox-cls-con-proportional_step33000.caffemodel'

#DIR = '2012/100-fs-rest-points/'
#MODEL = '100-fs-rest-points_step34000.caffemodel'

PROTOTEXT_FILE = MODELS_DIR + DIR + 'deploy.prototxt'
CAFFE_MODEL = MODELS_DIR + DIR + 'iters/' + MODEL
OUTPUT_DIR = MODELS_DIR + DIR 

NUM_CLASSES = 21 if is_pascal else 33

def main(argv):
	sys.path.append('/imagenetdb3/abearman/caffe/python/')
	print "Importing caffe ..."
	import caffe

	val_file = open(VAL_FILE)
	ids = [line[:-1] for line in val_file]

	overall_acc = 0
	accuracies = []

	# Load net
	net = caffe.Net(PROTOTEXT_FILE, CAFFE_MODEL, caffe.TEST)
	caffe.set_mode_gpu()
	caffe.set_device(0)

	mean_per_class_iou = [0.0] * NUM_CLASSES
	num_intersection = [0.0] * NUM_CLASSES
	num_union = [0.0] * NUM_CLASSES

	start_t = time.time();
	time_segm = 0;
	time_other = 0;
	for i in range(len(ids)):
		if (time.time() - start_t) > 10:
			print "iter:", i,"/",len(ids),"time_segm:",time_segm,"time_other:",time_other
			if is_numbers:
				print "Current mean per class: ", current_iou
				print "Current overall mean: ", mean(current_iou)
				print "Current overall mean ignore background: ", mean(current_iou[1:])
				print ""
			start_t = time.time()
			
		tt = time.time();
		image_path = IMAGES_DIR + ids[i] + '.jpg'
		gt_path = GT_DIR + ids[i] + '.png'		 
		predicted_segmentation = segment(image_path, net, is_visualize)	
		time_segm += time.time()-tt;

		tt = time.time()
		if is_numbers:
			(num_intersection,num_union) = computeIOU(predicted_segmentation,gt_path,num_intersection,num_union);
			current_iou = [float(num_intersection[i])/num_union[i] if num_union[i] > 0 else 0 for i in range(len(num_union))];

		time_other += time.time()-tt
		
	if is_numbers:
		print "Mean iou per class: ", current_iou
		print "Mean iou: ", mean(current_iou)
		print "Mean iou without background: ", mean(current_iou[1:])

def computeIOU(predicted, gt_path,num_intersection,num_union): 
	# Count number of class appearances
	#class_appearances = [0] * 21
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
		num_intersection[class_] += ni;
		num_union[class_] += nu;

	ious = [-1] * 21
	for i in range(len(num_union)):
		iou_for_class = -1	# just initializing 
		if num_union[i] > 0:
			iou_for_class = float(num_intersection[i]) / num_union[i]
		# set it to -1 if the class does not appear in the image, otherwise leave it as 0.0
		if i not in gt_:
			iou_for_class = -1
		ious[i] = iou_for_class
	#print "list: " + str(ious)
	
	return (num_intersection,num_union)

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

def segment(image_path, net, visualize):
	# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
	im = Image.open(image_path)
	(width, height) = im.size

	#im = im.resize((500, 500), Image.NEAREST)
#	if width < DIMENSION_MIN:
#		wpercent = DIMENSION_MIN / float(width)
#		hsize = int((float(height) * float(wpercent)))
#		im = im.resize((DIMENSION_MIN, hsize), PIL.Image.ANTIALIAS)
#	if height < DIMENSION_MIN:
#		hpercent = DIMENSION_MIN / float(height)
#
#		wsize = int((float(width) * float(hpercent)))
#		im = im.resize((wsize, DIMENSION_MIN), PIL.Image.ANTIALIAS)	
	#print "input image size: ", im.size


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
	#import scipy.io
	#scipy.io.savemat('out.mat', {'out': out})

	if visualize: visualize_image(out, image_path, im)	
	return out

# segmentation results, image path, original image
def visualize_image(out, image_path, im):		
	basename = os.path.splitext(os.path.basename(image_path))[0]

	if is_color:
		labels = np.unique(out).tolist()
		print labels

		output = np.zeros((out.shape[0], out.shape[1], 3))
		for k in labels:
			i, j = np.where(out == k)
			if is_color: 
				output[i, j] = hex_to_rgb(constants.COLOR_SCHEME_HEX[k])
			else:
				output[i, j] = k
		output = output.astype(np.uint8)

		#cb = colorbar_index(ncolors=len(classes), cmap=cmap, shrink=0.5, labels=classes)
		#cb.ax.tick_params(labelsize=12)

		if not os.path.exists(OUTPUT_DIR + '/images'):
			os.makedirs(OUTPUT_DIR + '/images')
		im.save(OUTPUT_DIR + '/images/' + basename + '_in.png')
		Image.fromarray(output).save(OUTPUT_DIR + '/images/' + basename + '_out.png')
	else:	
		out = out.astype(np.uint8)
		if not os.path.exists(OUTPUT_DIR + '/test-images'):
			os.makedirs(OUTPUT_DIR + '/test-images')
		Image.fromarray(out).save(OUTPUT_DIR + '/test-images/' + basename + '.png')

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

def hex_to_rgb(value):
	value = value.lstrip('#')
	lv = len(value)
	return tuple(int(value[i:i+lv/3], 16) for i in range(0, lv, lv/3))

if __name__ == "__main__":
	main(sys.argv)
