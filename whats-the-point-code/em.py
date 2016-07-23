# Train the model once over the set of training images, getting a probability for every pixel
# For every pixel, if class has probability > 0.5 (or something), set this pixel to have that class label. (Will need to make a new lmdb). Otherwise, leave it unlabelled. 
# Retrain. Until convergence.

#python /imagenetdb3/abearman/caffe/models/code/solve.py --year=2012 --output=real1-click1-cls-con-obj --lr=1e-5 --train-gt=/imagenetdb3/olga/data//segm_lmdb/lmdb-pascal_2012t_SBDtv_minus_2012v/lmdb-real1_click1-gt3 --expectation --location --objectness --constraint --classes --gpu=0 --display=1

import sys
from PIL import Image
import numpy as np
import shutil
import os.path
import os
import fileinput
sys.path.append('/imagenetdb3/abearman/caffe/python/')
print 'Importing caffe ...'
import caffe
import subprocess
import constants


YEAR = "2012"
OUTPUT_DIR = "real1-click1-cls-con-obj-EM"
LR = "1e-5"
BATCH_SIZE = 20
TRAIN_GT_LMDB_NAME = "lmdb-real1_click1-gt3"
INITIAL_TRAIN_GT_LMDB = "/imagenetdb3/olga/data/segm_lmdb/lmdb-pascal_2012t_SBDtv_minus_2012v/" + TRAIN_GT_LMDB_NAME 
GPU = 2
THRESHOLD = 0.5

MODELS_DIR = "/imagenetdb3/abearman/caffe/models/"
BASE_MODEL = MODELS_DIR + "vgg16-conv-pascal/vgg16-conv-pascal.caffemodel" 
SPECIFIC_MODEL_DIR = MODELS_DIR + "fcn-32s-pascal/" + YEAR + "/" + OUTPUT_DIR + "/"
TEMP_IMAGES_DIR = SPECIFIC_MODEL_DIR + "temp_images/"
TEMP_LMDB_DIR = SPECIFIC_MODEL_DIR + "temp_lmdbs/" + TRAIN_GT_LMDB_NAME
VIZ_IMAGES_DIR = SPECIFIC_MODEL_DIR + "visualizations/" 
IMG_DIR_VOC = "/imagenetdb3/olga/data/VOCdevkit/VOC2012/JPEGImages/"
IMG_DIR_SBD = "/imagenetdb3/olga/data/PASCAL_SBD/dataset/img/"
TRAIN_IMG_IDS = "/imagenetdb3/olga/data/segm_lmdb/pascal_2012t_SBDtv_minus_2012v.txt"
AMT_CLICKS_DIR = "/imagenetdb3/olga/data/PASCAL_AMT_clicks/real1_click1/"

# Booleans
copy_lmdbs = True
copy_images = True

def main():
	set_up()
	update_train_val()
	update_deploy()
	update_solver_prototxt()
	solver = set_up_net()
	print "Finished setup"

	counter = 0
	while True:
		model_name, counter = train_net(solver, 12800, counter)
		#model_name = "real1-click1-cls-con-obj_step1600.caffemodel"
		net = caffe.Net(SPECIFIC_MODEL_DIR + "deploy.prototxt", SPECIFIC_MODEL_DIR+ "iters/" + model_name, caffe.TEST) 
		#net = caffe.Net(SPECIFIC_MODEL_DIR + "deploy.prototxt", MODELS_DIR + "fcn-32s-pascal/2012/real1-click1-cls-con-obj/iters/" + model_name, caffe.TEST)
		assign_high_scoring_pixels(net)


def set_up():
	if not os.path.exists(SPECIFIC_MODEL_DIR):
		os.makedirs(SPECIFIC_MODEL_DIR)
	if not os.path.exists(TEMP_IMAGES_DIR):
		os.makedirs(TEMP_IMAGES_DIR)
	if not os.path.exists(VIZ_IMAGES_DIR):
		os.makedirs(VIZ_IMAGES_DIR)

	# Copy in the old initial lmdb files
	if copy_lmdbs:
		if os.path.exists(TEMP_LMDB_DIR):
			shutil.rmtree(TEMP_LMDB_DIR)
		print "Copying initial lmdb ..."
		shutil.copytree(INITIAL_TRAIN_GT_LMDB, TEMP_LMDB_DIR)

	if copy_images:
		copy_over_original_images()	

	caffe.set_mode_gpu()
	caffe.set_device(GPU)


def get_list_of_image_paths():
	img_paths = []
	with open(TRAIN_IMG_IDS) as ids_file:
		for img_id in ids_file:
			img_id = img_id[:-1]	# Strips newlines
			img_path_voc = IMG_DIR_VOC + img_id + ".jpg"
			img_path_sbd = IMG_DIR_SBD + img_id + ".jpg"
			if os.path.isfile(img_path_voc):	# is VOC
				 img_paths.append(img_path_voc)
			elif os.path.isfile(img_path_sbd):	# is SBD
				img_paths.append(img_path_sbd)
			else:
				raise Exception("Couldn't find corresponding image for ID: " + img_id)
	return img_paths


def copy_over_original_images():
	# Only copies over the training images, not all annotated (train + val) images
	with open(TRAIN_IMG_IDS) as ids_file:
		counter = 1
		for img_id in ids_file: 
			print counter
			counter += 1
			img_id = img_id[:-1]	# Strips the newlines
			shutil.copy2(AMT_CLICKS_DIR + img_id + ".png", TEMP_IMAGES_DIR + img_id + ".png")


def assign_high_scoring_pixels(net):	
	for image_path in get_list_of_image_paths():	
		annot_arr, img_id = predict_image_segmentation(net, image_path)
		visualize_image(annot_arr, img_id)
	print "Making lmdb now ..."
	subprocess.call(["python", os.path.abspath("make_lmdb_three_channel.py"), "/imagenetdb3/olga/data/segm_lmdb/pascal_2012t_SBDtv_minus_2012v.txt", TEMP_IMAGES_DIR, "/imagenetdb3/olga/data/PASCAL_objectness", TEMP_LMDB_DIR]) 

def predict_image_segmentation(net, image_path):
	# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
	print image_path
	im = np.array(Image.open(image_path), dtype=np.float32)
	im = im[:,:,::-1]
	im -= np.array((104.00698793, 116.66876762, 122.67891434))
	im = im.transpose((2,0,1))

	net.blobs['data'].reshape(1, *im.shape)
	net.blobs['data'].data[...] = im

	img_id = os.path.splitext(os.path.basename(image_path))[0]
	annot_image_path = TEMP_IMAGES_DIR + img_id + ".png"
	annot_arr = np.array(Image.open(annot_image_path))

	net.forward()
	out = net.blobs['upscore'].data[0]
	(max_probs, max_classes) = compute_softmax_probability(out)
	print np.unique(annot_arr)	
	annot_arr[max_probs > THRESHOLD] = max_classes[max_probs > THRESHOLD]
	print np.unique(annot_arr)
	Image.fromarray(annot_arr).save(annot_image_path)
	return annot_arr, img_id


def visualize_image(out, img_id):
	labels = np.unique(out).tolist()
	print "labels: " + str(labels)
	output = np.zeros((out.shape[0], out.shape[1], 3))
	for k in labels:
		i, j = np.where(out == k)
		if k != 255:
			output[i, j] = constants.hex_to_rgb(constants.COLOR_SCHEME_HEX[k])
		else:
			output[i, j] = constants.hex_to_rgb("#FFFFFF")  # White for 255 label
	output = output.astype(np.uint8)
	Image.fromarray(output).save(VIZ_IMAGES_DIR + img_id + '.png')

def compute_softmax_probability(scores):
	sumsOverClasses = np.sum(np.exp(scores), axis=0)
	max_probs = np.max((np.exp(scores) / sumsOverClasses), axis=0)
	max_classes = np.argmax((np.exp(scores) / sumsOverClasses), axis=0)
	return (max_probs, max_classes)
	#return [( np.exp(score_outer) / sum(np.exp(score_inner) for score_inner in scores) ) for score_outer in scores]	


def update_train_val():
	dest_train_val = SPECIFIC_MODEL_DIR + 'train_val.prototxt'
	src_train_val = MODELS_DIR + "fcn-32s-pascal/train_val.prototxt"
	print "src: " + src_train_val
	print "dest: " + dest_train_val
	shutil.copy2(src_train_val, dest_train_val)
	for line in fileinput.FileInput(dest_train_val, inplace=1):
		line = line.replace('/imagenetdb3/olga/data/segm_lmdb/lmdb-pascal_2012t_SBDtv_minus_2012v/lmdb-fs-gt1', TEMP_LMDB_DIR)
		line = line.replace('SoftmaxWithLoss', 'SoftmaxWithLossExpectation')
		line = line.replace('normalize: false', 'normalize: false\nignore_objectness: false')
		line = line.replace('normalize: false', 'normalize: false\nignore_location: false')
		line = line.replace('normalize: false', 'normalize: false\nignore_constraint: false')
		line = line.replace('normalize: false', 'normalize: false\nignore_classes: false')
		print line,
		

def update_deploy():
	filename = SPECIFIC_MODEL_DIR + 'deploy.prototxt'
	shutil.copyfile(MODELS_DIR + "fcn-32s-pascal/" + 'deploy.prototxt', filename)
	for line in fileinput.FileInput(filename, inplace=1):
		print line,

def update_solver_prototxt():
	filename =	SPECIFIC_MODEL_DIR + 'solver.prototxt'
	shutil.copyfile(MODELS_DIR + "fcn-32s-pascal/" + 'solver.prototxt', filename)
	for line in fileinput.FileInput(filename, inplace=1):
		if 'net:' in line:
			line = 'net: "' + SPECIFIC_MODEL_DIR + 'train_val.prototxt"'
		if 'base_lr:' in line:
			line = 'base_lr: ' + LR
		if 'display:' in line:
			line = 'display: 1'
		if 'momentum:' in line:
			line = 'momentum: 0.99' 
		print line,


def set_up_net():
	SOLVER_FILE = SPECIFIC_MODEL_DIR + 'solver.prototxt'
	solver = caffe.SGDSolver(SOLVER_FILE)

	# copy base weights for fine-tuning
	print 'Loading from base model'
	solver.net.copy_from(BASE_MODEL)
	print "Finished loading from base model"

	# do net surgery to set the deconvolution weights for bilinear interpolation
	interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
	interp_surgery(solver.net, interp_layers)
	print "Finished net surgery"

	# save initial model
	NAME = OUTPUT_DIR
	solver.net.save(SPECIFIC_MODEL_DIR + 'iters/' + NAME + '_step0.caffemodel')
	print "Saved model"

	return solver


def train_net(solver, num_iters, counter):
	NAME = OUTPUT_DIR

	num_steps = (int) (num_iters / BATCH_SIZE)
	for i in range(num_steps):
		counter += BATCH_SIZE
		solver.step(BATCH_SIZE)
		print check_for_nan(solver.net)
		if counter % 1000 == 0:  # Save every 1,000 iterations
			solver.net.save(SPECIFIC_MODEL_DIR + 'iters/' + NAME + '_step' + str(counter) + '.caffemodel')
 
	model_name = NAME + '_step' + str(counter) + '.caffemodel' 
	solver.net.save(SPECIFIC_MODEL_DIR + 'iters/' + model_name)
	return model_name, counter 

def check_for_nan(net):
	bQuit = False
	sums = [];
	for name in net.params:
		layer = net.params[name];
		for i in range(min(2,len(layer))):
			weights = layer[i].data
			s = np.mean(np.abs(weights))
			sums += [s]
			n = np.sum(np.isnan(weights))
			if n > 0:
				print ' NAN',name,i,n
				bQuit = True
	print sums
	if bQuit:
		exit(-1)


# set parameters s.t. deconvolutional layers compute bilinear interpolation
# N.B. this is for deconvolution without groups
def interp_surgery(net, layers):
				for l in layers:
								m, k, h, w = net.params[l][0].data.shape
								if m != k:
												print 'input + output channels need to be the same'
												raise
								if h != w:
												print 'filters need to be square'
												raise
								filt = upsample_filt(h)
								net.params[l][0].data[range(m), range(k), :, :] = filt

# make a bilinear interpolation kernel
# credit @longjon
def upsample_filt(size):
				factor = (size + 1) // 2
				if size % 2 == 1:
								center = factor - 1
				else:
								center = factor - 0.5
				og = np.ogrid[:size, :size]
				return (1 - abs(og[0] - center) / factor) * \
									 (1 - abs(og[1] - center) / factor)

if __name__ == "__main__": main()
