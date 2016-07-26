from optparse import OptionParser
import shutil
import fileinput
import os
import sys
import numpy as np

MODELS_DIR = '../../caffe/models/fcn-32s-pascal/'

def main():
	options = set_up_parser()
	global BASE_MODEL
	BASE_MODEL = '../../caffe/models/vgg16-conv-pascal/vgg16-conv-pascal.caffemodel'
	
	update_train_val(options)
	update_deploy(options)
	update_solver_prototxt(options)
	solver = set_up_net(options)
	train_net(solver, options)

def train_net(solver, options):
	NAME = options.output_dir
	N = int(options.batch_size);
	
	if options.start_iter == '':
		iter = 0
	else: 
		iter = int(options.start_iter)
	#save_every = 5;
	save_every = 1000;
	while True:
		#iter += 1
		iter += N
		solver.step(N)
		print check_for_nan(solver.net)
		if (iter % save_every == 0): # only save every 1000
		#if iter % save_every == 0: # only save every 1000
			solver.net.save(MODELS_DIR + options.year + '/' + options.output_dir + '/iters/' + NAME + '_step' + str(iter) + '.caffemodel')
			#save_every *= 2

def set_up_net(options):
	sys.path.append('../../caffe/python/')
	print 'Importing caffe ...'
	import caffe

	caffe.set_mode_gpu()
	caffe.set_device(int(options.gpu))

	SOLVER_FILE = MODELS_DIR + options.year + '/' + options.output_dir + '/solver.prototxt'
	solver = caffe.SGDSolver(SOLVER_FILE)

	# copy base weights for fine-tuning
	print 'Loading from base model'
	init = BASE_MODEL if options.init_from == '' else options.init_from
	print init 
	solver.net.copy_from(init)
	print "Loaded base model"
	
	# do net surgery to set the deconvolution weights for bilinear interpolation
	interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
	interp_surgery(solver.net, interp_layers)
	print "Did net surgery"

	# save initial model
	NAME = options.output_dir
	print "name of model: ", MODELS_DIR + options.year + '/' + options.output_dir + '/iters/' + NAME + '_step0.caffemodel'
	solver.net.save(MODELS_DIR + options.year + '/' + options.output_dir + '/iters/' + NAME + '_step0.caffemodel') 
	print "Saved initial model"

	return solver

def set_up_parser():
	parser = OptionParser()
	parser.add_option("--output", dest="output_dir", help="Where do you want the output to go?")
	parser.add_option("--year", dest="year", default="2012")
	
	parser.add_option("--train-img", dest="train_img", default="/imagenetdb3/olga/data/segm_lmdb/lmdb-pascal_2012t_SBDtv_minus_2012v/lmdb-img")
	parser.add_option("--train-gt", dest="train_gt", default="/imagenetdb3/olga/data/segm_lmdb/lmdb-pascal_2012t_SBDtv_minus_2012v/lmdb-fs-gt1")
	parser.add_option("--val-img", dest="val_img", default="lmdb-pascal_2012v-img")
	parser.add_option("--val-gt", dest="val_gt", default="lmdb-pascal_2012v-gt")
	parser.add_option("--test", action="store_true", dest="test", default=False) 
	
	parser.add_option("--lr", dest="lr", default="1e-10")
	parser.add_option("--momentum", dest="momentum", default=0.99)
	parser.add_option("--gpu", dest="gpu", default=3)
	parser.add_option("--display", dest="display", default=20);
	parser.add_option("--batch-size", dest="batch_size", default=20)
	parser.add_option("--init-from", dest="init_from", default='')

	parser.add_option("--expectation", action="store_true", dest="expectation", default=False)
	parser.add_option("--location", action="store_true", dest="location", default=False)
	parser.add_option("--objectness", action="store_true", dest="objectness", default=False)
	parser.add_option("--constraint", action="store_true", dest="constraint", default=False)
	parser.add_option("--rank", action="store_true", dest="rank", default=False)
	parser.add_option("--classes", action="store_true", dest="classes", default=False)
	parser.add_option("--siftflow", action="store_true", dest="siftflow", default=False)

	parser.add_option("--no_norm_sup", action="store_true", dest="no_norm_sup", default=False)
	parser.add_option("--no_norm_cls", action="store_true", dest="no_norm_cls", default=False)
	parser.add_option("--no_norm_con", action="store_true", dest="no_norm_con", default=False)
	parser.add_option("--no_norm_obj", action="store_true", dest="no_norm_obj", default=False)

	parser.add_option("--eight_stride", action="store_true", dest="eight_stride", default=False)
	parser.add_option("--start-iter", dest="start_iter", default='')

	(options, args) = parser.parse_args()
	print options
	if not options.output_dir:	 # if filename is not given
		parser.error('Output directory not given')
	if not options.year: # if year not given
		parser.error('Pascal year not given')

	mini_path = MODELS_DIR + options.year + '/' + options.output_dir
	print "mini_path: ", mini_path
	# Makes the new output directory if it doesn't exist
	if not os.path.exists(mini_path):
		os.makedirs(mini_path)
	if not os.path.exists(mini_path + "/iters"):
		os.makedirs(mini_path + "/iters")
	return options

def update_solver_prototxt(options):
	filename =	MODELS_DIR + options.year + '/' + options.output_dir + '/solver.prototxt'
	shutil.copyfile(MODELS_DIR + 'solver.prototxt', filename)
	for line in fileinput.FileInput(filename, inplace=1):
		if 'net:' in line:
			line = 'net: "' + MODELS_DIR + options.year + '/' + options.output_dir + '/train_val.prototxt"'
		if 'base_lr:' in line:
			line = 'base_lr: ' + options.lr
		if 'display:' in line:
			line = 'display: ' + str(options.display)
		if 'momentum::' in line:
						line = 'momentum: ' + int(options.momentum)
		print line,	

def update_deploy(options):
	filename = MODELS_DIR + options.year + '/' + options.output_dir + '/deploy.prototxt'
	shutil.copyfile(MODELS_DIR + 'deploy.prototxt', filename)
	for line in fileinput.FileInput(filename, inplace=1):
		if options.siftflow:
			line = line.replace('num_output: 21', 'num_output: 33')
		print line,
 
def update_train_val(options):
	filename = MODELS_DIR + options.year + '/' + options.output_dir + '/train_val.prototxt'
	print "MODELS_DIR: " + MODELS_DIR
	shutil.copyfile(MODELS_DIR + 'train_val.prototxt', filename) 
	for line in fileinput.FileInput(filename, inplace=1):
		if options.test:
			line = line.replace('lmdb-pascal_2012t_SBDtv_minus_2012v/lmdb-img',
								'lmdb-pascal_2012tv_SBDtv/lmdb-pascal_2012tv_SBDtv-img')
			line = line.replace('/imagenetdb3/olga/data/segm_lmdb/lmdb-pascal_2012t_SBDtv_minus_2012v/lmdb-fs-gt1',
								options.train_gt)
			line = line.replace('lmdb-pascal_2012v/lmdb-pascal_2012v-img',
								'lmdb-pascal_2012tv_SBDtv/lmdb-pascal_2012tv_SBDtv-img')
			line = line.replace('lmdb-pascal_2012v/lmdb-pascal_2012v-gt',
								'lmdb-pascal_2012tv_SBDtv/' + options.val_gt)
		elif options.siftflow:
			line = line.replace('lmdb-pascal_2012t_SBDtv_minus_2012v/lmdb-img', 
								'lmdb-siftflow_train/' + options.train_img)
			line = line.replace('lmdb-pascal_2012t_SBDtv_minus_2012v/lmdb-fs-gt1', 
								'lmdb-siftflow_train/' + options.train_gt)
			line = line.replace('lmdb-pascal_2012v/lmdb-pascal_2012v-img',
								'lmdb-siftflow_val/lmdb-img')
			line = line.replace('lmdb-pascal_2012v/lmdb-pascal_2012v-gt', 
								'lmdb-siftflow_val/' + options.val_gt)
			line = line.replace('num_output: 21', 'num_output: 33')

		elif options.year == "2012":
			line = line.replace('/imagenetdb3/olga/data/segm_lmdb/lmdb-pascal_2012t_SBDtv_minus_2012v/lmdb-img', options.train_img)
			line = line.replace('/imagenetdb3/olga/data/segm_lmdb/lmdb-pascal_2012t_SBDtv_minus_2012v/lmdb-fs-gt1', options.train_gt)
			line = line.replace('lmdb-pascal_2012v-img', options.val_img)
			line = line.replace('lmdb-pascal_2012v-gt', options.val_gt)
		elif options.year == "2011":
			line = line.replace('lmdb-pascal_2012t_SBDtv_minus_2012v/lmdb-img', 
								'lmdb-pascal_2011t/lmdb-pascal_2011t-img')
			line = line.replace('lmdb-pascal_2012t_SBDtv_minus_2012v/lmdb-fs-gt1', 
								'lmdb-pascal_2011t/' + options.train_gt)
			line = line.replace('lmdb-pascal_2012v/lmdb-pascal_2012v-img', 
								'lmdb-pascal_2011v/lmdb-pascal_2011v-img')
			line = line.replace('lmdb-pascal_2012v/lmdb-pascal_2012v-gt', 
								'lmdb-pascal_2011v/lmdb-pascal_2011v-gt')	

		if options.expectation:
			line = line.replace('SoftmaxWithLoss', 'SoftmaxWithLossExpectation')

			if options.objectness: # Don't ignore objectness
				line = line.replace('normalize: false', 'normalize: false\nignore_objectness: false')
			else: # Ignore objectness
				line = line.replace('normalize: false', 'normalize: false\nignore_objectness: true')

			if options.location: # Don't ignore location
				line = line.replace('normalize: false', 'normalize: false\nignore_location: false')
			else: # Ignore location
				line = line.replace('normalize: false', 'normalize: false\nignore_location: true')

			if options.constraint: # Don't ignore constraint
				line = line.replace('normalize: false', 'normalize: false\nignore_constraint: false')
			else: # Ignore constraint
				line = line.replace('normalize: false', 'normalize: false\nignore_constraint: true')

			if options.classes: # Don't ignore classes
				line = line.replace('normalize: false', 'normalize: false\nignore_classes: false')
			else: # Ignore classes
				line = line.replace('normalize: false', 'normalize: false\nignore_classes: true')

			if options.rank: # Don't ignore rank
				line = line.replace('normalize: false', 'normalize: false\nignore_rank: false')	
			else: 
				line = line.replace('normalize: false', 'normalize: false\nignore_rank: true')		

			if options.no_norm_sup: # Don't normalize SUPERVISED term 
				 line = line.replace('normalize: false', 'normalize: false\nnormalize_supervised: false')
			else:
				 line = line.replace('normalize: false', 'normalize: false\nnormalize_supervised: true')

			if options.no_norm_cls: # Don't normalize CLS term	
				line = line.replace('normalize: false', 'normalize: false\nnormalize_classes: false')
			else:
				 line = line.replace('normalize: false', 'normalize: false\nnormalize_classes: true')

			if options.no_norm_con: # Don't normalize CON term 
				 line = line.replace('normalize: false', 'normalize: false\nnormalize_constraint: false')
			else:
				 line = line.replace('normalize: false', 'normalize: false\nnormalize_constraint: true')

			if options.no_norm_obj: # Don't normalize OBJ term 
				 line = line.replace('normalize: false', 'normalize: false\nnormalize_objectness: false')
			else:
				 line = line.replace('normalize: false', 'normalize: false\nnormalize_objectness: true')

			if options.siftflow: # Use SiftFlow data 
				line = line.replace('normalize: false', 'normalize: false\nis_siftflow: true')
			else:
				 line = line.replace('normalize: false', 'normalize: false\nis_siftflow: false')

		else: # Normal softmax loss layer
			if options.siftflow: 
				line = line.replace('ignore_label:255', 'ignore_label:0')
				line = line.replace('normalize: false', 'normalize: false\nis_siftflow: true')
			else: 
				line = line.replace('normalize: false', 'normalize: false\nis_siftflow: false')
	
		print line,

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


if __name__ == '__main__':main()

