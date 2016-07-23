caffe_root = '/imagenetdb3/abearman/caffe/'
import sys
sys.path.append(caffe_root + 'python')
print 'Importing Caffe ...'
import caffe
import numpy as np

# Load the original network and extract the fully connected layers' parameters
net = caffe.Net('/imagenetdb3/abearman/caffe/models/vgg16/deploy.prototxt',
				'/imagenetdb3/abearman/caffe/models/vgg16/VGG_ILSVRC_16_layers.caffemodel',
				caffe.TEST)
params = ['fc6', 'fc7', 'fc8']
# fc_params = {name: (weights, biases)}
fc_params = {pr: (net.params[pr][0].data, net.params[pr][1].data) for pr in params} 

for fc in params:
	print '{} weights are {} dimensional and biases are {} dimensional'.format(fc, fc_params[fc][0].shape, fc_params[fc][1].shape)

# Load the fully conv network to transplant the parameters
net_full_conv = caffe.Net('/imagenetdb3/abearman/caffe/models/vgg16-conv-pascal/deploy.prototxt',
						  '/imagenetdb3/abearman/caffe/models/vgg16/VGG_ILSVRC_16_layers.caffemodel',
						  caffe.TEST)

params_full_conv = ['fc6-conv', 'fc7-conv', 'score-fr']
# conv_params = {name: (weights, biases)}

print net_full_conv.params.keys()
print ""

conv_params = {pr: (net_full_conv.params[pr][0].data, net_full_conv.params[pr][1].data) for pr in params_full_conv}

for conv in params_full_conv:
	print '{} weights are {} dimensional and biases are {} dimensional'.format(conv, conv_params[conv][0].shape, conv_params[conv][1].shape)

row_idx = [405, 672, 13, 815, 908, 655, 610, 285, 560, 346, 533, 235, 340, 666, 982, 739, 349, 832, 467, 665]
#row_idx = [230, 255, 388, 237, 831, 920, 269, 95, 309, 108, 315, 64, 39, 277, 954, 838, 81, 311, 887, 869] 
row_idx = [x - 1 for x in row_idx]
print row_idx 

# Transplant weights!
for pr, pr_conv in zip(params, params_full_conv):
	# We do copy the last layer, but only the correct 21 indices
	if pr == 'fc8':
		print fc_params[pr][0].shape # weights
		fc_weights = fc_params[pr][0]
		fc_biases = fc_params[pr][1]
		subset_fc_weights = fc_weights[np.array(row_idx),:] # 20 weights
		subset_fc_biases = fc_biases[np.array(row_idx)] # 20 biases

		# Zero-initialize the background class = 0
		subset_fc_weights = np.insert(subset_fc_weights, 0, np.zeros((1, subset_fc_weights.shape[1])), 0)  	
		subset_fc_biases = np.insert(subset_fc_biases, 0, np.zeros(1,), 0)

		conv_params[pr_conv][0].flat = subset_fc_weights.flat
		conv_params[pr_conv][1][...] = subset_fc_biases
	else:
		conv_params[pr_conv][0].flat = fc_params[pr][0].flat # flat unroll the arrays
		conv_params[pr_conv][1][...] = fc_params[pr][1]

# Save the new model weights
net_full_conv.save('/imagenetdb3/abearman/caffe/models/vgg16-conv-pascal/vgg16-conv-pascal.caffemodel')

