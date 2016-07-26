# Files to Run and Evaluate Models:

## ``solve.py``

This script trains a fully-connected, 32 pixel stride convolutional Caffe network for the PASCAL VOC 2012 segmentation task. It uses our [custom Caffe softmax loss layer](https://github.com/abearman/whats-the-point1/blob/454f0b04d8875349d287801d1041aa9820fe7f50/caffe/src/caffe/layers/softmax_loss_expectation_layer.cu) to do semantic segmentation using the information provided by image-level labels, point supervision, and an objectness prior. 

By default, this script runs Caffe in GPU mode. You can change this by changing the line ``caffe.set_mode_gpu()`` to ``caffe.set_mode_cpu()``. 

By default, the new model is initialized from [this model](https://github.com/abearman/whats-the-point1/tree/master/caffe/models/vgg16-conv-pascal), but you can optionally specify a different model to intialize from (e.g., if you want to suspend and resume training).
 
The script saves a model every 1,000 iterations.

### Inputs


## ``eval.py``

This file evaluates a Caffe model on the PASCAL VOC segmentation dataset. At its completion, it will print the per-class and overall mean intersection-over-union achieved by the model on PASCAL VOC, as compared to the ground-truth segmentations. It will also visualize the segmentation masks predicted by the given model and save them as png images.

### Inputs: 

All inputs are specified as constants at the top of the file. 

#### Program arguments

* **is_pascal**: specifies whether or not to run on the PASCAL VOC dataset or another dataset (we have also experimented with SIFTFLOW). Should probably always be set to ``True``. 
* **is_test**: specifies whether or not run on the PASCAL VOC validation set or test set. 
* **is_numbers**: specifies whether or not to evaluate the mIOU performance of the given model. (For example, ``is_numbers=False`` and ``is_visualize=True``, the script will just visualize the predicted segmentation masks). 
* **is_visualize**: specifies whether or not to also visualize the segmentation masks predicted by the given model and save them as png images.
* **is_color**: specifies whether to visualize the predicted segmentation masks in human-visible color masks, or with their class labels. If ``is_color=True``, the script will visualize the segmentation masks with different RGB colors. If ``is_color=False``, the script will visualize the segmentation masks with the PASCAL class labels (0-20). 

#### File paths

* **MODELS_DIR**: specifies the general directory containing all models. In our case, this is ``../../caffe/models/fcn-32s-pascal/``
* **DIR**: specifies the specific model directory containing the saved .caffemodel files, visualized segmentations, and .prototxt files. It is concatenated onto the ``MODELS_DIR`` path. An example of one ``DIR`` is ``2012/real1-squiggle1-cls-con-obj``. 


