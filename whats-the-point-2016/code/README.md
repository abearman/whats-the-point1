# Files to Run and Evaluate Models:

## ``solve.py``

This script trains a fully-connected, 32 pixel stride convolutional Caffe network for the PASCAL VOC 2012 segmentation task. It uses our [custom Caffe softmax loss layer](https://github.com/abearman/whats-the-point1/blob/454f0b04d8875349d287801d1041aa9820fe7f50/caffe/src/caffe/layers/softmax_loss_expectation_layer.cu) to do semantic segmentation using the information provided by image-level labels, point supervision, and an objectness prior. 

This script creates a new output directory based on the path you give it. In this directory, the script creates a new ``train_val.prototxt``, ``solver.prototxt``, and ``deploy.prototxt`` based on the arguments your provide. It also saves all models here (one snapshot every 1,000 iterations). 

By default, this script runs Caffe in GPU mode. You can change this by changing the line ``caffe.set_mode_gpu()`` to ``caffe.set_mode_cpu()``. 

By default, the new model is initialized from [this model](https://github.com/abearman/whats-the-point1/tree/master/caffe/models/vgg16-conv-pascal), but you can optionally specify a different model to intialize from (e.g., if you want to suspend and resume training).
 
### Inputs
You can see all inputs and their defaults in the ``set_up_parser`` function.

* **output**: A name for the model output directory (not a full path), e.g. "real1-click1-cls-con-obj." No default, must be specified.
* **year**: The year of the PASCAL VOC challenge. Default: 2012.
* **train-img**: The path to the lmdb containing the training jpg images.
* **train-gt**: The path to the lmdb containing the ground truth segmentations (with the correct class labels) for the set of training images.
* **val-img**: The path to the lmdb containing the validation jpg images.
* **val-gt**: The path to the lmdb containing the ground truth segmentations (with the correct class labels) for the set of validation images. 
* **test**: If true, train the model using all of the training and validation images, because it will be evaluated on the test set. Default: false.
* **lr**: The learning rate hyperparameter. Default: 1e-5.
* **momentum**: The momentum hyperparameter. Default: 0.9
* **gpu**: Which GPU ID to use, if on GPU mode. Default: 0.
* **display**: Print output every <display> iterations. Default: 20.
* **batch-size**: Size of minibatch of images. Default: 20.
* **init-from**: If specified, initialize the model from the .caffemodel at <init-from> path. Otherwise, the VGG16-CONV model will be used. Default: None. 
* **expectation**: Whether or not to use our custom loss function. Default: false.
* **location**: The "supervised" term; if true, use point supervision (i.e., the location information provided by human points). Default: false. 
* **constraint**: The "constraint" term; if true, use the *absence* of a class label as a signal. Default: false.
* **rank**: Whether or not to rank multiple human points by the order in which they were pointed to. This is only relevant if you're using more than one point per object class (which we did not do for our main experiment). Default: false.  
* **classes**: The "classes" term; if true, use the *presence* of a class label as a signal. Default: false.
* **no-norm-sup**: If true, do not normalize the supervised term. Default: false.
* **no-norm-cls**: If true, do not normalize the classes term. Default: false.
* **no-norm-con**: If true, do not normalize the constraint term. Default: false.
* **no-norm-obj**: If true, do not normalize the objectness term. Default: false.
* **start-iter**: If specified, start the number of iterations at <start-iter>. Useful for starting and stopping training, when you want the name of the model to reflect how many iterations it has actually undergone. Default: None. 

So, a typical use case of this script looks like:
``python solve.py --year=2012 --output=real1-click1-cls-con-obj --train-img=path/to/training/images/lmdb --train-gt=path/to/training/gt/images/lmdb --val-img=path/to/validation/images/lmdb --val-gt=path/to/validation/gt/images/lmdb --expectation --location --constraint --classes --objectness 

where we use the default learning rate of 1e-5 and momentum of 0.9, use a minibatch size of 20, display every 20 iterations, use GPU 0, initialize from the VGG16-CONV model.

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


