##Key Files:

### ``eval.py``

This file evaluates a Caffe model on the PASCAL VOC segmentation dataset. At its completion, it will print the per-class and overall mean intersection-over-union achieved by the model on PASCAL VOC, as compared to the ground-truth segmentations. It will also visualize the segmentation masks predicted by the given model and save them as png images.

#### Inputs: 
All inputs are specified as constants at the top of the file. 
* ***is_pascal***: specifies whether or not to run on the PASCAL VOC dataset or another dataset (we have also experimented with SIFTFLOW). Should probably always be set to ``True``. 
* ***is_test***: specifies whether or not run on the PASCAL VOC validation set or test set. 
* ***is_numbers***: specifies whether or not to evaluate the mIOU performance of the given model. (For example, ``is_numbers=False`` and ``is_visualize=True``, the script will just visualize the predicted segmentation masks). 
* ***is_visualize***: specifies whether or not to also visualize the segmentation masks predicted by the given model and save them as png images.
* ***is_color***: specifies whether to visualize the predicted segmentation masks in human-visible color masks, or with their class labels. If ``is_color=True``, the script will visualize the segmentation masks with different RGB colors. If ``is_color=False``, the script will visualize the segmentation masks with the PASCAL class labels (0-20). 

is_pascal = True
is_test = True
is_numbers = True
is_visualize = True
is_color = True

* ***MODELS_DIR***: specifies the general directory containing all models. In our case, this is ``../../caffe/models/fcn-32s-pascal/``
* ***DIR***: specifies the specific model directory containing the saved .caffemodel files, visualized segmentations, and .prototxt files. It is concatenated onto the ``MODELS_DIR`` path. An example of one ``DIR`` is ``2012/real1-squiggle1-cls-con-obj``. 


