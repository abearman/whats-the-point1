##Key Files:

### ``eval.py``

This file evaluates a Caffe model on the PASCAL VOC segmentation dataset. At its completion, it will print the per-class and overall mean intersection-over-union achieved by the model on PASCAL VOC, as compared to the ground-truth segmentations.

#### Inputs: 
All inputs are specified as constants at the top of the file. 
* ``MODELS_DIR``: specifies the general directory containing all models. In our case, this is ``../../caffe/models/fcn-32s-pascal/``
* ``DIR``: specifies the specific model directory containing the saved .caffemodel files, visualized segmentations, and .prototxt files. It is concatenated onto the ``MODELS_DIR`` path. An example of one ``DIR`` is ``2012/real1-squiggle1-cls-con-obj``. 
*  


