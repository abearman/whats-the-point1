Code for our ECCV paper [What's the Point: Semantic Segmentation with Point Supervision](http://vision.stanford.edu/whats_the_point/).

## Summary
This library is a custom build of Caffe for semantic image segmentation with point supervision. It is written for the "FCN-32S-PASCAL" model (fully-convolutional network with a stride of 32 for PASCAL VOC 2012), based on [this model](https://github.com/shelhamer/fcn.berkeleyvision.org/tree/master/voc-fcn32s). More details on the original model are available [here](https://github.com/shelhamer/fcn.berkeleyvision.org). 
## How to Use

## Code Structure

All Caffe src files are in the [caffe](caffe/) directory. All code and scripts to run and evaluate the various models are in the [whats-the-point-2016](whats-the-point-2016/) directory.

``caffe/src/caffe/layers/softmax_loss_expectation_layer.cpp``

* 
