Code for our ECCV paper [What's the Point: Semantic Segmentation with Point Supervision](http://vision.stanford.edu/whats_the_point/).

## Summary
This library is a custom build of Caffe for semantic image segmentation with point supervision. It is written for the "FCN-32S-PASCAL" model (fully-convolutional network, stride of 32 for PASCAL VOC 2012), based on [this model](https://github.com/shelhamer/fcn.berkeleyvision.org/tree/master/voc-fcn32s). More details on the original model are available [here](https://github.com/shelhamer/fcn.berkeleyvision.org). 

## Quick Start 

## Code Structure

All Caffe src files and models are in the [caffe](caffe/) directory. All code and scripts to run and evaluate the various models are in the [whats-the-point-2016](whats-the-point-2016/) directory.

Modified Caffe src files:
* [``softmax_loss_expectation_layer.cpp``](https://github.com/abearman/whats-the-point1/blob/master/caffe/src/caffe/layers/softmax_loss_expectation_layer.cpp): CPU version of our custom loss layer 
* [``softmax_loss_expectation_layer.cu``](https://github.com/abearman/whats-the-point1/blob/454f0b04d8875349d287801d1041aa9820fe7f50/caffe/src/caffe/layers/softmax_loss_expectation_layer.cu): GPU version of our custom loss layer
* [``loss_layers.hpp``](https://github.com/abearman/whats-the-point1/blob/454f0b04d8875349d287801d1041aa9820fe7f50/caffe/include/caffe/loss_layers.hpp): Registers our new loss layer
* [``caffe.proto``](https://github.com/abearman/whats-the-point1/blob/454f0b04d8875349d287801d1041aa9820fe7f50/caffe/src/caffe/proto/caffe.proto): Adds new loss parameters.
