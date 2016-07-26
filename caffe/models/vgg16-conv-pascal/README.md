Initial network for all our fully convolutional networks trained on PASCAL VOC 2012. All classifier weights are zero, except for weights learned by the [original VGG network](https://gist.github.com/ksimonyan/211839e770f7b538e2d8) for classes common to both PASCAL and ILSVRC. It does a 21-way pixel-wise softmax classification for the 21 PASCAL VOC 2012 classes (including background).

