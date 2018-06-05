# MNIST-Pytorch
A simple CNN for MNIST classification task using Pytorch

## Requirements
* Python3
* Pytorch 0.4
* numpy
* pandas
* sklearn
* skimage

## Overview
I implement a simple Lenet on Pytorch to solve the MNIST task, by using  jitter, rotate data and other
augmentation, the final accuracy on the 10K test set is 99.76%.

## Usage
*NOTE*: before training, you need to download the data from [MNIST](http://yann.lecun.com/exdb/mnist/)
you can also download some extra data from [Kaggle Competition](https://www.kaggle.com/c/digit-recognizer/data)

### Train
	python3 convnet.py --epoch 10
see python3 convnet.py -h for more options.

## Conclusion
1. more data, more better your model's performance will be.
2. distortion could improve the performance, but not significantly.
3. A more complex model may not be useful when you work on such small task.

## Reference
1. [Kaggle SiWei's Keneral](https://www.kaggle.com/endlesslethe/siwei-digit-recognizer-top20)
this code use some feature selection technique for Machine Learning model, I copied some of them,
but not use them in my model. You may use this code on your Machine Learning model such as SVM,
KNN etc.
