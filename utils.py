# coding=utf-8
# Copyright (c) 2018 Aria-K-Alethia
# Licence: MIT
import os
import struct
import platform
import numpy as np
import pandas as pd
import random
from time import time
from sklearn import preprocessing
from skimage import transform
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, train_test_split
if(platform.system() != "Linux"):
    import matplotlib.pyplot as plt

size_img = 28
threshold_color = 100 / 255

'''
def load_kaggle(path, prefix):
        overview:
            load kaggle data from path + prefix
            the prefix should be either 'train' or 'test'
    '''

def int2float_grey(x):
    x = x / 255
    return x
def zero_one_format(x, threshold = 100):
    x[x < threshold] = 0
    x[x >= threshold] = 1
    return x

def find_left_edge(x):
    edge_left = []
    n_samples = x.shape[0]
    for k in range(n_samples):
        for j in range(size_img):
            for i in range(size_img):
                if (x[k, size_img*i+j] >= threshold_color):
                    edge_left.append(j)
                    break
            if (len(edge_left) > k):
                break
    return edge_left

def find_right_edge(x):
    edge_right = []
    n_samples = x.shape[0]
    for k in range(n_samples):
        for j in range(size_img):
            for i in range(size_img):
                if (x[k, size_img*i+(size_img-1-j)] >= threshold_color):
                    edge_right.append(size_img-1-j)
                    break
            if (len(edge_right) > k):
                break
    return edge_right

def find_top_edge(x):
    edge_top = []
    n_samples = x.shape[0]
    for k in range(n_samples):
        for i in range(size_img):
            for j in range(size_img):
                if (x[k, size_img*i+j] >= threshold_color):
                    edge_top.append(i)
                    break
            if (len(edge_top) > k):
                break
    return edge_top

def find_bottom_edge(x):
    edge_bottom = []
    n_samples = x.shape[0]
    for k in range(n_samples):
        for i in range(size_img):
            for j in range(size_img):
                if (x[k, size_img*(size_img-1-i)+j] >= threshold_color):
                    edge_bottom.append(size_img-1-i)
                    break
            if (len(edge_bottom) > k):
                break
    return edge_bottom

def stretch_image(x):
    edge_left = find_left_edge(x)
    edge_right = find_right_edge(x)
    edge_top = find_top_edge(x)
    edge_bottom = find_bottom_edge(x)
    n_samples = x.shape[0]
    x = x.reshape(n_samples, size_img, size_img)
    for i in range(n_samples):      
        x[i] = transform.resize(x[i][edge_top[i]:edge_bottom[i]+1, edge_left[i]:edge_right[i]+1], (size_img, size_img))
    x = x.reshape(n_samples, size_img ** 2)
    return x

def var_select(x_t, x_e):
    selector = VarianceThreshold(threshold = 0).fit(x_t)
    x_t = selector.transform(x_t)
    x_e = selector.transform(x_e)
    return x_t, x_e

def chi2_select(x_t, y_t, x_e, k = 500):
    selector = SelectKBest(chi2, k = k).fit(x_t, y_t)
    x_t = selector.transform(x_t)
    x_e = selector.transform(x_e)
    return x_t, x_e

def pca_select(x_t, x_e, threshold = 0.95):
    pca = PCA(n_components = threshold)
    pca.fit(x_t)
    x_t = pca.transform(x_t)
    x_e = pca.transform(x_e)
    return x_t, x_e

def process_data(xt, yt, xe, ye, jitter = False):
    if(jitter):
        print('jitter the data...')
        xt, yt = jitter_data(xt, yt)
    print('Nomalize the data...')
    xt = int2float_grey(xt)
    xe = int2float_grey(xe)
    print('Stretch the data...')
    xt = stretch_image(xt)
    xe = stretch_image(xe)
    print('Remove the zero-variance feature...')
    xt, xe = var_select(xt, xe)
    print(xt.shape, xe.shape)
    print('Using PCA to select feature...')
    xt, xe = pca_select(xt, xe)
    print(xt.shape, xe.shape)
    return xt, yt, xe, ye

def accuracy(yp, yt):
    acc = (yp == yt).sum() / yp.shape[0]
    print('accuracy: %f' % acc)
    return acc

def cross_validation(model, xt, yt, cv = 5):
    scores = cross_val_score(model, xt, yt, cv)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    return scores

def dump_prediction(yp, name):
    '''
        this is for the kaggle competition
    '''
    pred = {"ImageId": range(1, yp.shape[0]+1), "Label": yp}
    pred = pd.DataFrame(pred)
    pred.to_csv("%s.csv" % name, index = False)

def load_mnist(path,prefix):
    '''
        overview: load data
        params:
            path: path of the file
            prefix: should be 'train' or 't10k'
        return:
            image,label
    '''
    assert prefix == "train" or prefix == "t10k"
    imgpath = os.path.join(path,"%s-images.idx3-ubyte" % prefix)
    labelpath = os.path.join(path,"%s-labels.idx1-ubyte" % prefix)
    imgfile = open(imgpath,"rb")
    labelfile = open(labelpath,"rb")
    img_magic,img_n,row,col = struct.unpack(">IIII",imgfile.read(16))
    label_magic,label_n = struct.unpack(">II",labelfile.read(8))
    image = np.fromfile(imgfile,dtype=np.uint8).reshape(img_n,row*col)
    label = np.fromfile(labelfile,dtype=np.uint8)
    return image,label

def load_kaggle(path):
    xk = pd.read_csv(path)
    xk = np.array(xk)
    yk, xk = xk[:,0], xk[:,1:]
    return xk, yk

def split_data(x,y,ratio):
    '''
        overview:
            split x and y by corresponding ratio
        params:
            x,y: data
            ratio: the percentage of the data needed by caller
        return:
            xs,ys
    '''
    assert ratio <= 1 and ratio >= 0
    n = len(y)
    xt, xd, yt, yd = train_test_split(x, y, test_size = ratio)
    return xt, yt, xd, yd

def look_data(x,y,number):
    '''
        overview: look some data of certain number
        params:
            x: image
            y: label
            number: [0,9]
    '''
    fig,ax = plt.subplots(nrows=5,ncols=5,sharex=True,sharey=True)
    ax = ax.flatten()
    temp = x[y==number]
    for i in range(25):
        img = temp[random.randint(0,temp.shape[0]-1)].reshape(28,28)
        ax[i].imshow(img,cmap='Greys',interpolation="nearest")
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()

def check_image(x):
    '''
        overview:
            This is a assistant function for process_data
    '''
    temp = np.sum(x[:,0,:],axis=1) # (60000,)
    assert sum(temp[temp > 0] > 7) == 0, "up side"
    temp = np.sum(x[:,:,-1],axis=1)
    assert sum(temp[temp > 0] > 7) == 0, "right side"
    temp = np.sum(x[:,-1,:],axis=1)
    assert sum(temp[temp > 0] > 7) == 0, "down side"
    temp = np.sum(x[:,:,0],axis=1)
    assert sum(temp[temp > 0] > 7) == 0, "left side"

def rotate_data(x,y,low = 10,high = 30):
    random.seed(time())
    outcome = x.copy().reshape(x.shape[0], int(x.shape[1]**0.5), int(x.shape[1]**0.5))
    for i in range(outcome.shape[0]):
        angle = random.randint(low, high)
        sign = random.randint(0,1)
        angle = -angle if sign == 0 else angle
        outcome[i] = transform.rotate(outcome[i], angle)
    outcome = outcome.reshape(x.shape[0],x.shape[1])
    x = np.concatenate([x, outcome], 0)
    y = np.concatenate([y]*2, 0)
    return x, y
def jitter_data(x,y):
    '''
        overview:
            jitter the data
        params:
            x: image
            y: label
        return:
            new_x,new_y
        NOTE:
            x and y will be altered directly
    '''
    # scale the data
    n = x.shape[0]
    x = x.reshape(x.shape[0],28,28)
    check_image(x) # should have no exception
    #upside jitter
    temp = np.zeros([n,1,28])
    up = np.delete(x,0,axis=1) #(60000,27,28)
    up = np.concatenate([up,temp],axis=1)
    up = up.reshape([n,784])
    #rightside jitter
    temp = np.zeros([n,28,1])
    right = np.delete(x,-1,axis=2) #(60000,28,27)
    right = np.concatenate([temp,right],axis=2)
    right = right.reshape([n,784])
    #downside jitter
    temp = np.zeros([n,1,28])
    down = np.delete(x,-1,axis=1)
    down = np.concatenate([temp,down],axis=1)
    down = down.reshape([n,784])
    #leftside jitter
    temp = np.zeros([n,28,1])
    left = np.delete(x,0,axis=2)
    left = np.concatenate([left,temp],axis=2)
    left = left.reshape([n,784])
    #concatenate all data
    x = x.reshape(n,784)
    x = np.concatenate([x,up,right,down,left],axis=0) #(300000,784)
    y = np.concatenate([y]*5,axis=0) #(300000,)
    return x,y



    
