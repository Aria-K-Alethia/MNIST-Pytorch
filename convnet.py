# coding=utf-8
# Copyright (c) 2018 Aria-K-Alethia@github.com
# Licence: MIT
# CNN for MNIST classification

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from math import ceil
import sys
import argparse
import utils
from time import time
from itertools import count

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--inpath_mnist', type=str, default='.', help="the mnist data path")
parser.add_argument('--inpath_kaggle',type=str, default=None, help="the kaggle data path")
parser.add_argument('--gpu', type=int, default=-1, help="the gpu id, default=-1")
parser.add_argument('--jitter', action="store_true", help="whether to jitter the data or not")
parser.add_argument('--epoch', type=int, default=50, help="total epochs")
parser.add_argument('--pretrained', type=str, default=None, help="the pretrained model")
parser.add_argument('--verbose', type=int, default=0, help="the verbose level")
parser.add_argument('--mode', type=str, default='train', help="train or test")
parser.add_argument('--model_size', type=str, default='normal', help="normal or test")
parser.add_argument('--dev_ratio', type=float, default='0.3', help="the dev ratio")
parser.add_argument('--lr', type=float, default=1e-4, help="the learning rate default=1e-4")
parser.add_argument('--stretch', action="store_true", help="whether to stretch the data or not")
parser.add_argument('--method', type=str, default='adam', help="the optimize method, default=adam")
parser.add_argument('--rotate', action="store_true", help="whether to rotate the data or not")
parser.add_argument('--plus', action="store_true", help="use the plus model")
class Lenet(nn.Module):
    """
    """
    def __init__(self):
        super(Lenet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.lsm = nn.LogSoftmax(-1)
    def forward(self, x):
        x = F.relu(self.conv1(x)) # 24 x 24
        x = F.max_pool2d(x, (2,2)) # 12 x 12
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, (2,2))
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.lsm(self.fc3(x))
        return x

class LenetPlus(nn.Module):
    """
    """
    def __init__(self):
        super(LenetPlus, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding = 1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding = 1)
        self.fc1 = nn.Linear(64 * 7 * 7, 1000)
        self.fc2 = nn.Linear(1000, 100)
        self.fc3 = nn.Linear(100, 10)
        self.lsm = nn.LogSoftmax(-1)
    def forward(self, x):
        x = F.relu(self.conv1(x)) # 28 x 28
        x = F.max_pool2d(x, (2,2)) # 14 x 14
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, (2,2))
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.lsm(self.fc3(x))
        return x

class Config(object):
    def __init__(self, mode):
        super(Config, self).__init__()
        # the common paras
        self.mode = mode
        self.optim_method = 'adam'
        self.lr = 1e-4
        if(mode == 'test'):
            self.small()
        elif(mode == 'normal'):
            self.normal()
        else:
            raise Exception('No such mode in config: %s' % mode)
    def small(self):
        self.batch_size = 2
    def normal(self):
        self.batch_size = 128
def save_model(loss, model, config, epoch):
    model_state = model.state_dict()
    ckpt = {
        'model': model_state,
        'config': config,
        'loss': loss,
        'epoch': epoch
    }
    path = './model/%.5f_loss_%d_epoch.pt' % (loss, epoch)
    print('**Saving the model to %s' % path)
    torch.save(ckpt, path)

def batch_iter(x, y, batch_size, device, shuffle = False):
    # shuffle the data
    assert x.shape[0] == y.shape[0]
    if(shuffle):
        indices = torch.randperm(x.shape[0])
        x = x[indices]
        y = y[indices]
    # generate the data
    l = ceil(x.shape[0] / batch_size)
    for idx in range(l):
        x_batch = x[idx*batch_size:(idx+1)*batch_size]
        y_batch = y[idx*batch_size:(idx+1)*batch_size]
        x_batch = torch.tensor(x_batch).float().view(-1, 1, 28, 28)
        y_batch = torch.tensor(y_batch).long()
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        yield x_batch, y_batch


def validate(data, model, criterion, device, batch_size):
    model.eval()
    xe, ye = data
    dataset = batch_iter(xe, ye, batch_size, device)
    current_loss = 0
    out = []
    for step, (x, y) in enumerate(dataset):
        output = model(x)
        _,labels = output.max(1)
        out.append(labels)
        loss = criterion(output, y)
        current_loss += float(loss)
        torch.cuda.empty_cache()
    model.train()
    out = torch.cat(out, 0).cpu().numpy()
    acc = (out == ye).sum() / ye.shape[0]
    return current_loss / (step + 1), acc


def train(model, train_data, dev_data, device, epochs, optimizer, config, ckpt = None):
    # criterion
    criterion = nn.NLLLoss()
    # init some vars
    if(ckpt is not None):
        last_epoch = ckpt['epoch']
        best_loss = ckpt['loss']
        print('*You have pretrained model, best acc: %f, last_epoch: %d'\
            %(best_loss, last_epoch))
    else:
        best_loss = 0
        last_epoch = -1
    # mode
    model.train()
    # get train data
    xt, yt = train_data
    print("*Train shape: %s, dev shape: %s" % (str(xt.shape), str(dev_data[0].shape)))
    print('**Train begin, data size: %d, step per epoch: %d' %\
        (xt.shape[0], xt.shape[0] / config.batch_size))
    print('**Total epoch: %d, start from: %d' %\
        (epochs, last_epoch + 1))
    for epoch in range(last_epoch+1, epochs):
        # clear the gradient
        model.zero_grad()
        current_loss = 0
        epoch_begin = time()
        # data iterator
        dataset = batch_iter(xt, yt, config.batch_size, device, shuffle = True)
        for step, (x, y) in enumerate(dataset):
            try:
                output = model(x)
            except:
                print(x.shape)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            current_loss += float(loss)
            torch.cuda.empty_cache()
        # epoch end, record the time and validate
        epoch_end = time()
        if(dev_data is not None):
            dev_loss, acc = validate(dev_data, model, criterion, device, config.batch_size)
        else:
            dev_loss = current_loss / (step+1)
            acc = -1
        print('*Current epoch: %d, time: %f, train loss: %f,dev acc: %f, dev loss: %f, best loss: %f'\
            % (epoch, epoch_end - epoch_begin, current_loss / (step+1), acc, dev_loss, best_loss))
        # compare the current dev loss with best loss
        if(acc > best_loss):
            # get new best loss, save the model
            print('**New best acuracy: %f...' % acc)
            print('**Save model...')
            save_model(acc, model, config, epoch)
            best_loss = acc
    print('**Train end...')

def make_model(device,ckpt = None, plus = False):
    if(ckpt is None):
        if(plus):
            model = LenetPlus()
        else:
            model = Lenet()
        print('*Initializing parameters...')
        for p in model.parameters():
            p.data.uniform_(-1e-1, 1e-1)
    else:
        if(plus):
            model = LenetPlus()
        else:
            mode = Lenet()
        model.load_state_dict(ckpt['model'])
    model = model.to(device)
    return model

def make_optimizer(model, method, lr):
    if(method == 'adam'):
        optimizer = optim.Adam(model.parameters(), lr = lr)
    elif(method == 'sgd'):
        optimizer = optim.SGD(model.parameters(), lr = lr)
    else:
        raise Exception("No such optimizer: %s" % method)
    return optimizer

def make_data(flags):
    if(flags.mode == 'train'):
        xt, yt = utils.load_mnist(flags.inpath_mnist, 'train')
        if(flags.inpath_kaggle is not None):
            xk, yk = utils.load_kaggle(flags.inpath_kaggle)
            xt = np.concatenate([xt, xk], 0)
            yt = np.concatenate([yt, yk], 0)
        xt = utils.int2float_grey(xt)
        if(flags.rotate):
            xt, yt = utils.rotate_data(xt, yt)
        if(flags.jitter):
            xt, yt = utils.jitter_data(xt, yt)
        xd, yd = utils.load_mnist(flags.inpath_mnist, 't10k')
        xd = utils.int2float_grey(xd)
        if(flags.stretch):
            xt = utils.stretch_image(xt)
            xd = utils.stretch_image(xd)
        if(flags.verbose >= 2):
            print(xd.shape)
        return xt, yt, xd, yd
        #return utils.split_data(xt, yt, flags.dev_ratio)
    else:
        x, y = utils.load_mnist(flags.inpath_mnist, 't10k')
        x = utils.int2float_grey(x)
        if(flags.stretch):
            x = utils.stretch_image(x)
        return x, y
def train_wrapper(flags):
    print('Training...')
    # prepare the data
    print('Loading data...')
    xt, yt, xd, yd = make_data(flags)
    if(xd.shape[0] == 0):
        xd, yd = None, None
    # device
    device = torch.device('cpu') if flags.gpu == -1 else torch.device('cuda')
    # load ckpt, if exists, otherwise get a config
    if(flags.pretrained is None):
        config = Config(flags.model_size)
        ckpt = None
    else:
        ckpt = torch.load(flags.pretrained, map_location = lambda sto, loc: sto)
        config = ckpt['config']
    # make model
    print('Making model...')
    model = make_model(device, ckpt, flags.plus)
    # make optimizer
    print('Making optimizer...')
    optimizer = make_optimizer(model, flags.method, flags.lr)
    # train
    train(model, [xt, yt], [xd, yd] if xd is not None else None, device, flags.epoch, optimizer, config, ckpt)

def evaluate(flags):
    print('Evaluating...')
    # prepare the data
    print('Making data...')
    xe, ye = make_data(flags)
    # device
    device = torch.device('cpu') if flags.gpu == -1 else torch.device('cuda')
    # make pretrained model
    print('Loading pretrained model...')
    if(flags.pretrained is not None):
        ckpt = torch.load(flags.pretrained, map_location = lambda sto, loc: sto)
        config = ckpt['config']
        model = make_model(device, ckpt)
    else:
        raise Exception("No pretrained model given in mode evaluate!")
    # get the pred
    print('Begining...')
    dataset = batch_iter(xe, ye, config.batch_size, device)
    out = []
    for x, y in dataset:
        # get output
        output = model(x)
        _, labels = output.max(1)
        out.append(labels)
    # report the accuracy
    print('Done...')
    out = torch.cat(out, 0).numpy()
    assert out.shape[0] == ye.shape[0]
    acc = (out == ye).sum() / ye.shape[0]
    print('Accuracy: %f' % acc)
    # return
    return acc
def main(flags):
    # mode
    if(flags.mode == 'train'):
        train_wrapper(flags)
    else:
        evaluate(flags)


if __name__ == '__main__':
    flags, _ = parser.parse_known_args(sys.argv[1:])
    main(flags)

