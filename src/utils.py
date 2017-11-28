"""
Author: Jay Meng
E-mail: jalymo@126.com
Wechat：345238818
"""

import os
import scipy
import numpy as np
import tensorflow as tf

from config import cfg

def load_mnist(path):
    fd = open(os.path.join(path, 'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trX = loaded[16:].reshape((cfg.train_batch, cfg.image_width, cfg.image_height, cfg.num_channels)).astype(np.float)

    fd = open(os.path.join(path, 'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trY = loaded[8:].reshape((cfg.train_batch)).astype(np.int32)

    fd = open(os.path.join(path, 't10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teX = loaded[16:].reshape((cfg.test_batch, cfg.image_width, cfg.image_height, cfg.num_channels)).astype(np.float)

    fd = open(os.path.join(path, 't10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teY = loaded[8:].reshape((cfg.test_batch)).astype(np.int32)
 
    return trX, trY, teX, teY

# Create accuracy function
def get_accuracy(logits, targets):
    batch_predictions = np.argmax(logits, axis=1)
    num_correct = np.sum(np.equal(batch_predictions, targets))
    #return(100. * num_correct/batch_predictions.shape[0])
    return(1.0 * num_correct/batch_predictions.shape[0])
