"""
License: Apache-2.0
Author: Huadong Liao
E-mail: naturomics.liao@gmail.com
"""

import tensorflow as tf
from config import cfg


class MnistNet(object):
    def __init__(self):   
        # Declare model parameters
        self.conv1_weight = tf.Variable(tf.truncated_normal([4, 4, cfg.num_channels, cfg.conv1_features],
                                                       stddev=0.1, dtype=tf.float32))
        self.conv1_bias = tf.Variable(tf.zeros([cfg.conv1_features], dtype=tf.float32))
        
        self.conv2_weight = tf.Variable(tf.truncated_normal([4, 4, cfg.conv1_features, cfg.conv2_features],
                                                       stddev=0.1, dtype=tf.float32))
        self.conv2_bias = tf.Variable(tf.zeros([cfg.conv2_features], dtype=tf.float32))
        
        # fully connected variables
        resulting_width = cfg.image_width // (cfg.max_pool_size1 * cfg.max_pool_size2)
        resulting_height = cfg.image_height // (cfg.max_pool_size1 * cfg.max_pool_size2)        
        full1_input_size = resulting_width * resulting_height * cfg.conv2_features
        
        self.full1_weight = tf.Variable(tf.truncated_normal([full1_input_size, cfg.fully_connected_size1],
                                  stddev=0.1, dtype=tf.float32))
        self.full1_bias = tf.Variable(tf.truncated_normal([cfg.fully_connected_size1], stddev=0.1, dtype=tf.float32))
        
        self.full2_weight = tf.Variable(tf.truncated_normal([cfg.fully_connected_size1, cfg.target_size],
                                                       stddev=0.1, dtype=tf.float32))
        self.full2_bias = tf.Variable(tf.truncated_normal([cfg.target_size], stddev=0.1, dtype=tf.float32))
        
        # Declare model placeholders
        self.x_input_shape = (cfg.batch_size, cfg.image_width, cfg.image_height, cfg.num_channels)
        self.x_input = tf.placeholder(tf.float32, shape=self.x_input_shape)
        self.y_target = tf.placeholder(tf.int32, shape=(cfg.batch_size))
        
        self.eval_input_shape = (cfg.batch_size, cfg.image_width, cfg.image_height, cfg.num_channels)
        self.eval_input = tf.placeholder(tf.float32, shape=self.eval_input_shape)
        self.eval_target = tf.placeholder(tf.int32, shape=(cfg.batch_size))

        tf.logging.info('Seting up the main structure')


    def build_arch(self, input_data):
        # First Conv-ReLU-MaxPool Layer
        conv1 = tf.nn.conv2d(input_data, self.conv1_weight, strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, self.conv1_bias))
        max_pool1 = tf.nn.max_pool(relu1, ksize=[1, cfg.max_pool_size1, cfg.max_pool_size1, 1],
                                   strides=[1, cfg.max_pool_size1, cfg.max_pool_size1, 1], padding='SAME')
    
        # Second Conv-ReLU-MaxPool Layer
        conv2 = tf.nn.conv2d(max_pool1, self.conv2_weight, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, self.conv2_bias))
        max_pool2 = tf.nn.max_pool(relu2, ksize=[1, cfg.max_pool_size2, cfg.max_pool_size2, 1],
                                   strides=[1, cfg.max_pool_size2, cfg.max_pool_size2, 1], padding='SAME')
    
        # Transform Output into a 1xN layer for next fully connected layer
        final_conv_shape = max_pool2.get_shape().as_list()
        final_shape = final_conv_shape[1] * final_conv_shape[2] * final_conv_shape[3]
        flat_output = tf.reshape(max_pool2, [final_conv_shape[0], final_shape])
    
        # First Fully Connected Layer
        fully_connected1 = tf.nn.relu(tf.add(tf.matmul(flat_output, self.full1_weight), self.full1_bias))
    
        # Second Fully Connected Layer
        final_model_output = tf.add(tf.matmul(fully_connected1, self.full2_weight), self.full2_bias)
        
        return(final_model_output)
    

    def loss(self, model_output, test_model_output):
        # Declare Loss Function (softmax cross entropy)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model_output, labels=self.y_target))
        
        # Create a prediction function
        prediction = tf.nn.softmax(model_output)
        test_prediction = tf.nn.softmax(test_model_output)
            
        # Create an optimizer
        mnist_optimizer = tf.train.MomentumOptimizer(cfg.learning_rate, 0.9)
        train_step = mnist_optimizer.minimize(loss)
        
        return loss, prediction, test_prediction, train_step
    
    