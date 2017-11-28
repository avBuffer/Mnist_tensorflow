"""
Author: Jay Meng
E-mail: jalymo@126.com
Wechat：345238818
"""

# Introductory CNN Model: MNIST Digits
# In this example, we will load the MNIST handwritten
# digits and create a simple CNN network to predict the
# digit category (0-9)

import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import time
import shutil

from config import cfg
from utils import load_mnist
from utils import get_accuracy
from mnistNet import MnistNet

    
def main(_):    
   
    mnistNet = MnistNet()
    model_output = mnistNet.build_arch(mnistNet.x_input)    
    test_model_output = mnistNet.build_arch(mnistNet.eval_input)
    
    loss, prediction, test_prediction, train_step = mnistNet.loss(model_output, test_model_output)    
   
    num_batch = int(cfg.train_batch / cfg.batch_size)
    num_test_batch = cfg.test_batch // cfg.batch_size
    
    # Convert images into 28x28 (they are downloaded as 1x784)
    train_xdata, train_labels, test_xdata, test_labels = load_mnist(cfg.data_dir)    
        
    train_loss = []
    train_acc = []
    test_acc = []
    
    # Saving accuracy results
    path = cfg.result_dir + '/accuracy.csv'
    if not os.path.exists(cfg.result_dir):
        os.mkdir(cfg.result_dir)
    elif os.path.exists(path):
        os.remove(path)

    fd_results = open(path, 'w')
    fd_results.write('Epoch,Step,TrainLoss,TrainAcc,TestAcc,CostTime\n')

    if not os.path.exists(cfg.model_dir):
        os.mkdir(cfg.model_dir)
    elif os.path.exists(cfg.model_dir):
        shutil.rmtree(cfg.model_dir)
    
    # Initialize Variables
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
            
        for epoch in range(cfg.epoch):  
            temp_train_acc = 0
            for step in tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):            
                start = step * cfg.batch_size
                end = start + cfg.batch_size
                
                rand_x = train_xdata[start:end]
                rand_y = train_labels[start:end]
                train_dict = {mnistNet.x_input: rand_x, mnistNet.y_target: rand_y}
                
                sess.run(train_step, feed_dict=train_dict)
                temp_train_loss, temp_train_preds = sess.run([loss, prediction], feed_dict=train_dict)
                temp_train_acc = get_accuracy(temp_train_preds, rand_y)
                
                train_loss.append(temp_train_loss)
                train_acc.append(temp_train_acc)            
                            
                if (step+1) % cfg.test_sum_freq == 0:
                    temp_test_acc = 0
                    startTime = time.time()
                    for i in range(num_test_batch):
                        start = i * cfg.batch_size
                        end = start + cfg.batch_size
                        
                        eval_x = test_xdata[start:end]
                        eval_y = test_labels[start:end]
            
                        test_dict = {mnistNet.eval_input: eval_x, mnistNet.eval_target: eval_y}
                        test_preds = sess.run(test_prediction, feed_dict=test_dict)                    
                        temp_test_acc += get_accuracy(test_preds, eval_y)     
                                                            
                    temp_test_acc = temp_test_acc / num_test_batch
                    test_acc.append(temp_test_acc)          
                        
                    endTime = time.time()
                    costTime = (endTime - startTime)*1000
                    fd_results.write(str(epoch) + ',' + str(step + 1) + ',' + 
                                     str(temp_train_loss) + ',' + str(temp_train_acc) + ',' + 
                                     str(temp_test_acc) + ',' + str(costTime) + '\n')
                    fd_results.flush()
    
                    acc_and_loss = [epoch, (step+1), 100*temp_train_loss, 100*temp_train_acc, 100*temp_test_acc, costTime]
                    acc_and_loss = [np.round(x,2) for x in acc_and_loss]
                    print('Epoch # {}. Step # {}. Train Loss: {:.2f}%. Train Acc (Test Acc): {:.2f}% ({:.2f}%). Cost Time: {}ms'.format(*acc_and_loss)) 
        
            if epoch % cfg.save_freq == 0:
                saver.save(sess, cfg.model_dir + '/model_epoch_%04d_step_%02d' % (epoch, step)) 
                               
    fd_results.close()            
    tf.logging.info('Training done') 
      
    if cfg.if_showplt:    
        # Matlotlib code to plot the loss and accuracies
        eval_indices = range(0, cfg.epoch, cfg.eval_every)
        # Plot loss over time
        plt.plot(eval_indices, train_loss, 'k-')
        plt.title('Softmax Loss per Generation')
        plt.xlabel('Generation')
        plt.ylabel('Softmax Loss')
        plt.show()
        
        # Plot train and test accuracy
        plt.plot(eval_indices, train_acc, 'k-', label='Train Set Accuracy')
        plt.plot(eval_indices, test_acc, 'r--', label='Test Set Accuracy')
        plt.title('Train and Test Accuracy')
        plt.xlabel('Generation')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
        plt.show()
        
        # Plot some samples
        # Plot the 6 of the last batch results:
        actuals = rand_y[0:6]
        predictions = np.argmax(temp_train_preds,axis=1)[0:6]
        images = np.squeeze(rand_x[0:6])
        
        Nrows = 2
        Ncols = 3
        for i in range(6):
            plt.subplot(Nrows, Ncols, i+1)
            plt.imshow(np.reshape(images[i], [28,28]), cmap='Greys_r')
            plt.title('Actual: ' + str(actuals[i]) + ' Pred: ' + str(predictions[i]),
                                       fontsize=10)
            frame = plt.gca()
            frame.axes.get_xaxis().set_visible(False)
            frame.axes.get_yaxis().set_visible(False)

      
    
if __name__ == "__main__":
    tf.app.run()
    