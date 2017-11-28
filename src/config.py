import tensorflow as tf

flags = tf.app.flags

############################
#    environment setting   #
############################
flags.DEFINE_string('data_dir', '../data/mnist', 'path for mnist dataset')
flags.DEFINE_string('result_dir', '../results', 'path for saving results')
flags.DEFINE_string('model_dir', '../models', 'path for saving results')


############################
#    hyper parameters      #
############################
flags.DEFINE_integer('train_batch', 60000, 'train data batch numbers')
flags.DEFINE_integer('test_batch', 10000, 'test data batch numbers')

flags.DEFINE_integer('batch_size', 128, 'batch size')
flags.DEFINE_integer('image_width', 28, 'image width')
flags.DEFINE_integer('image_height', 28, 'image height')
flags.DEFINE_integer('num_channels', 1, 'image channel')
flags.DEFINE_float('learning_rate', 0.0005, 'learning rate')

flags.DEFINE_integer('target_size', 10, 'target size')
flags.DEFINE_integer('epoch', 10, 'epoch')

flags.DEFINE_integer('test_sum_freq', 5, 'the frequency of saving test summary(step)')
flags.DEFINE_integer('save_freq', 2, 'the frequency of saving model(epoch)')
flags.DEFINE_boolean('if_showplt', False, 'eval every times')


############################
#    mnist net setting     #
############################
flags.DEFINE_integer('conv1_features', 25, 'conv1_features')
flags.DEFINE_integer('conv2_features', 20, 'conv2_features')
flags.DEFINE_integer('max_pool_size1', 2, 'max_pool_size1')
flags.DEFINE_integer('max_pool_size2', 2, 'max_pool_size2')
flags.DEFINE_integer('fully_connected_size1', 100, 'fully_connected_size1')


cfg = tf.app.flags.FLAGS
# tf.logging.set_verbosity(tf.logging.INFO)
