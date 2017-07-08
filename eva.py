"""Builds the EVA network.

Summary of available functions:

 # Compute input spectrograms and labels for training.
 spectrograms, labels = inputs()

 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)

 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(predictions, labels)

 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)
"""
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import Utils.Eva_config_consts as config
from Storage.TFStorage import TFStorage, TFStorageLabelOption, TFStorageOpenOptions

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', config.BATCH_SIZE,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('storage_path', config.DATESET_FILE_PATH(),
                           """Path to the CIFAR-10 data directory.""")

# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = config.NUM_PHONOME_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = config.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = config.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0  # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.001  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.02  # Initial learning rate.
INITIAL_CONV_VARIABLES_STDDEV = 5e-2  # Initial stddev for convolution layer variables
DROPOUT_COEFICIENT = 0.5  # 50%


def _activation_summary(x):
    """
    Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.

    :param x: Tensor
    :return:
    """

    tensor_name = x.op.name
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
    """
    Helper to create a Variable stored on CPU memory.

    :param name: name of the variable
    :param shape: list of ints
    :param initializer: initializer for Variable
    :return: variable tensor
    """
    with tf.device('/cpu:0'):
        dtype = tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var


def _variable(name, shape, stddev):
    """
    Helper to create an initialized Variable.
    All variables are initialized with truncated normal distribution.

    :param name: name of the variable
    :param shape: list of ints which denotes a variable shape
    :param stddev: standard deviation of a truncated Gaussian distribution
    :return: variable tensor
    """
    dtype = tf.float32
    variable = _variable_on_cpu(
        name,
        shape,
        tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    return variable

def _variable_with_weight_decay(name, shape, stddev, wd):
    """
    Helper to create an initialized Variable with weight decay.
    All variables are initialized with truncated normal distribution.

    :param name: name of the variable
    :param shape: list of ints which denotes a variable shape
    :param stddev: standard deviation of a truncated Gaussian distribution
    :param wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
    :return: variable tensor
    """

    dtype = tf.float32
    var = _variable_on_cpu(
        name,
        shape,
        tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

def conv_layer(layer_name, filter_shape, strides, input):
    """
    Creates a convolutional layer. The kernels are initialized with truncated normal distribution
    and 0 weight decay. All biases are initialized with 0.
    
    :param layer_name: A layer name
    :param filter_shape: Filter shape of type [h, w, c_i, c_o] where
     h - height
     w - width
     c_i - input channels
     c_o - output channels
    :param strides: the same as tf.nn.conv2.strides 
    :param input: input batch
    :return: activations
    """
    with tf.variable_scope(layer_name) as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=filter_shape,
                                             stddev=INITIAL_CONV_VARIABLES_STDDEV,
                                             wd=0.0)
        conv = tf.nn.conv2d(input, kernel, strides, padding='SAME')
        biases_count = filter_shape[-1];
        biases = _variable_on_cpu('biases', [biases_count], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        activations = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(activations)
        return activations

def pool_layer(layer_name, filter_shape, strides, input):
    """
    Creates a pooling layer.

    :param layer_name: layer name
    :param filter_shape: tf.nn.max_pool.ksize
    :param strides: the same as tf.nn.max_pool.strides 
    :param input: input batch
    :return: activations
    """
    return tf.nn.max_pool(input, ksize=filter_shape, strides=strides,
                   padding='SAME', name=layer_name)

def fully_connected_layer(layer_name, neurons_number, input):
    """
    Creates a fully connected layer
    
    :param layer_name: layer name
    :param neurons_number: number of neurons in the layer
    :param input: input
    :return: activations
    """
    with tf.variable_scope(layer_name) as scope:
        inputs = input
        # Dropout
        if train:
            inputs = tf.nn.dropout(input, DROPOUT_COEFICIENT)

        dim = inputs.get_shape()[1].value
        weights = _variable_with_weight_decay('weights', shape=[dim, neurons_number], stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [neurons_number], tf.constant_initializer(0.1))
        fc = tf.nn.relu(tf.matmul(inputs, weights) + biases, name=scope.name)
        _activation_summary(fc)
        return fc


def inference2(spectograms, train=False):
    """
    Build the EVA model.


    | Layer     | Layer output size |
    |-----------|-------------------|
    | INPUT     | 64x11x1          |
    | CONV3-64  | 64x11x64         |
    | CONV3-64  | 64x11x64         |
    | POOL 2x1  | 32x11x64         |
    | CONV3-128 | 32x11x128        |
    | CONV3-128 | 32x11x128        |
    | POOL 2x1  | 16x11x128        |
    | CONV3-256 | 16x11x256        |
    | CONV3-256 | 16x11x256        |
    | POOL 2x1  | 8x11x256         |
    |           |                   |
    | dropout   |                   |
    | FC1       | 2048x1x1          |
    | dropout   |                   |
    | FC2       | 2048x1x1          |
    | dropout   |                   |
    | FC3       | Speaker / Phoneme |

    :param spectograms: Spectrograms of size [config.SPECTROGRAM_HEIGHT x config.SPECTROGRAM_CHUNK_LENGTH x 2], which
        are obtained from inputs().
    :return: Logits
    """

    conv1_1 = conv_layer('conv1_1', [3, 3, 1, 64], [1, 1, 1, 1], spectograms)
    conv1_2 = conv_layer('conv1_2', [3, 3, 64, 64], [1, 1, 1, 1], conv1_1)
    pool1 = pool_layer('pool1', [1, 2, 1, 1], [1, 2, 1, 1], conv1_2)

    conv2_1 = conv_layer('conv2_1', [3, 3, 64, 128], [1, 1, 1, 1], pool1)
    conv2_2 = conv_layer('conv2_2', [3, 3, 128, 128], [1, 1, 1, 1], conv2_1)
    pool2 = pool_layer('pool2', [1, 2, 1, 1], [1, 2, 1, 1], conv2_2)

    conv3_1 = conv_layer('conv3_1', [3, 3, 128, 256], [1, 1, 1, 1], pool2)
    conv3_2 = conv_layer('conv3_2', [3, 3, 256, 256], [1, 1, 1, 1], conv3_1)
    pool3 = pool_layer('pool3', [1, 2, 1, 1], [1, 2, 1, 1], conv3_2)

    # Move everything into a vector so we can perform a single matrix multiply.
    reshaped = tf.reshape(pool3, [FLAGS.batch_size, -1])
    fc1 = fully_connected_layer('fc1', 2048, reshaped)
    fc2 = fully_connected_layer('fc2', 2048, fc1)

    # linear layer(WX + b),
    # We don't apply softmax here because
    # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
    # and performs the softmax internally for efficiency.
    with tf.variable_scope('softmax_linear') as scope:
        # Dropout
        if train:
            fc2 = tf.nn.dropout(fc2, DROPOUT_COEFICIENT)
        weights = _variable_with_weight_decay('weights', [2048, config.NUM_PHONOME_CLASSES], stddev=1 / 2048.0, wd=0.0)
        biases = _variable_on_cpu('biases', [config.NUM_PHONOME_CLASSES],
                                  tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(fc2, weights), biases, name=scope.name)
        _activation_summary(softmax_linear)

    return softmax_linear

def inference(spectograms, train=False):
    """
    Build the EVA model.

    | Layer     | Layer output size |
    |-----------|-------------------|
    | INPUT     | 64x11x1          |
    | CONV3-64  | 64x11x64         |
    | CONV3-64  | 64x11x64         |
    | POOL 2x1  | 32x11x64         |
    |           |                   |
    | dropout   |                   |
    | FC1       | 2048x1x1          |
    | dropout   |                   |
    | FC2       | 2048x1x1          |
    | dropout   |                   |
    | FC3       | Speaker / Phoneme |

    :param spectograms: Spectrograms of size [config.SPECTROGRAM_HEIGHT x config.SPECTROGRAM_CHUNK_LENGTH x 2], which
        are obtained from inputs().
    :return: Logits
    """

    conv1_1 = conv_layer('conv1_1', [3, 3, 1, 64], [1, 1, 1, 1], spectograms)
    conv1_2 = conv_layer('conv1_2', [3, 3, 64, 64], [1, 1, 1, 1], conv1_1)
    pool1 = pool_layer('pool1', [1, 2, 1, 1], [1, 2, 1, 1], conv1_2)

    # Move everything into a vector so we can perform a single matrix multiply.
    reshaped = tf.reshape(pool1, [FLAGS.batch_size, -1])
    fc1 = fully_connected_layer('fc1', 2048, reshaped)
    fc2 = fully_connected_layer('fc2', 2048, fc1)

    # linear layer(WX + b),
    # We don't apply softmax here because
    # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
    # and performs the softmax internally for efficiency.
    with tf.variable_scope('softmax_linear') as scope:
        # Dropout
        if train:
            fc2 = tf.nn.dropout(fc2, DROPOUT_COEFICIENT)
        weights = _variable_with_weight_decay('weights', [2048, config.NUM_PHONOME_CLASSES], stddev=1 / 2048.0, wd=0.0)
        biases = _variable_on_cpu('biases', [config.NUM_PHONOME_CLASSES],
                                  tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(fc2, weights), biases, name=scope.name)
        _activation_summary(softmax_linear)

    return softmax_linear


def loss(logits, labels):
    """
    Add L2Loss to all the trainable variables.

    :param logits: Logits from inference().
    :param labels: Labels from inputs(). 1-D tensor of shape [batch_size]
    :return: Loss tensor of type float.
    """

    # Calculate the average cross entropy loss across the batch.
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
    """
    Add summaries for losses in EVA model.

    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.

    :param total_loss: Total loss from loss().
    :return: loss_averages_op: op for generating moving averages of losses.
    """

    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op


def train(total_loss, global_step):
    """
    Train CIFAR-10 model.

    Create an optimizer and apply to all trainable variables. Add moving
    average for all trainable variables.
    :param total_loss: Total loss from loss().
    :param global_step: Integer Variable counting the number of training steps
        processed.
    :return: op for training.
    """

    # Variables that affect learning rate.
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    tf.summary.scalar('learning_rate', lr)

    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op
