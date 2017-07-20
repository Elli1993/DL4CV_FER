import numpy as np
import io
import urllib
import os
import sys
import time
import cPickle as pickle

from utils.dataset import load_fer
from utils.data_iterator import iterate_minibatches

import theano
import theano.tensor as T
import lasagne
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer, NonlinearityLayer
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import MaxPool2DLayer as MaxPoolLayer
from lasagne.layers import dropout as dropout

theano.config.floatX = 'float64'

# def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
#     assert len(inputs) == len(targets)
#     if shuffle:
#         indices = np.arange(len(inputs))
#         np.random.shuffle(indices)
#     for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
#         if shuffle:
#             excerpt = indices[start_idx:start_idx + batchsize]
#         else:
#             excerpt = slice(start_idx, start_idx + batchsize)
#         yield inputs[excerpt], targets[excerpt]

#load FER data
def load_dataset():
    # FER load data
    fer_train = load_fer(0, one_hot=False, flat=False, expand=True)
    fer_valid = load_fer(1, one_hot=False, flat=False, expand=True)
    fer_test = load_fer(2, one_hot=False, flat=False, expand=True)

    #data
    X_train = fer_train['data'] / np.float32(256)
    X_val = fer_valid['data'] / np.float32(256)
    X_test = fer_test['data'] / np.float32(256)

    #targets
    y_train = fer_train['target']
    y_val = fer_valid['target']
    y_test = fer_test['target']

    #comment in if you want to use only a subset of the available data
    nrSamplesTrain = 500
    nrSamplesVal = 200
    nrSamplesTest = 100
    X_train, y_train = X_train[:nrSamplesTrain], y_train[:nrSamplesTrain]
    X_val, y_val = X_val[:nrSamplesVal], y_val[:nrSamplesVal]
    X_test, y_test = X_test[:nrSamplesTest], y_test[:nrSamplesTest]

    # We just return all the arrays in order, as expected in main().
    return X_train, y_train, X_val, y_val, X_test, y_test

# Build model VGG16 architecture
def build_model_vgg16():
    net = {}
    net['input'] = InputLayer((None, 3, 224, 224))
    net['conv1_1'] = ConvLayer(net['input'], 64, 3, pad=1, flip_filters=False)
    net['conv1_2'] = ConvLayer(net['conv1_1'], 64, 3, pad=1, flip_filters=False)
    net['pool1'] = MaxPoolLayer(net['conv1_2'], 2)
    net['conv2_1'] = ConvLayer(net['pool1'], 128, 3, pad=1, flip_filters=False)
    net['conv2_2'] = ConvLayer(net['conv2_1'], 128, 3, pad=1, flip_filters=False)
    net['pool2'] = MaxPoolLayer(net['conv2_2'], 2)
    net['conv3_1'] = ConvLayer(net['pool2'], 256, 3, pad=1, flip_filters=False)
    net['conv3_2'] = ConvLayer(net['conv3_1'], 256, 3, pad=1,   flip_filters=False)
    net['conv3_3'] = ConvLayer(net['conv3_2'], 256, 3, pad=1, flip_filters=False)
    net['pool3'] = MaxPoolLayer(net['conv3_3'], 2)
    net['conv4_1'] = ConvLayer(net['pool3'], 512, 3, pad=1, flip_filters=False)
    net['conv4_2'] = ConvLayer(net['conv4_1'], 512, 3, pad=1, flip_filters=False)
    net['conv4_3'] = ConvLayer(net['conv4_2'], 512, 3, pad=1, flip_filters=False)
    net['pool4'] = MaxPoolLayer(net['conv4_3'], 2)
    net['conv5_1'] = ConvLayer(net['pool4'], 512, 3, pad=1, flip_filters=False)
    net['conv5_2'] = ConvLayer(net['conv5_1'], 512, 3, pad=1, flip_filters=False)
    net['conv5_3'] = ConvLayer(net['conv5_2'], 512, 3, pad=1, flip_filters=False)
    net['pool5'] = MaxPoolLayer(net['conv5_3'], 2)
    net['fc6'] = DenseLayer(net['pool5'], num_units=4096)
    net['fc6_dropout'] = DropoutLayer(net['fc6'], p=0.5)
    net['fc7'] = DenseLayer(net['fc6_dropout'], num_units=4096)
    net['fc7_dropout'] = DropoutLayer(net['fc7'], p=0.5)
    net['fc8'] = DenseLayer(net['fc7_dropout'], num_units=1000, nonlinearity=None)
    net['prob'] = NonlinearityLayer(net['fc8'], lasagne.nonlinearities.softmax)
    net['output_layer'] = net['prob']
    return net

# FC network
def build_fc_net(input_var=None):
    # Input : imagees in 3 channels
    l_in = InputLayer(shape=(None, 3, 48, 48),input_var=input_var)

    # droupout of 20%
    l_in_drop = DropoutLayer(l_in, p=0.2)

    l_hid1 = DenseLayer(l_in_drop, num_units=800, nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())

    # dropout of 50%:
    l_hid1_drop = DropoutLayer(l_hid1, p=0.5)

    l_hid2 = DenseLayer(l_hid1_drop, num_units=800, nonlinearity=lasagne.nonlinearities.rectify)

    # 50% dropout again:
    l_hid2_drop = DropoutLayer(l_hid2, p=0.5)

    l_out = DenseLayer(l_hid2_drop, num_units=7, nonlinearity=lasagne.nonlinearities.softmax)

    return l_out

# Build network for FER with layers from pretrained network VGG
def build_cnn(input_var, pretrained_model):
    #pretrained layers from vgg16
    conv1_1 = pretrained_model['conv1_1']
    conv1_2 = pretrained_model['conv1_2']

    #new layers
    network = InputLayer(shape=(None, 3, 48, 48), input_var=input_var)

    network = ConvLayer(network, 64, 3, pad=1, flip_filters=False, W=conv1_1.W.get_value(), b=conv1_1.b.get_value())

    network = ConvLayer(network, 64, 3, pad=1, flip_filters=False, W=conv1_2.W.get_value(), b=conv1_2.b.get_value())

    network = MaxPoolLayer(network, pool_size=(2, 2))

    network = DenseLayer(dropout(network, p=.5), num_units=256, nonlinearity=lasagne.nonlinearities.rectify)

    network = DenseLayer(dropout(network, p=.5), num_units=7, nonlinearity=lasagne.nonlinearities.softmax)

    return network



########################################### Main ###########################################

####### Load pretrained model #######
name_pretrained_model = 'vgg16.pkl'
print 'load pretrained model: ', name_pretrained_model
model = pickle.load(open('models/'+name_pretrained_model, 'rb'))
#MEAN_IMAGE = np.array([np.full((224, 224), model['mean value'][0]), np.full((224, 224), model['mean value'][1]), np.full((224, 224), model['mean value'][2])])
CLASSES = model['synset words']

# Set pretrained model values
print 'set pretrained model values of', name_pretrained_model
pretrained_model = build_model_vgg16()
lasagne.layers.set_all_param_values(pretrained_model['output_layer'], model['param values'])

####### Load the dataset #######
print("Loading data...")
X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

# Prepare Theano variables for inputs and targets
input_var = T.tensor4('inputs', dtype=theano.config.floatX)
target_var = T.ivector('targets')

####### Create neural network model to train on #######
print("Building model and compiling functions...")
# network = build_fc_net(input_var) # Simple FC network
network = build_cnn(input_var, pretrained_model) # Build cnn model with pretrained model and new layers


######### Training Phase #########
# Training params
num_epochs = 10
learning_rate = 0.01
momentum = 0.9


batchsize_training = 100
batchsize_validation = 50
batchsize_test = 50


# Create a loss expression for training, i.e., a scalar objective we want to minimize
# (for our multi-class problem, it is the cross-entropy loss):
prediction = lasagne.layers.get_output(network)
loss = lasagne.objectives.categorical_crossentropy(prediction, target_var) #categorical cross-entropy loss
loss = loss.mean()
# We could add some weight decay as well here, see lasagne.regularization.

# Create update expressions for training, i.e., how to modify the parameters at each training step.
# Here, we'll use Stochastic Gradient Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
params = lasagne.layers.get_all_params(network, trainable=True)
updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=learning_rate, momentum=momentum)

# Create a loss expression for validation/testing. The crucial difference here is that we do a deterministic forward pass through the network, disabling dropout layers.
test_prediction = lasagne.layers.get_output(network, deterministic=True)

test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
test_loss = test_loss.mean()

# As a bonus, also create an expression for the classification accuracy:
test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var), dtype=theano.config.floatX)

# Compile a function performing a training step on a mini-batch (by giving the updates dictionary) and returning the corresponding training loss:
train_fn = theano.function([input_var, target_var], loss, updates=updates)

# Compile a second function computing the validation loss and accuracy:
val_fn = theano.function([input_var, target_var], [test_loss, test_acc])


############## Finally, launch the training loop ##############
print("Starting training...")
# We iterate over epochs:
for epoch in range(num_epochs):

    # In each epoch, we do a full pass over the training data:
    train_err = 0
    train_batches = 0
    start_time = time.time()
    for batch in iterate_minibatches(X_train, y_train, batchsize_training, shuffle=True):
        inputs, targets = batch
        train_err += train_fn(inputs, targets)
        train_batches += 1

    # And a full pass over the validation data:
    val_err = 0
    val_acc = 0
    val_batches = 0
    for batch in iterate_minibatches(X_val, y_val, batchsize_validation, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        val_err += err
        val_acc += acc
        val_batches += 1

    # Then we print the results for this epoch:
    print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
    print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
    print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
    print("  validation accuracy:\t\t{:.2f} %".format(val_acc / val_batches * 100))

# After training, we compute and print the test error:
test_err = 0
test_acc = 0
test_batches = 0
for batch in iterate_minibatches(X_test, y_test, batchsize_test, shuffle=False):
    inputs, targets = batch
    err, acc = val_fn(inputs, targets)
    test_err += err
    test_acc += acc
    test_batches += 1
print ''
print("Final results:")
print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
print("  test accuracy:\t\t{:.2f} %".format(test_acc / test_batches * 100))

print 'end!'