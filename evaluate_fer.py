import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import io
import urllib
import os
import matplotlib.patheffects as PathEffects
# import pickle
import cPickle as pickle
import datetime
import time
from utils.dataset import load_fer
from utils.data_iterator import iterate_minibatches
from utils.visualize import display_one_image

import theano
import lasagne
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer
# from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer  #needs GPU support (!use this when training on GPU)
from lasagne.layers import Conv2DLayer as ConvLayer   #use if you do not have GPU support
from lasagne.layers import MaxPool2DLayer as PoolLayer
from lasagne.layers import LocalResponseNormalization2DLayer as NormLayer
from lasagne.utils import floatX

#Configurations
SAVE_BEST_PARAMS = False
LOAD_PREVIOUS_PARAMS = False #If True: specify previous param file below, if False: std. pretrained net will be loaded
PARAM_FILE_TO_LOAD = 'best_params_epoch_12.pkl'


# create new experiment folder
dateTimeOfExperiment = str(datetime.datetime.today().strftime('%Y_%m_%d_%H_%M_%S'))
additionalInfo = '_prediction'   #add additional Info about the experiment you are runneing
experimentname = dateTimeOfExperiment + additionalInfo
experimentpath = 'experiments/' + experimentname +'/'
if not os.path.exists(experimentpath):
    os.makedirs(experimentpath)

epoch = 0



#Define the network structure
print 'define network'
# pretrained layers
network ={}
network['input'] = lasagne.layers.InputLayer(shape=(None, 3, 48, 48))
network['pre_conv1_1'] = ConvLayer(network['input'], 64, 3, pad=1, flip_filters=False)
network['pre_conv1_2'] = ConvLayer(network['pre_conv1_1'], 64, 3, pad=1, flip_filters=False)
network['pre_pool1'] = lasagne.layers.MaxPool2DLayer(network['pre_conv1_2'], pool_size=(2, 2))
network['pre_conv1_2'] = ConvLayer(network['pre_pool1'], 128, 3, pad=1, flip_filters=False)
network['pre_conv2_2'] = ConvLayer(network['pre_conv1_2'], 128, 3, pad=1, flip_filters=False)
network['pre_pool2'] = lasagne.layers.MaxPool2DLayer(network['pre_conv2_2'], pool_size=(2, 2))
# new layers
network['add_batch_norm'] = lasagne.layers.batch_norm(ConvLayer(network['pre_pool2'], num_filters=32, filter_size=(5, 5), nonlinearity=lasagne.nonlinearities.rectify, flip_filters=False, W=lasagne.init.GlorotUniform()))
network['add_dense1'] = lasagne.layers.DenseLayer(lasagne.layers.dropout(network['add_batch_norm'], p=.5), num_units=256, nonlinearity=lasagne.nonlinearities.rectify)
network['add_dense2'] = lasagne.layers.DenseLayer(lasagne.layers.dropout(network['add_dense1'], p=.5), num_units=7,nonlinearity=lasagne.nonlinearities.softmax)
output_layer = network['add_dense2']


#-------use the following code to test vie images in folder 'testImages'--------
#load pretrained model
print 'load pretrained model'
model = pickle.load(open(PARAM_FILE_TO_LOAD, 'rb'))
CLASSES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
MEAN_IMAGE = np.load('utils/mean.npz')
MEAN_IMAGE = MEAN_IMAGE['mean']


# Function to read the pickle file with the network learnt parameters
def load_network_params(layer, filename):
    with open(filename, 'rb') as f:
        while True:
            try:
                # Load pickle file that contains network parameters
                network_parameters = pickle.load(f)
            except EOFError:
                break
    lasagne.layers.set_all_param_values(layer, network_parameters)

if LOAD_PREVIOUS_PARAMS == True:
    #set params of previous experiment
    load_network_params(output_layer,PARAM_FILE_TO_LOAD)
else:
    #set std. pretrained model values
    print 'set pretrained model values'
    lasagne.layers.set_all_param_values(output_layer, model)

test_data = load_fer(2, True, False, True, False, True)
org_data = load_fer(2, False, False, False, False, False)
pos_counts = np.zeros(7)
neg_counts = np.zeros(7)
emotions = []



i = 0
for image, target in iterate_minibatches(test_data['data'], test_data['target'], 1, False):
    input = image
    prob = np.array(lasagne.layers.get_output(output_layer, image, deterministic=True).eval())
    predict = np.argmax(prob)
    target = np.argmax(target)
    if predict == target:
        pos_counts[predict] +=1
    else:
        neg_counts[target] +=1
        emotions.append((i, target, predict))
        display_one_image(org_data['data'][i], True, 'predicted{}_right{}'.format(predict,target))
    i += 1

with open('rightpred.pkl', 'wb') as outfile:
    pickle.dump(pos_counts, outfile, pickle.HIGHEST_PROTOCOL)

with open('wrongpred.pkl', 'wb') as outfile:
    pickle.dump(neg_counts, outfile, pickle.HIGHEST_PROTOCOL)

with open('indices.pkl', 'wb') as outfile:
    pickle.dump(emotions, outfile, pickle.HIGHEST_PROTOCOL)


print "finished script"