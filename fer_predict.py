import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import io
import skimage.transform
import urllib
import os
import matplotlib.patheffects as PathEffects
# import pickle
import cPickle as pickle
import datetime
import time

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
PARAM_FILE_TO_LOAD = 'experiments/' + '2017_07_28_13_01_14_2PretrainedConv+Dense_augmented'+ '/bestParams/' +'best_params_epoch_08.pkl'



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

#get images from a folder
def get_image(image_path):
    from scipy import misc
    arr = misc.imread(image_path, mode='RGB')
    return arr

PATH_TO_TESTIMAGE_FOLDER = 'testImages/'
testFileList = os.listdir(PATH_TO_TESTIMAGE_FOLDER)
testFileList[:] = [f for f in testFileList if f.endswith(".jpg") or f.endswith(".png")]
#np.random.shuffle(testFileList)

def prep_image_from_folder(image):
    image_path = PATH_TO_TESTIMAGE_FOLDER +image
    im = get_image(image_path)
    # Resize so smallest dim = 256, preserving aspect ratio
    h, w, _ = im.shape
    if h < w:
        im = skimage.transform.resize(im, (48, w * 48 / h), preserve_range=True)
    else:
        im = skimage.transform.resize(im, (h * 48 / w, 48), preserve_range=True)

    # Central crop to 48x48
    h, w, _ = im.shape
    im = im[h // 2 - 24:h // 2 + 24, w // 2 - 24:w // 2 + 24]

    rawim = np.copy(im).astype('uint8')

    # Shuffle axes to c01
    im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)

    # Convert to BGR
    im = im[::-1, :, :]

    im = im - MEAN_IMAGE
    return rawim, floatX(im[np.newaxis])

for nrImage,image in enumerate(testFileList):
        print 'image: ' +image
        rawim, im = prep_image_from_folder(image)

        prob = np.array(lasagne.layers.get_output(output_layer, im, deterministic=True).eval())
        top5 = np.argsort(prob[0])[-1:-6:-1]
        top5Prob = np.sort(prob[0])[-1:-6:-1]

        image_path = PATH_TO_TESTIMAGE_FOLDER + image
        image_originalSize = get_image(image_path)
        fig = plt.figure()
        plt.imshow(image_originalSize.astype('uint8'))
        plt.axis('off')
        for n, label in enumerate(top5):
            plt.text(0, 10 + n * 17, '{}. {}: {}'.format(n+1, "{0:.2f}".format(top5Prob[n]), CLASSES[label]), fontsize=10, backgroundcolor=(1, 1, 1, 0.5), alpha=1)
        #plt.draw()
        fig.savefig(experimentpath + 'predLabels'+str(nrImage)+'.png')
#plt.show()

print "finished script"