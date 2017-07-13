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
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer  #needs GPU support (!use this when training on GPU)
# from lasagne.layers import Conv2DLayer as ConvLayer   #use if you do not have GPU support
from lasagne.layers import MaxPool2DLayer as PoolLayer
from lasagne.layers import LocalResponseNormalization2DLayer as NormLayer
from lasagne.utils import floatX

#Configurations
SAVE_BEST_PARAMS = False
LOAD_PREVIOUS_PARAMS = False #If True: specify previous param file below, if False: std. pretrained net will be loaded
PARAM_FILE_TO_LOAD = 'experiments/' + '2017_07_13_10_52_15_onlyPretrainedNet_BestParams/' +'best_params_0.pkl'



# create new experiment folder
dateTimeOfExperiment = str(datetime.datetime.today().strftime('%Y_%m_%d_%H_%M_%S'))
additionalInfo = '_pretrinedOnly'   #add additional Info about the experiment you are runneing
experimentname = dateTimeOfExperiment + additionalInfo
experimentpath = 'experiments/' + experimentname +'/'
if not os.path.exists(experimentpath):
    os.makedirs(experimentpath)

epoch = 0



#Define the network structure
print 'define network'
net = {}
net['input'] = InputLayer((None, 3, 224, 224))
net['conv1'] = ConvLayer(net['input'], num_filters=96, filter_size=7, stride=2, flip_filters=False)
net['norm1'] = NormLayer(net['conv1'], alpha=0.0001) # caffe has alpha = alpha * pool_size
net['pool1'] = PoolLayer(net['norm1'], pool_size=3, stride=3, ignore_border=False)
net['conv2'] = ConvLayer(net['pool1'], num_filters=256, filter_size=5, flip_filters=False)
net['pool2'] = PoolLayer(net['conv2'], pool_size=2, stride=2, ignore_border=False)
net['conv3'] = ConvLayer(net['pool2'], num_filters=512, filter_size=3, pad=1, flip_filters=False)
net['conv4'] = ConvLayer(net['conv3'], num_filters=512, filter_size=3, pad=1, flip_filters=False)
net['conv5'] = ConvLayer(net['conv4'], num_filters=512, filter_size=3, pad=1, flip_filters=False)
net['pool5'] = PoolLayer(net['conv5'], pool_size=3, stride=3, ignore_border=False)
net['fc6'] = DenseLayer(net['pool5'], num_units=4096)
net['drop6'] = DropoutLayer(net['fc6'], p=0.5)
net['fc7'] = DenseLayer(net['drop6'], num_units=4096)
net['drop7'] = DropoutLayer(net['fc7'], p=0.5)
net['fc8'] = DenseLayer(net['drop7'], num_units=1000, nonlinearity=lasagne.nonlinearities.softmax)
output_layer = net['fc8']






#-------use the following code to test vie images in folder 'testImages'--------
#load pretrained model
print 'load pretrained model'
model = pickle.load(open('vgg_cnn_s.pkl', 'rb'))
CLASSES = model['synset words']
MEAN_IMAGE = model['mean image']


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
    lasagne.layers.set_all_param_values(output_layer, model['values'])

#get images from a folder
def get_image(image_path):
    from scipy import misc
    arr = misc.imread(image_path, mode='RGB')
    return arr

PATH_TO_TESTIMAGE_FOLDER = 'testImages/'
testFileList = os.listdir(PATH_TO_TESTIMAGE_FOLDER)
testFileList[:] = [f for f in testFileList if f.endswith(".jpg")]
#np.random.shuffle(testFileList)

def prep_image_from_folder(image):
    image_path = PATH_TO_TESTIMAGE_FOLDER +image
    im = get_image(image_path)
    # Resize so smallest dim = 256, preserving aspect ratio
    h, w, _ = im.shape
    if h < w:
        im = skimage.transform.resize(im, (256, w * 256 / h), preserve_range=True)
    else:
        im = skimage.transform.resize(im, (h * 256 / w, 256), preserve_range=True)

    # Central crop to 224x224
    h, w, _ = im.shape
    im = im[h // 2 - 112:h // 2 + 112, w // 2 - 112:w // 2 + 112]

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

        fig = plt.figure()
        plt.imshow(rawim.astype('uint8'))
        plt.axis('off')
        for n, label in enumerate(top5):
            plt.text(0, 10 + n * 17, '{}. {}: {}'.format(n+1, "{0:.2f}".format(top5Prob[n]), CLASSES[label]), fontsize=10, backgroundcolor=(1, 1, 1, 0.5), alpha=1)
        #plt.draw()
        fig.savefig(experimentpath + 'predLabels'+str(nrImage)+'.png')
#plt.show()




# save the parameters of the network after that epoch of training
if SAVE_BEST_PARAMS == True:
    params = lasagne.layers.get_all_param_values(output_layer)
    paramname = experimentpath + 'best_params_epoch_' + str(epoch) + '.pkl'
    pickle.dump(params, open(paramname, 'wb'))







#----Use the following code to test on images via url-------

# #get test images from url
# print 'get test images'
# index = urllib.urlopen('http://www.image-net.org/challenges/LSVRC/2012/ori_urls/indexval.html').read()
# image_urls = index.split('<br>')
#
# np.random.seed() #23
# np.random.shuffle(image_urls)
# image_urls = image_urls[:25]
#
# #preprocess images
# def prep_image(url):
#     ext = url.split('.')[-1]
#     im = plt.imread(io.BytesIO(urllib.urlopen(url).read()), ext)
#
#     # Resize so smallest dim = 256, preserving aspect ratio
#     h, w, _ = im.shape
#     if h < w:
#         im = skimage.transform.resize(im, (256, w * 256 / h), preserve_range=True)
#     else:
#         im = skimage.transform.resize(im, (h * 256 / w, 256), preserve_range=True)
#
#     # Central crop to 224x224
#     h, w, _ = im.shape
#     im = im[h // 2 - 112:h // 2 + 112, w // 2 - 112:w // 2 + 112]
#
#     rawim = np.copy(im).astype('uint8')
#
#     # Shuffle axes to c01
#     im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)
#
#     # Convert to BGR
#     im = im[::-1, :, :]
#
#     im = im - MEAN_IMAGE
#     return rawim, floatX(im[np.newaxis])
#
#
# #process test images and print top 5 predicted labels
# print 'process test images and print top 5 predicted labels'
# for nrImage,url in enumerate(image_urls):
#     try:
#         print 'url: ' +url
#         rawim, im = prep_image(url)
#
#
#         prob = np.array(lasagne.layers.get_output(output_layer, im, deterministic=True).eval())
#         top5 = np.argsort(prob[0])[-1:-6:-1]
#         top5Prob = np.sort(prob[0])[-1:-6:-1]
#
#         fig = plt.figure()
#         plt.imshow(rawim.astype('uint8'))
#         plt.axis('off')
#         for n, label in enumerate(top5):
#             plt.text(0, 10 + n * 17, '{}. {}: {}'.format(n+1, "{0:.2f}".format(top5Prob[n]), CLASSES[label]), fontsize=10, backgroundcolor=(1, 1, 1, 0.5), alpha=1)
#         #plt.draw()
#         fig.savefig('prediction'+str(nrImage)+'.png')
#     except IOError:
#         print('bad url: ' + url)
# #plt.show()


print "finished script"