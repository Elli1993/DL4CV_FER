import caffe
import numpy as np
import cPickle as pickle

import theano
import theano.tensor as T
import lasagne
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer, NonlinearityLayer, FlattenLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.utils import floatX

net_caffe = caffe.Net('models/vgg_face/VGG_FACE_deploy.prototxt', 'VGG_FACE.caffemodel', caffe.TEST)


net = {}
net['input'] = InputLayer((None, 3, 224, 224))
net['conv1_1'] = ConvLayer(net['input'], num_filters=64, filter_size=3, pad=1, flip_filters=False)
net['conv1_2'] = ConvLayer(net['conv1_1'], num_filters=64, filter_size=3, pad=1, flip_filters=False)
net['pool1'] = PoolLayer(net['conv1_2'], pool_size=2, stride=2, mode='max', ignore_border=False)
net['conv2_1'] = ConvLayer(net['pool1'], num_filters=128, filter_size=3, pad=1, flip_filters=False)
net['conv2_2'] = ConvLayer(net['conv2_1'], num_filters=128, filter_size=3, pad=1, flip_filters=False)
net['pool2'] = PoolLayer(net['conv2_2'], pool_size=2, stride=2, mode='max', ignore_border=False)
net['conv3_1'] = ConvLayer(net['pool2'], num_filters=256, filter_size=3, pad=1, flip_filters=False)
net['conv3_2'] = ConvLayer(net['conv3_1'], num_filters=256, filter_size=3, pad=1, flip_filters=False)
net['conv3_3'] = ConvLayer(net['conv3_2'], num_filters=256, filter_size=3, pad=1, flip_filters=False)
net['pool3'] = PoolLayer(net['conv3_3'], pool_size=2, stride=2, mode='max', ignore_border=False)
net['conv4_1'] = ConvLayer(net['pool3'], num_filters=512, filter_size=3, pad=1, flip_filters=False)
net['conv4_2'] = ConvLayer(net['conv4_1'], num_filters=512, filter_size=3, pad=1, flip_filters=False)
net['conv4_3'] = ConvLayer(net['conv4_2'], num_filters=512, filter_size=3, pad=1, flip_filters=False)
net['pool4'] = PoolLayer(net['conv4_3'], pool_size=2, stride=2, mode='max', ignore_border=False)
net['conv5_1'] = ConvLayer(net['pool4'], num_filters=512, filter_size=3, pad=1, flip_filters=False)
net['conv5_2'] = ConvLayer(net['conv5_1'], num_filters=512, filter_size=3, pad=1, flip_filters=False)
net['conv5_3'] = ConvLayer(net['conv5_2'], num_filters=512, filter_size=3, pad=1, flip_filters=False)
net['pool5'] = PoolLayer(net['conv5_3'], pool_size=2, stride=2, mode='max', ignore_border=False)
net['fc6'] = DenseLayer(net['pool5'], num_units=4096)
net['drop6'] = DropoutLayer(net['fc6'], p=0.5)
net['fc7'] = DenseLayer(net['fc6'], num_units=4096)
net['drop7'] = DropoutLayer(net['fc7'], p=0.5)
net['fc8'] = DenseLayer(net['fc7'], num_units=2622, nonlinearity=None)
net['prob'] = NonlinearityLayer(net['fc8'], lasagne.nonlinearities.softmax)
net['output'] = lasagne.layers.FlattenLayer(net['prob'])


layers_caffe = dict(zip(list(net_caffe._layer_names), net_caffe.layers))

for name, layer in net.items():
    try:
        layer.W.set_value(layers_caffe[name].blobs[0].data)
        layer.b.set_value(layers_caffe[name].blobs[1].data)
    except AttributeError:
        continue












