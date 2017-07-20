import numpy as np
import theano
import theano.tensor as T
# from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
import lasagne
import cPickle as pickle

# build model VGG16 architecture
def build_model_vgg16():
    net = {}
    net['input'] = lasagne.layers.InputLayer((None, 3, 224, 224))
    net['conv1_1'] = ConvLayer(net['input'], 64, 3, pad=1, flip_filters=False)
    net['conv1_2'] = ConvLayer(net['conv1_1'], 64, 3, pad=1, flip_filters=False)
    net['pool1'] = lasagne.layers.MaxPool2DLayer(net['conv1_2'], 2)
    net['conv2_1'] = ConvLayer(net['pool1'], 128, 3, pad=1, flip_filters=False)
    net['conv2_2'] = ConvLayer(net['conv2_1'], 128, 3, pad=1, flip_filters=False)
    net['pool2'] = lasagne.layers.MaxPool2DLayer(net['conv2_2'], 2)
    net['conv3_1'] = ConvLayer(net['pool2'], 256, 3, pad=1, flip_filters=False)
    net['conv3_2'] = ConvLayer(net['conv3_1'], 256, 3, pad=1,   flip_filters=False)
    net['conv3_3'] = ConvLayer(net['conv3_2'], 256, 3, pad=1, flip_filters=False)
    net['pool3'] = lasagne.layers.MaxPool2DLayer(net['conv3_3'], 2)
    net['conv4_1'] = ConvLayer(net['pool3'], 512, 3, pad=1, flip_filters=False)
    net['conv4_2'] = ConvLayer(net['conv4_1'], 512, 3, pad=1, flip_filters=False)
    net['conv4_3'] = ConvLayer(net['conv4_2'], 512, 3, pad=1, flip_filters=False)
    net['pool4'] = lasagne.layers.MaxPool2DLayer(net['conv4_3'], 2)
    net['conv5_1'] = ConvLayer(net['pool4'], 512, 3, pad=1, flip_filters=False)
    net['conv5_2'] = ConvLayer(net['conv5_1'], 512, 3, pad=1, flip_filters=False)
    net['conv5_3'] = ConvLayer(net['conv5_2'], 512, 3, pad=1, flip_filters=False)
    net['pool5'] = lasagne.layers.MaxPool2DLayer(net['conv5_3'], 2)
    net['fc6'] = lasagne.layers.DenseLayer(net['pool5'], num_units=4096)
    net['fc6_dropout'] = lasagne.layers.DropoutLayer(net['fc6'], p=0.5)
    net['fc7'] = lasagne.layers.DenseLayer(net['fc6_dropout'], num_units=4096)
    net['fc7_dropout'] = lasagne.layers.DropoutLayer(net['fc7'], p=0.5)
    net['fc8'] = lasagne.layers.DenseLayer(net['fc7_dropout'], num_units=1000, nonlinearity=None)
    net['prob'] = lasagne.layers.NonlinearityLayer(net['fc8'], lasagne.nonlinearities.softmax)
    net['output_layer'] = net['prob']
    return net


# Build network for FER with layers from pretrained network VGG
def build_vgg_cnn(input_var, name_pretrained_model):
    # load pretrained model
    print 'load pretrained model: ', name_pretrained_model
    model = pickle.load(open('models/' + name_pretrained_model, 'rb'))
    # MEAN_IMAGE = np.array([np.full((224, 224), model['mean value'][0]), np.full((224, 224), model['mean value'][1]), np.full((224, 224), model['mean value'][2])])

    # Set pretrained model values
    print 'set pretrained model values of', name_pretrained_model
    pretrained_model = build_model_vgg16()
    lasagne.layers.set_all_param_values(pretrained_model['output_layer'], model['param values'])

    #pretrained layers from vgg16
    conv1_1 = pretrained_model['conv1_1']
    conv1_2 = pretrained_model['conv1_2']
    conv2_1 = pretrained_model['conv2_1']
    conv2_2 = pretrained_model['conv2_2']

    # pretrained layers
    network = lasagne.layers.InputLayer(shape=(None, 3, 48, 48), input_var=input_var)

    network = ConvLayer(network, 64, 3, pad=1, flip_filters=False, W=conv1_1.W.get_value(), b=conv1_1.b.get_value())

    network = ConvLayer(network, 64, 3, pad=1, flip_filters=False, W=conv1_2.W.get_value(), b=conv1_2.b.get_value())

    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    network = ConvLayer(network, 128, 3, pad=1, flip_filters=False, W=conv2_1.W.get_value(), b=conv2_1.b.get_value())
    network = ConvLayer(network, 128, 3, pad=1, flip_filters=False, W=conv2_2.W.get_value(), b=conv2_2.b.get_value())
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2,2))

    # new layers
    network = lasagne.layers.batch_norm(ConvLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity= lasagne.nonlinearities.rectify,
            flip_filters=False,
            W=lasagne.init.GlorotUniform()))

    network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network, p=.5), num_units=256, nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network, p=.5), num_units=7, nonlinearity=lasagne.nonlinearities.softmax)

    return network
