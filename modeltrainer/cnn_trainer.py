import theano
import lasagne
from utils.dataset import load_fer
import theano.tensor as T
import sys
import time
from utils.data_iterator import iterate_minibatches
import numpy as np
from models.cnn_models import build_cnn
import cPickle as pickle

from models.cnn_models import build_shallow_cnn
from models.pretrained_models import build_vgg_cnn

theano.config.floatX = 'float64'

def train_model(networkname = None, num_epochs = 10, batch_size = 200):
    weight_decay =1e-6
    losses = {}
    training_loss_history = []
    validation_loss_history = []
    input_var = T.tensor4('inputs', dtype=theano.config.floatX)
    target_var = T.ivector('targets')

    print('starting to create network...')

    if networkname.startswith('cnn'):
        print('load fer data')
        train_data = load_fer(0, one_hot=False, flat=False)
        val_data = load_fer(1, one_hot=False, flat=False)
        test_data = load_fer(2, one_hot=False, flat=False)

        print ('creating network')
        network = build_shallow_cnn(input_var=input_var)
    elif networkname.startswith('vgg'):
        print('load fer data')
        train_data = load_fer(0, one_hot=False, flat=False, expand=True)
        val_data = load_fer(1, one_hot=False, flat=False, expand=True)
        test_data = load_fer(2, one_hot=False, flat=False, expand=True)

        print ('creating network')
        network = build_vgg_cnn(input_var=input_var, name_pretrained_model='vgg16.pkl')
    else:
        print('no correct network provided')

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    # We could add some weight decay as well here, see lasagne.regularization.
    weightsl2 = lasagne.regularization.regularize_network_params(network, lasagne.regularization.l2)
    loss += weight_decay * weightsl2
    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    #updates = lasagne.updates.nesterov_momentum(
    #    loss, params, learning_rate=0.001, momentum=0.9)
    updates = lasagne.updates.adam(loss, params, learning_rate=0.001)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(train_data['data'], train_data['target'], batch_size, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1
            if (train_batches%5) == 0:
                print('Batchnumber {} done'.format(train_batches))

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(val_data['data'], val_data['target'], 200, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))
        training_loss_history.append(train_err/train_batches)
        validation_loss_history.append(val_err / val_batches)

        if (epoch % 5) == 0:
            print ('saving network')
            params = lasagne.layers.get_all_param_values(network)
            paramname = 'shallow_cnn_trained' + str(epoch) + '.pkl'
            pickle.dump(params, open(paramname, 'wb'))
            print('network saved')

    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(test_data['data'], test_data['target'], 500, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))

    #np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
    # Optionally, you could now dump the network weights to a file like this:
    # np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
    #
    # And load them again later on like this:
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)
    losses['train_loss'] = training_loss_history
    losses['val_loss'] = validation_loss_history



    return losses


