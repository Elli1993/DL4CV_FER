import lasagne
import theano
import numpy as np

class set_zero(lasagne.layers.Layer):
    def __init__(self, incoming, number=None, **kwargs):
        super(set_zero, self).__init__(incoming, **kwargs)
        self.number = number

    def get_output_for(self, input, number=None, **kwargs):
        if self.number:
            zero = theano.tensor.zeros_like(input)
            output = theano.tensor.set_subtensor(zero[:, self.number, :, :], input[:, self.number, :, :])
            return output
        else:
            return input