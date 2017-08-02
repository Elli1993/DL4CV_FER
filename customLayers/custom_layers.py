import lasagne
import theano
import numpy as np

class set_zero(lasagne.layers.Layer):
    def get_output_for(self, input, **kwargs):
        zero = theano.tensor.zeros_like(input)
        flat = theano.tensor.set_subtensor(zero.flatten()[input.argmax()], input.max())
        output = theano.tensor.reshape(flat, input.shape)
        return output