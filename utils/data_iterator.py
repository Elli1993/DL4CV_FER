import numpy as np
from utils.dataset import  load_fer

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    if inputs == None or targets == None:
        inputs , targets = load_fer(0, True, False).values()
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]