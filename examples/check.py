from utils.data_iterator import iterate_minibatches
from utils.visualize import display_one_image
from utils.dataset import load_fer

#this is an example of how to use the iterator and to load the dataset

fer = load_fer(0, one_hot=True, flat=False, expand=False)
#loading the training data (0 is for training, 1 for validation and 2 for test
fer['data'] = fer['data'][:500]
fer['target'] = fer['target'][:500]
#clipping the training set to the first 500 images - for overfitting

for data, target in iterate_minibatches(fer['data'], fer['target'], batchsize=200, shuffle=True):
    #do one epoch of training in here!
    print (data.shape)
    print (target.shape)

## the display image function only works for grayscale images - not the expanded ones
# but there should be some around in your code :)
display_one_image(data[0], save=False, file_name=None)

