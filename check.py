from utils.data_iterator import iterate_minibatches
from utils.visualize import display_one_image
from utils.dataset import load_fer


fer = load_fer(0, one_hot=True, flat=False)

for i, j in iterate_minibatches(fer['data'], fer['target'], batchsize=200, shuffle=True):
    print (i.shape)
    print (j.shape)

display_one_image(i[0], flattend=False, save=False, file_name=None)

