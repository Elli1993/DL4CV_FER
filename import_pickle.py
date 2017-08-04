import cPickle as pickle
import numpy as np
import csv
import scipy.misc

mean_image = np.load('utils/mean.npz')
mean_image = mean_image['mean']
mean_image = np.reshape(mean_image, (48, 48))

list = []
with open('index_layer_conv1_1_fer.csv','rb') as file:
    spamreader = csv.reader(file, delimiter=' ', quotechar='|')

    for row in spamreader:
        list.append(int(row[0]))

counts = np.bincount(np.asarray(list))
key = counts.argmax()
indexlist = np.where(np.asarray(list) == key)

with open('deconv_layer_conv1_1_fer.pkl', 'rb') as infile:
    result = pickle.load(infile)

for i in indexlist[0]:
    image = result[i]
    image = image.reshape(3, 48, 48)
    image += mean_image.mean()
    image = np.clip(image, 0, 255)
    image = np.swapaxes(image, 0, 2)
    scipy.misc.imsave('outfile.jpg', image)


pass