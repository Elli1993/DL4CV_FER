from utils.data_iterator import iterate_minibatches
from utils.visualize import display_one_image
from models.cnn_models import build_cnn
from modeltrainer.cnn_trainer import train_model
import numpy as np


print ('Launch Training...')

losses = train_model(networkname='cnn', num_epochs=51, batch_size=200)

np.save('loss_history.npy', losses)

print(losses)