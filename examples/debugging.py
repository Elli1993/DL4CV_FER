from utils.data_iterator import iterate_minibatches
from utils.visualize import display_one_image
from models.cnn_models import build_cnn
from modeltrainer.cnn_trainer import train_model


print ('Launch Training...')

network = train_model(networkname='cnn', num_epochs=10, batch_size=500)


