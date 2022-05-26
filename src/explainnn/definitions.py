from pathlib import Path
import quantus


ROOT_DIR = Path(__file__).parent.parent.parent
SRC_DIR = ROOT_DIR.joinpath('src')
DATASET_DIR = ROOT_DIR.joinpath('dataset')
MODEL_DIR = ROOT_DIR.joinpath('model')
OUTPUT_DIR = ROOT_DIR.joinpath('output')

CONFIG_FILE = ROOT_DIR.joinpath('config.yaml')

SUPPORTED_DATASET = ['MNIST', 'CIFAR10']
SUPPORTED_MODEL = ['LeNet']
SUPPORTED_CLASSIFICATION_LAYERS_NAMES = {'fc', 'classifier'}
MAX_DIM = 600

ATTR_DATASET = {
    'MNIST' : {
        'imsize' : 32,
        'n_channels' : 1,
        'n_classes' : 10
    },
    'CIFAR10' : {
        'imsize' : 32,
        'n_channels' : 3,
        'n_classes' : 10
    }
}


METRICS_TYPE = quantus.available_categories() 
METRICS = quantus.available_metrics() 

STD = [0.05, 0.1, 0.2]
MEAN = [0.1, 0.2, 0.3]