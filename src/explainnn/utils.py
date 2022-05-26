import torch
from yaml import load

from explainnn.definitions import MODEL_DIR
from explainnn.models import LeNet

def load_model(**kwargs):
    """
    Inputs
    -------
    n_classes : int, default=10

    in_channels : int, default=1

    model_name : {LeNet}, str

    dataset_name : {MNIST, CIFAR10}

    device : torch.device, default="cpu"

    Output
    -------
    network : LeNet
    """
    # default values
    n_classes = 10
    in_channels = 1
    model_name = 'LeNet'
    dataset_name = 'MNIST'
    device = torch.device('cpu')

    if 'n_classes' in kwargs.keys():
        n_classes = kwargs['n_classes']
    if 'in_channels' in kwargs.keys():
        in_channels = kwargs['in_channels']
    if 'dataset_name' in kwargs.keys():
        dataset_name = kwargs['dataset_name']
    if 'device' in kwargs.keys():
        device = kwargs['device']
    
    model_path = MODEL_DIR.joinpath(model_name, dataset_name, 'net.pth')
    if not(model_path.exists()):
        raise FileNotFoundError(model_path.resolve())
    network = LeNet(in_channels=in_channels, n_classes=n_classes)
    network.load_state_dict(torch.load(model_path))
    network = network.to(device)
    return network
    

if __name__=='__main__':
    load_model()

