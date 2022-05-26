import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from explainnn.dataset import get_dataset

def load_test_image(dataset_name, class_idx, batch_size=1, shuffle=True):  
    """
    Load a sample from the test set

    Inputs
    -------
    dataset_name : {MNIST, CIFAR10}, str
        The name of the sampled dataset

    class_idx : int
        The index of the sampled class 

    batch_size : int, default=1
        The sample size

    shuffle : bool, default=True
        Wether to shuffle the dataset or not
    
    Outputs 
    -------
    sample : torch.tensor of shape (batch_size, n_channels, width, height)
        The sample

    labels : torch.tensor of shape (batch_size)
        The labels corresponding to the sample
    """
    _, test_set = get_dataset(dataset_name)
    loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=shuffle, num_workers=8)
    if hasattr(loader.dataset, 'targets'):
        target_indices = np.where(np.asarray(loader.dataset.targets) == class_idx)[0]
    else:
        targets = [t for _, t in loader.dataset]
        target_indices = np.where(np.asarray(targets) == class_idx)[0]
    sampler = torch.utils.data.sampler.SubsetRandomSampler(target_indices)
    test_loader = DataLoader(dataset=test_set, sampler=sampler, batch_size=batch_size, num_workers=8)
    sample, labels = next(iter(test_loader))
    return sample, labels

def select_easy_test_examples(model, dataset_name, class_idx, n_sample, batch_size, shuffle=False, internal_batch_size=32, device=torch.device('cpu')):   
    _, test_set = get_dataset(dataset_name)
    loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=shuffle, num_workers=8, pin_memory=True)
    if hasattr(loader.dataset, 'targets'):
        target_indices = np.where(np.asarray(loader.dataset.targets) == class_idx)[0]
    else:
        targets = np.asarray([t for _,_, t in loader.dataset])
        target_indices = np.where(np.asarray(targets) == class_idx)[0]

    sampler = torch.utils.data.sampler.SubsetRandomSampler(target_indices)
    test_loader = DataLoader(dataset=test_set, sampler=sampler, batch_size=1, num_workers=8, pin_memory=True)
    samples = torch.Tensor().to(device)
    labels = torch.LongTensor().to(device)
    probs = torch.Tensor().to(device)
    count = 0
    for _, (x, targets) in enumerate(test_loader):   
        with torch.no_grad():
            x, targets = x.to(device), targets.to(device)
            logits = model(x)
            
            out = logits.cpu().numpy() if device.type == "cuda"  else np.asarray(logits)
            
            idx = np.where(np.argmax(out, axis=1) == class_idx)[0]
           
            samples = torch.cat([samples, x[idx]])
            labels = torch.cat([labels, targets[idx]])
            
            probs = torch.cat([probs, F.softmax(logits[idx], dim=1)])
            count += len(idx)
            
            if count >= n_sample:
                break
    probs = probs[:, class_idx].mean()
    print(f"\nModel test accuracy on easy samples of label [{class_idx}] : {(100 * probs):.2f}%")


    return samples, labels


