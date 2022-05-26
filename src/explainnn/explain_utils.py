from collections import defaultdict
from copy import deepcopy
import cv2
from functools import partial
import numpy as np
import pickle
from scipy import signal, special
import torch
import torch.nn.functional as F

from explainnn.definitions import SUPPORTED_CLASSIFICATION_LAYERS_NAMES, MAX_DIM, OUTPUT_DIR

def get_structural_module(name, model):
    s = name.split('.')
    if len(s)>1:
        module = model._modules[s[0]]
        for idx in range(1, len(s)):
            module = module._modules[s[idx]]
        return module, s[idx]
    else:
        return model._modules[name], name

def get_last_conv_dims(model, input_dims, device):
    """
    using hook method
    """
    feat_dims = {}
    hdl = []
    def hook(module, input, output, layer_name):
        output = output
        feat_dims[layer_name] = (input[0].shape, output.shape)
    
    x = torch.rand(1, *input_dims).type(torch.FloatTensor).to(device)
    for name, module in model.named_modules():
        hdl.append(module.register_forward_hook(partial(hook, layer_name=name)))
    
    model(x)
    for h in hdl:
        h.remove()
    names = list(feat_dims.keys())
    for k in range(1, len(feat_dims)):
        prev = names[k-1]
        curr = names[k]
        if 'conv' in prev and 'fc' in curr:
            c = feat_dims[prev][1][1]
            d = feat_dims[curr][0][1]
            feat_dims['pre-flattened'] = torch.Size([1, c, int((d//c)**0.5), int((d//c)**0.5)])
        
        elif prev in ['conv', 'features'] and 'fc' in curr:
            c = feat_dims[prev][1][0]
            d = feat_dims[curr][0][1]
            feat_dims['pre-flattened'] = torch.Size([1, c, int((d//c)**0.5), int((d//c)**0.5)])
        
        elif prev in ['conv', 'features'] and 'classifier' in curr:
            feat_dims['pre-flattened'] = feat_dims[prev][1]
        
        elif 'avgpool' in prev and any(x in curr for x in SUPPORTED_CLASSIFICATION_LAYERS_NAMES):
            feat_dims['pre-flattened'] = feat_dims[prev][0]
    return feat_dims

def get_module(model):
    out = dict()
    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.Conv2d):
            out[name] = layer
        if isinstance(layer, torch.nn.Linear):
            out[name] = layer
        if isinstance(layer, torch.nn.BatchNorm2d):
            out[name] = layer
    return out

def get_interventional_weights(layer_wise_params, Ln_1_name, Ln_name, target_neuron_idx, pre_flattened_dims):
    weights = deepcopy(layer_wise_params)
    
    flatten_alpha = False
    rule = 'conv' in Ln_1_name and any(x in Ln_name for x in SUPPORTED_CLASSIFICATION_LAYERS_NAMES)
    
    if rule:
        shape = list(pre_flattened_dims)[1:]
        flatten_alpha = True
    
    for k, idx in enumerate(target_neuron_idx):
        
        w = weights[idx]
        if flatten_alpha:
            w = w.reshape(shape)
        
        if k == 0:
            I_weights = w.unsqueeze(0).clone()
        else:
            I_weights = torch.cat([I_weights, w.unsqueeze(0).clone()], dim=0)
    
    if w.shape[0] > MAX_DIM:
        if len(w.shape)>1:
            w = torch.sum(w, axis=(1,2))
        indices = torch.argsort(w)
        topk = int(len(w) * 0.1)
        smallestk = int(len(w) * 0.05)
        I_indices = indices[len(w)-topk:]
        I_indices = sorted(list(set(torch.cat([I_indices, indices[:smallestk], indices[len(w)//2-smallestk//2:len(w)//2+smallestk//2]])))) 
    else:
        I_indices = []
    return I_weights, flatten_alpha, I_indices

def multiple_interventions_effect(neuron_idx, input, effect_neurons, flatten_alpha, net, L_n_name, weights, u=0.0):
    """
    combine the effect of all neurons of interest defined in effect_neurons -- mediators
    """    
    with torch.no_grad():
        module, _ = get_structural_module(L_n_name, net) 
        initial_weights = module.weight.data.clone()        
        
        for index, edge in enumerate(effect_neurons):
            
            alpha = deepcopy(weights[index])
            alpha[neuron_idx] = u
            
            if flatten_alpha:
                alpha = alpha.flatten()
            
            module.weight.data[edge] = alpha.unsqueeze(0).unsqueeze(0).float()
        
        y_hat = net(input)
        probs = F.softmax(y_hat, dim=1)

        module.weight.data = initial_weights.clone()
        y_hat = y_hat.squeeze().detach().cpu().numpy()
        probs = probs.squeeze().detach().cpu().numpy()
        
        return y_hat, probs  

def save_causal_graph(causal_path, args):
    graph_path = OUTPUT_DIR.joinpath(args['dataset_name'], f'class_{args["target_idx"]}')
    filename = f"{args['model_name']}_causal_graph.pkl"
    if not graph_path.exists():
        graph_path.mkdir(parents=True)
    with open(graph_path.joinpath(filename), 'wb') as f:   
        pickle.dump(causal_path, f)
    print(f"Causal graph saved at {graph_path.joinpath(filename)}")

def check_flattened_layer(child_name, parent_name):
    rule1 = 'conv' in parent_name and any(x in child_name for x in SUPPORTED_CLASSIFICATION_LAYERS_NAMES)
    rule2 = 'feature' in parent_name and 'fc' in child_name #any(x in child_name for x in SUPPORTED_CLASSIFICATION_LAYERS_NAMES)
    if rule1 or rule2:
        flatten_layer = True
    else:
        flatten_layer = False
    return flatten_layer

def get_last_conv_dims(model, input_dims, device):
    """
    using hook method
    """
    feat_dims = {}
    hdl = []
    def hook(module, input, output, layer_name):
        output = output
        feat_dims[layer_name] = (input[0].shape, output.shape)
    
    x = torch.rand(1, *input_dims).type(torch.FloatTensor).to(device)
    for name, module in model.named_modules():
        hdl.append(module.register_forward_hook(partial(hook, layer_name=name)))
    
    model(x)
    for h in hdl:
        h.remove()
    names = list(feat_dims.keys())
    for k in range(1,len(feat_dims)):
        prev = names[k-1]
        curr = names[k]
        if 'conv' in prev and 'fc' in curr:
            c = feat_dims[prev][1][1]
            d = feat_dims[curr][0][1]
            feat_dims['pre-flattened'] = torch.Size([1, c, int((d//c)**0.5), int((d//c)**0.5)])
        
        elif prev in ['conv', 'features'] and 'fc' in curr:
            c = feat_dims[prev][1][0]
            d = feat_dims[curr][0][1]
            feat_dims['pre-flattened'] = torch.Size([1, c, int((d//c)**0.5), int((d//c)**0.5)])
        
        elif prev in ['conv', 'features'] and 'classifier' in curr:
            feat_dims['pre-flattened'] = feat_dims[prev][1]
        
        elif 'avgpool' in prev and any(x in curr for x in SUPPORTED_CLASSIFICATION_LAYERS_NAMES):
            feat_dims['pre-flattened'] = feat_dims[prev][0]
    
    return feat_dims

def adjust_indices_after_flatten(indices, flattened_dims):
    if torch.is_tensor(flattened_dims):
        flattened_dims = flattened_dims[1:].numpy()
    else:
        flattened_dims = flattened_dims[1:]

    filter_shape = flattened_dims[1] * flattened_dims[2]
    adjusted_indices = []
    for idx in indices:
        adjusted_indices.append(torch.arange(idx * filter_shape, idx * filter_shape + filter_shape))
       
    adjusted_indices = torch.stack(adjusted_indices).flatten()    
    return adjusted_indices 

def get_path_specific_visual_attributions(input, layer_activations, weights, causal_path, layer_name, layer_map):
    """
    this function computes the causal attributions from a causal diagram defined by the causal path dictionary
    if the layer_name is None:
        the function will provide causal attributions for all conv layers
    if the layer_name is defined:
        the function will provide the causal attributions for the given layer
        example: layer_name = conv2

    NOTE: for computations and memory, we select to only save features in last conv layer
    """

    output = defaultdict(list)
    layers = list(causal_path.keys())[:-1]    

    if layer_name not in layers:
        raise ValueError("layer name doesnt match names in causal graph, check causal_path dictionary")
    
    if layer_name == 'input':
        return
    
    input_dims = input.squeeze(0).detach().cpu().numpy().shape[1:]

    layers_path_id = layers.index(layer_name)
    index = layer_map[layer_name]

    weights = weights[index]        
    if layers_path_id == len(layers) - 1:
        layers.append("input")

    parent_layer = layers[layers_path_id+1]
    child_layer = layers[layers_path_id]

    child_neurons = torch.tensor(causal_path[layer_name][1])

    parent_neurons = torch.tensor(np.array(causal_path[layer_name][0]))

    
    activations = layer_activations[parent_layer][0]
    responses = layer_activations[child_layer][0]
    features = activations.squeeze(0)[parent_neurons]
    
    features = features.detach().cpu().numpy()
    dims = features.shape
    maps = []

    count = 0
    
    for _, ch in enumerate(child_neurons): 
        count += 1
        neruon_response = responses.squeeze(0)[ch]
        
        channel_weights = weights[ch]
        
        if len(dims) > 1:
            v = []
            w = channel_weights[parent_neurons]
            
            if len(w.shape) > 3:
                raise ValueError("check shape of weights: conv filters have 3 dims")                
            if len(w.shape) == 1:
                top_w_id = np.flip(np.argsort(w))
                K = max(int(len(w) * 0.2), 1)
            else:
                top_w_id = np.flip(np.argsort(np.linalg.norm(w.cpu().numpy(), axis=(1,2))))
                K = max(int((w.shape[0]) * 0.2), 1)  
            K = w.shape[0]
            for i in top_w_id[:K]:
                if len(w.shape) == 1:
                    v.append(features[i,:,:] * w[i])
                else:
                    v.append(torch.tensor(signal.convolve2d(features[i,:,:], w[i,:,:].cpu().numpy())))
            
            v = torch.stack(v)
            
        
            normalized_resp = special.softmax(v, axis=(1,2))

            
            maps.append(normalized_resp.mean(0))    
            output['total_response'].append(neruon_response.detach().cpu().numpy())
            
            attributions = cv2.resize(normalized_resp.mean(0).cpu().numpy(), input_dims)
            output['attributions'].append(attributions)

    
    output['indices'] = child_neurons   
    output['layer'] = layer_name            
    output['root'] = input.squeeze(0).detach().cpu().numpy()    
    return output
        