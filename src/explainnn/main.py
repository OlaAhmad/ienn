import re

import matplotlib.pyplot as plt
import pickle
import sys
import torch
import yaml

import explainnn.baseline_att as batt
from explainnn.dataloader import load_test_image, select_easy_test_examples
from explainnn.definitions import OUTPUT_DIR, CONFIG_FILE, ATTR_DATASET
from explainnn.evaluation_metrics import EvaluationMetrics
from explainnn.explain import ExplainNN
from explainnn.utils import load_model
from explainnn.visualisations import visualize_image_attr


def get_causal_connections(args, model, device=torch.device('cpu'), verbose=False):
    """
    Compute the causal graph for the model specified in args
    
    Inputs
    --------
    model : torch.nn.Module
        Instance of the model specified in config.yaml

    args : dict
        Parameters specified in config.yaml

    beta : float, default=0.0
        Parameter for the soft constraint

    device : torch.device, default='cpu'

    verbose : bool

    Outputs
    --------
    causal_path : dict

    explainar : ExplainNN

    """
    explainar = ExplainNN(model, args, beta=0, device=device, verbose=verbose)
    if args['learn_explainer']:
        if verbose:
            print("-*-"*10 + "\n\tGenerating explanations\n" + "-*-"*10)
        causal_path = explainar.run()
    else:
        if verbose:
            print("-*-"*10 + "\n\t Loading explanations\n" + "-*-"*10)
        saved_graph_filename = OUTPUT_DIR.joinpath(args['dataset_name'], f'class_{args["target_idx"]}', f"{args['model_name']}_causal_graph.pkl")
        try:           
            with open(saved_graph_filename, 'rb') as f:
                causal_path = pickle.load(f)
        except FileNotFoundError:
            print(f"Couldn't find the causal graph at {saved_graph_filename}")
            sys.exit(1)
    return causal_path, explainar

def get_sample_explanations(args, input, causal_path, explainar):
    """
    Compute the attributions for a causal graph on an image

    Inputs
    -------
    args : dict
        Parameters specified in config.yaml

    input : torch.tensor of shape (imsize, imsize)
        The picture to compute attributions

    causal_path : dict
        The causal path for the model on the class of input

    explainar : ExplainNN

    Outputs
    --------
    attributions : dict
    """

    assert len(causal_path) > 0, "Check causal model for this object exists otherwise run get_causal_connections"

    x = input
    attributions = []
    layer_names = list(causal_path.keys())
    layer_name = args['layer_name']
    if layer_name not in layer_names:
        raise ValueError(f'"{layer_name}" not in causal graph keys. Valid keys are : {", ".join(layer_names)} ')
    layer_names = [name for name in layer_names if re.search('conv', name) or re.search('feature', name)]
    
    layer_names = layer_name
    attributions = explainar.extract_attributions(x, layer_names, causal_path)

    return attributions

def visualize_attributions(args, input, attributions=None, device=torch.device('cpu')):
    """
    Scale attribution

    Inputs
    -------
    args : dict
    
    input : torch.tensor of shape (n_channels, imsize, imsize)
    
    attributions : dict

    Outputs
    --------
    vis : dict
        The attributions for each neuron

    """
    if attributions is None:
        attributions_path = OUTPUT_DIR.joinpath(args['dataset_name'], f'class_{args["target_idx"]}', 'instance_path_specific_attributions.pkl')
        try:
            with open(attributions_path, 'rb') as f:
                attributions = pickle.load(f)
        except FileNotFoundError:
            print(f"Couldn't find the attribution file at {attributions_path}")
        print(f"Reading attributions from : {attributions_path}")
        input = attributions['root']
    return visualize_image_attr(attributions, input) 

def evaluate_explanations(args, model, device=torch.device('cpu')):
    """
    Evaluate the explanations for some baseline methods :
    'IntegratedGradients', 'Saliency', 'GradientShap', 'InputXGradient', 'RISE'

    Inputs
    -------
    args : dict

    model : torch.nn.Module

    device : torch.device, default='cpu'
    """

    batch_size = 100
    n_sample = args['n_samples']
    attr_list = ['IntegratedGradients', 'Saliency', 'GradientShap', 'InputXGradient', 'RISE']
    
    inputs, labels = select_easy_test_examples(model, args['dataset_name'], args['target_idx'], n_sample=n_sample, batch_size=batch_size, device=device)
    filename = OUTPUT_DIR.joinpath(args['dataset_name'], f'class_{args["target_idx"]}', f"{args['model_name']}_causal_graph.pkl")
    EvaluationMetrics(attr_list, model, inputs, labels, filename, args, device=device)

def compute_baseline_attr(args, model, input, device=torch.device('cpu')):
    """
    Computes the attributions for baseline methods on an image

    Inputs
    -------
    args : dict

    model : torch.nn.Module

    input : torch.tensor of shape (n_channels, imsize, imsize)

    device : torch.device, default='cpu'

    Outputs
    --------
    attrs : dict
        The attributions for each method
    """
    attrs = {}
    attr_methods = {'IG': batt.int_gradients, 
                    'GB': batt.guided_backprop, 
                    'Deconv': batt.deconvolution,
                    'Saliency':batt.saliency, 
                    'InputXGradient': batt.gradxinput
                    }
    x = input.clone()
    for name in list(attr_methods.keys()):
        attributions = attr_methods[name](model, x.to(device), args['target_idx'])
        attrs[name] = attributions[0]
    return attrs

def graph_stability(args, model, causal_graph_in, verbose=False, device=torch.device('cpu')):
    """
    Generates causal graphs with a beta parameter
    ranging from 0 to 0.5 with a 0.01 step

    Inputs
    -------
    args : dict

    model : torch.nn.Module

    causal_graph_in : dict

    verbose : bool, default=False

    device : torch.device, default='cpu'

    Outputs
    -------
    res : dict
        Dictionary containing all the causal graph generated 
    """
    beta_arr = torch.arange(0.,0.51,0.01)
    layer_name = args['layer_name_soft']
    res = {}
    for beta in beta_arr:
        if verbose:
            print(f'Beta : {beta}')
        tmp = []
        for _ in range(10):
            explainar = ExplainNN(model, args, beta, device=device)
            causal_path = explainar.cross_sample_path_effect_analysis_layer(explainar.global_info_dict['layers_index'], causal_graph_in, layer_name, soft=True)
            if layer_name in causal_path.keys():
                tmp.append(causal_path[layer_name])
            else:
                tmp.append([])
        res[beta] = tmp
    return res

def main():
    with open(CONFIG_FILE, 'r') as f:
       args = yaml.load(f, Loader=yaml.FullLoader)
    device = torch.device(args['device'])
    verbose = args['verbose']
    model = load_model( 
        model_name=args['model_name'],
        dataset_name=args['dataset_name'],
        n_classes=ATTR_DATASET[args['dataset_name']]['n_classes'],
        in_channels=ATTR_DATASET[args['dataset_name']]['n_channels'],
        device=device
        )
    causal_path, explainer = get_causal_connections(args, model, device=device, verbose=verbose)
    test_image, _ = load_test_image(args['dataset_name'], args['target_idx'])
    
    if args["gen_attr"]:
        attributions = get_sample_explanations(args, test_image, causal_path, explainer)
        if args['save_attr']:
                with open(OUTPUT_DIR.joinpath(args['dataset_name'], f'class_{args["target_idx"]}', 'instance_path_specific_attributions.pkl'), 'wb') as f:
                    pickle.dump(attributions, f)
    else:
        attributions = None

    if args['graph_stab']:
        args['soft_interventions'] = True
        args['learn_explainer'] = True
        res = graph_stability(args, model, causal_path, verbose=True, device=device)
        with open(OUTPUT_DIR.joinpath(args['dataset_name'], f'class_{args["target_idx"]}', 'stability.pkl'), 'wb') as f:
            pickle.dump(res, f)

    if args['vis_attr']:
        vis = visualize_attributions(args, test_image, attributions=attributions)
        fig1, ax1 = plt.subplots(1, len(vis.keys())-1, figsize=((len(vis.keys())-1)*3,3))
        i=0
        for k in vis.keys():
            if k != 'origin_image':
                ax1[i].axis('off')
                ax1[i].set_title(f'Node {k}')
                ax1[i].imshow(vis[k], cmap='seismic')
                i+=1
            fig1.suptitle(f'Visualization of the attributions for each node of layer {args["layer_name"]} using causal graph')

    if args['eval_attr']:
        evaluate_explanations(args, model, device=device)

    if args['baseline_attr']:
        baseline_attr = compute_baseline_attr(args, model, test_image,  device=device)
        fig2, ax2 = plt.subplots(1, len(baseline_attr.keys()), figsize=((len(baseline_attr.keys()))*3,3))
        for i, k in enumerate(baseline_attr.keys()):
            ax2[i].imshow(baseline_attr[k].detach().squeeze().numpy(), cmap='seismic')
            ax2[i].axis('off')
            ax2[i].set_title(k)
        fig2.suptitle(f'Visualization of the attributions using baseline methods')
    
    if args['vis_attr'] or args['baseline_attr']:
        plt.tight_layout()
        plt.show()

if __name__== "__main__":
    main()