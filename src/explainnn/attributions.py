import copy

import torch
import torch.nn.functional as F

from explainnn.explain_utils import get_structural_module, get_last_conv_dims, check_flattened_layer, adjust_indices_after_flatten

def compute_attribution_effect(input, model, causal_graph, verbose=0, out="all", device=torch.device('cpu')):
    """
    causal_attributions: a dictionary -- key: layers, value: causal neuron/s
    model: pre-trained neural network
    input: image input and its label
    """
    number_neurons = dict()

    local_model = copy.deepcopy(model)
    local_model.eval()
    local_model = local_model.to(device)

    if not torch.is_tensor(input):
        input = torch.from_numpy(input)

    with torch.no_grad():

        label = causal_graph['label']

        input = input.to(device)

        orig_yhat = local_model(input)
        orig_pred = F.softmax(orig_yhat, dim=1)
        orig_yhat = orig_yhat.squeeze(0).detach().cpu().numpy()
        orig_pred = orig_pred.squeeze(0).detach().cpu().numpy()

        if verbose > 0:
            print("--------------------------------------")
            print("initial prediction: ", orig_yhat[label], orig_pred[label])
            print("--------------------------------------")
        modules_names = [key.split('.weight')[0] for key, _ in local_model.named_parameters() if 'weight' in key]
        modules_names = list(reversed(modules_names))

        for i, name in enumerate(modules_names):

            module, _ = get_structural_module(name, local_model)
            if not name in causal_graph.keys():
                continue

            original_weights = module.weight.data.clone()

            positive_causes, _, effects = causal_graph[name][0], causal_graph[name][1], causal_graph[name][2]


            if i < len(modules_names) - 1:
                number_neurons[modules_names[i+1]] = len(positive_causes)

            causes = positive_causes
            module.weight.data = torch.tensor(0.0).float() * module.weight.data
            # keeping only positively contribution paths among non-causal and negative paths
            p = 1.0
            if verbose > 0:
                print("causal neurons in current layer ", causes)

            if i < len(modules_names) - 1:
                flatten = check_flattened_layer(name, modules_names[i+1])
            else:
                flatten = False
            if flatten:
                feat_dims = get_last_conv_dims(model, input.shape[1:], device)
                causes = adjust_indices_after_flatten(causes, feat_dims['pre-flattened'])

            for idx_c in range(len(causes)):
                # reached to input
                initial_weight = original_weights[effects].clone()
                coefficients = p * initial_weight[:, causes[idx_c]]
                module.weight.data[effects, causes[idx_c],...] = coefficients


        y_hat = local_model(input)
        pred = F.softmax(y_hat, dim=1)

    del local_model
    torch.cuda.empty_cache()

    y_hat = y_hat.squeeze(0).detach().cpu().numpy()
    pred = pred.squeeze(0).detach().cpu().numpy()

    if verbose > 0:
        print("intervened prediction: ", y_hat[label], pred[label])

    if out == "all":
        return [y_hat, orig_yhat]
    elif out == "softmax":
        return pred
    elif out == "logits":
        return orig_yhat