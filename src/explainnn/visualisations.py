from captum.attr import visualization as viz
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict

def visualize_image_attr(attributions: Dict,
                        input: np.array,
                        ):
    original_image = np.transpose((np.asarray(input.squeeze(0))), (1,2,0))
    if np.max(original_image) > 1:
        original_image = (original_image - np.min(original_image))/(np.max(original_image) - np.min(original_image))

    to_return = dict()
    to_return['origin_image'] = original_image
    original_image = np.transpose((np.asarray(input.squeeze(0))), (1,2,0))
    if np.max(original_image) > 1:
        original_image = (original_image - np.min(original_image))/(np.max(original_image) - np.min(original_image))
    
    for _, k in enumerate(attributions.keys()):
        
        if k == 'attributions':
            features  = np.asarray(attributions[k])
            for i in range(features.shape[0]):
                feat = features[i]
                vmin, vmax = np.min(feat), np.max(feat)
                feat = (feat - vmin) / (vmax - vmin)

                feat = 2 * feat - 1 
                feat = feat[:, :, np.newaxis]
                to_return[attributions['indices'][i]] = feat
    return to_return
