import numpy as np
import pandas as pd
import pickle
import quantus
from quantus.metrics.robustness_metrics import LocalLipschitzEstimate
from quantus.metrics.faithfulness_metrics import IterativeRemovalOfFeatures
from quantus.helpers import utils, asserts
"""
This code includes modified materials from quantus to run evaluation metrics on our method
"""
import re
import time
import torch
from tqdm import tqdm

import explainnn.baseline_att as batt
from explainnn.definitions import OUTPUT_DIR, METRICS_TYPE
from explainnn.explain import ExplainNN
class CausalLocalLipschitzEstimate(LocalLipschitzEstimate):
    
    def __init__(self, **kwargs):
        super(LocalLipschitzEstimate, self).__init__(**kwargs)
        self.nr_samples = self.kwargs.get("nr_samples")
        self.perturb_func = self.kwargs.get("perturb_func")
        self.norm_numerator = self.kwargs.get("norm_numerator")
        self.norm_denominator = self.kwargs.get("norm_denominator")
        self.perturb_std = self.kwargs.get("perturb_std")
        self.perturb_mean = self.kwargs.get("perturb_mean")
        self.similarity_func = self.kwargs.get("similarity_func")
        self.layer = kwargs.get("layer")
        self.abs = self.kwargs.get("abs")
        self.normalise = self.kwargs.get("normalise")
        self.all_results = []
        self.last_results = []
        
    def normalise_fun(self, a):
        if np.max(a) <= 0.0:
            a = - a / np.min(a)
        elif np.min(a) >= 0.0:
            a = a/np.max(a)
        else:
            a =   (a > 0.0) * a / np.max(a) - (a < 0.0) * a / np.min(a)
        return a

    def causal_explanation(self, args, model, x_batch, x_batch_tensor, causal_graph, device):
        
        ex = ExplainNN(model, args, beta=0, device=device)
        
        # Reshape input batch to channel first order:
        self.channel_first = self.kwargs.get("channel_first", utils.infer_channel_first(x_batch))
        x_batch_s = utils.make_channel_first(x_batch, self.channel_first)
        
        # Wrap the model into an interface
        if model:
            model = utils.get_wrapped_model(model, self.channel_first)

        # Update kwargs.
        self.nr_channels = self.kwargs.get("nr_channels", np.shape(x_batch_s)[1])
        self.img_size = self.kwargs.get("img_size", np.shape(x_batch_s)[-1])
        self.kwargs = {
            **{k: v for k, v in self.__dict__.items() if k not in ["args", "kwargs"]},
        }
        
        self.last_result = []
        similarity_max = 0.0
        for _, (x, x_tensor) in enumerate(zip(x_batch_s, x_batch_tensor)):
            x_shape = x.shape
            
            start = time.time()
            a = ex.extract_attributions(x_tensor.unsqueeze(0), self.layer, causal_graph) 
            print("time elabsed: ", time.time() - start)
            a = np.asarray(a['attributions']).sum(axis=0)
            
            if self.abs:
                a = np.abs(a)
            
            if self.normalise:
                a = self.normalise_fun(a)
            
            
            for i in range(self.nr_samples):
                # Perturb input.
        
                x_perturbed = self.perturb_func(arr=x.flatten(), **self.kwargs)
                x_perturbed = model.shape_input(x_perturbed, x_shape, channel_first=True)
                
                asserts.assert_perturbation_caused_change(x=x, x_perturbed=x_perturbed)
                
                a_perturbed = ex.extract_attributions(x_perturbed, self.layer, causal_graph)
                a_perturbed = np.asarray(a_perturbed['attributions']).sum(axis=0)
                if self.abs:
                    a_perturbed = np.abs(a_perturbed)
                
                if self.normalise:
                    a_perturbed = self.normalise_fun(a_perturbed)
                
                # Measure similarity.
                similarity = self.similarity_func(
                    a=a.flatten(),
                    b=a_perturbed.flatten(),
                    c=x.flatten(),
                    d=x_perturbed.flatten(),
                    **self.kwargs,
                )
                
                if similarity > similarity_max:
                    similarity_max = similarity
                    
            # Append similarity score.
            self.last_results.append(similarity_max)
            

        self.all_results.append(self.last_results)
        
       
        return self.last_results


class CausalIROF(IterativeRemovalOfFeatures):
    
    def __init__(self, **kwargs):
        super(IterativeRemovalOfFeatures, self).__init__(**kwargs)
        self.kwargs = kwargs
        self.perturb_baseline = self.kwargs.get("perturb_baseline")
        self.perturb_func = self.kwargs.get("perturb_func")
        self.segmentation_method = self.kwargs.get("segmentation_method")
        self.abs = self.kwargs.get("abs")
        self.layer = self.kwargs.get("layer")
        self.normalise = self.kwargs.get("normalise")
        self.pos_only = False # default in quantus
        self.last_results = []
        self.all_results = []
    
    def normalise_fun(self, a):
        if np.max(a) <= 0.0:
            a = -a / np.min(a) 
        elif np.min(a) >= 0.0:
            a = a/np.max(a)
        else:
            a =   (a > 0.0) * a / np.max(a) - (a < 0.0) * a / np.min(a)

        return a

    def causal_explanation(self, args, model, x_batch, x_batch_tensor, y_batch, causal_graph, device, **kwargs):
        ex = ExplainNN(model, args, beta=0, device=device) 
        
        # Reshape input batch to channel first order:
        self.channel_first = self.kwargs.get("channel_first", utils.infer_channel_first(x_batch))
        x_batch_s = utils.make_channel_first(x_batch, self.channel_first)
        
        # Wrap the model into an interface
        if model:
            model = utils.get_wrapped_model(model, self.channel_first)

        # Update kwargs.
        self.nr_channels = np.shape(x_batch_s)[1]
        
        self.kwargs = {
            **kwargs,
            **{k: v for k, v in self.__dict__.items() if k not in ["args", "kwargs"]},
        }
        self.last_result = []

        for _, (x, x_tensor, y) in tqdm(enumerate(zip(x_batch_s, x_batch_tensor, y_batch)), total=len(x_batch_s)):
            x_shape = x.shape
            a = ex.extract_attributions(x_tensor.unsqueeze(0), self.layer, causal_graph)  
            
            a = np.asarray(a['attributions']).sum(axis=0)
            if self.pos_only:
                a[a < 0] = 0.0

            if x_tensor.shape[0] == 3:
                x_tensor = x_tensor[np.newaxis,...]
            
            # added this to check dims, can be improved
            a = a[np.newaxis, :, :]
            a = utils.expand_attribution_channel(a, x_tensor)
            
            if self.abs:
                a = np.abs(a)
            if self.normalise:
                a = self.normalise_fun(a)

            asserts.assert_attributions(x_batch=x_tensor, a_batch=a)
            if len(a.shape)>3:
                a = a.squeeze(0)
            x_tensor = x_tensor.squeeze(0)
            
            # Predict on input.
            x_input = model.shape_input(x, x_shape, channel_first=True)   
            y_pred = float(
                model.predict(x_input, softmax_act=True, **self.kwargs)[:, y]
            )
            
            # Segment image.
            segments = utils.get_superpixel_segments(
                img=np.moveaxis(x, 0, -1).astype("double"),
                segmentation_method=self.segmentation_method,
            )
            nr_segments = segments.max()
            asserts.assert_nr_segments(nr_segments=nr_segments)
            # Calculate average attribution of each segment.
            att_segs = np.zeros(nr_segments)
            for i, s in enumerate(range(nr_segments)):
                att_segs[i] = np.mean(a[:,segments == s])
            # Sort segments based on the mean attribution (descending order).
            s_indices = np.argsort(-att_segs)
            preds = []
            
            for _, s_ix in enumerate(s_indices):
                 # Perturb input by indices of attributions.
                a_ix = np.nonzero(np.repeat((segments == s_ix).flatten(), self.nr_channels))[0]

                x_perturbed = self.perturb_func(
                    arr=x_input.flatten(),
                    indices=a_ix,
                    **self.kwargs,
                )
                asserts.assert_perturbation_caused_change(x=x, x_perturbed=x_perturbed)
                # Predict on perturbed input x.
                x_input = model.shape_input(x_perturbed, x_shape, channel_first=True)
                y_pred_perturb = float(
                    model.predict(x_input, softmax_act=True, **self.kwargs)[:, y]
                )
                preds.append(float(y_pred_perturb / y_pred))
            
            self.last_result.append(np.trapz(np.array(preds), dx=1.0))
        self.last_result = [np.mean(self.last_result)]       
        self.all_results.append(self.last_result)
        
        return self.last_result


class EvaluationMetrics:
    def __init__(self, attrs, model, inputs, targets, filename, args, masks=None, device=torch.device('cpu')):

        torch.cuda.empty_cache()
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
                
        self.attrs = attrs
        self.inputs = inputs
        self.device = device
        self.targets = targets
        self.masks = torch.Tensor().to(device) if masks is None else masks
        
        batch_size = 16
        if len(inputs) > batch_size:
            chunks = len(inputs) // batch_size
            self.inputs, self.targets = torch.split(self.inputs, chunks), torch.split(self.targets, chunks)
            self.masks = torch.split(self.masks, chunks) if len(self.masks) > 0 else self.masks

        self.args = args
        self.outdir = OUTPUT_DIR
        self.internal_batch_size = 16
        
        with open(filename, 'rb') as f:
            self.causal_graph = pickle.load(f)
        
        layer_names = list(self.causal_graph.keys())
        layer_names = [name for name in layer_names if re.search('conv', name) or re.search('feature', name)]
        self.layer = layer_names[0]
        
        
        self.model = model.to(device)
        self.forward_func = args['model_name']

        self.outdir = self.outdir.joinpath('evaluation_metrics', self.forward_func)
        self.outdir.mkdir(parents=True, exist_ok=True)
        print(f"\nModel test accuracy on this class : {(100 * self.compute_predictions()):.2f}%")

        self.evaluate_attrs()

    def compute_predictions(self,):
        logits = torch.Tensor().to(self.device)
        gt = torch.LongTensor().to(self.device)
       
        for inputs_batch, targets_batch in zip(self.inputs, self.targets):
            with torch.no_grad():
                x, label = inputs_batch.to(self.device), targets_batch.to(self.device)
                logits = torch.cat([logits, self.model(x)])
                gt = torch.cat([gt, label])
        
        pred = torch.mean((torch.argmax(logits, axis=1) == gt)*1.0)
        return pred

    
    def _to_numpy(self, batch):
        return batch.detach().cpu().numpy()

    def evaluate_attrs(self,):
        
        r_df = f_df = pd.DataFrame({})
        
        for _, (inputs_batch, targets_batch) in enumerate(zip(self.inputs, self.targets)):

            x_batch = self._to_numpy(inputs_batch)
            y_batch = self._to_numpy(targets_batch)
        
            outputs = []
            methods = []
            
            if 'Robustness' in METRICS_TYPE:
                
                for name in self.attrs:
                    
                    print("running: ", name)
                    if name in ['Occlusion', 'RISE', 'FeatureAblation', 'GuidedBackProp', 'Deconvolution', 'ExcitationBackProp', 'ExtremalPeurt']: continue
                    metric = quantus.LocalLipschitzEstimate(**{
                                                    "nr_samples": 10,
                                                    "perturb_std": 0.1,
                                                    "perturb_mean": 0.1,
                                                    "norm_numerator": quantus.distance_euclidean,
                                                    "norm_denominator": quantus.distance_euclidean,    
                                                    "perturb_func": quantus.gaussian_noise,
                                                    "similarity_func": quantus.lipschitz_constant,
                                                    "normalise": False,
                                                    "disable_warnings": True
                                                })(model=self.model, 
                                                x_batch=x_batch,
                                                y_batch=y_batch,
                                                a_batch=None, 
                                                **{"explain_func": quantus.explain, "method": name, "device": self.device, "internal_batch_size": self.internal_batch_size})
                    
                    
                    outputs = outputs + metric
                    methods = methods + [name] * len(metric)
                
                
                metric = CausalLocalLipschitzEstimate(**{
                                                    "nr_samples": 10,
                                                    "perturb_std": 0.1,
                                                    "perturb_mean": 0.1,
                                                    "norm_numerator": quantus.distance_euclidean,
                                                    "norm_denominator": quantus.distance_euclidean,    
                                                    "perturb_func": quantus.gaussian_noise,
                                                    "similarity_func": quantus.lipschitz_constant,
                                                    "layer": self.layer,
                                                    "abs": True,
                                                    "normalise": True,
                                                    "disable_warnings": True
                                                }).causal_explanation(
                                                    args=self.args,
                                                    model=self.model,
                                                    x_batch=x_batch,
                                                    x_batch_tensor=inputs_batch,
                                                    causal_graph=self.causal_graph,
                                                    device=self.device)
                    
                outputs = outputs + metric
                methods = methods + ["causal"] * len(metric)
                
                d = dict()
                d['method'] = methods
                d['metric'] = outputs

                r_df = pd.concat([r_df, pd.DataFrame.from_dict(d, orient='index').transpose()], ignore_index=True)
                        
            
            if 'Faithfulness' in METRICS_TYPE:
                # ===============================================
                # compute IROF faithfulness correaltion metric
                outputs = []
                methods = []
                
                metric = CausalIROF(**{"perturb_func": quantus.baseline_replacement_by_indices,
                                        "perturb_baseline": "mean", 
                                        "segmentation_method": "slic", 
                                        "layer": self.layer,
                                        "abs": True,
                                        "normalise": True,
                                        "disable_warnings": True
                                        }).causal_explanation(
                                                    args=self.args,
                                                    model=self.model,
                                                    x_batch=x_batch,
                                                    x_batch_tensor=inputs_batch,
                                                    y_batch=y_batch,
                                                    causal_graph=self.causal_graph,
                                                    device=self.device)
                outputs = outputs + metric
                methods = methods + ["Causal"] * len(metric)
            
                d = dict()
                d['method'] = methods
                d['metric'] = outputs

                f_df = pd.concat([f_df, pd.DataFrame.from_dict(d, orient='index').transpose()], ignore_index=True)
                
        # ==================================================
        #save results to csv
        if len(r_df) > 0: r_df.to_csv(self.outdir.joinpath('LipschitzEstimate_metric.csv')); print(f'File saved at {self.outdir.joinpath("LipschitzEstimate_metric.csv")}')
        if len(f_df) > 0: f_df.to_csv(self.outdir.joinpath('IORF_metric.csv')); print(f'File saved at {self.outdir.joinpath("IORF_metric.csv")}')
