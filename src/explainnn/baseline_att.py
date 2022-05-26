from captum.attr import GradientShap, IntegratedGradients, GuidedBackprop, DeepLift, LRP, LimeBase, Saliency, GuidedGradCam, LayerGradCam, Occlusion, InputXGradient, Deconvolution
from captum.attr._utils.attribution import LayerAttribution
from torchray.attribution.grad_cam import grad_cam
from torchray.attribution.rise import rise
from torchray.attribution.excitation_backprop import excitation_backprop
from torchray.attribution.extremal_perturbation import extremal_perturbation  
from torchray.attribution.deconvnet import deconvnet

from explainnn.explain_utils import get_structural_module

def int_gradients(model, input, label, **kwargs):
    input.requires_grad = True
    model.zero_grad()
    ig = IntegratedGradients(model)
    attributions = ig.attribute(input, target=label, baselines=input * 0, n_steps=100, internal_batch_size=1)
    return attributions

def guided_backprop(model, input, label, **kwargs):
    input.requires_grad = True
    gb = GuidedBackprop(model)
    attributions = gb.attribute(input, target=label)
    return attributions

def grad_cam_captum(model, input, label, **kwargs):
    input.requires_grad = False
    layer = kwargs.get("layer")
    module, _ = get_structural_module(layer, model) 
    gc = LayerGradCam(model, module)
    attributions = gc.attribute(input, target=label)
    upsampled_attr = LayerAttribution.interpolate(attributions, input.shape[2:])
    print("Upsampled Shape:", upsampled_attr.shape)
    return upsampled_attr

def gradshape(model, input, label, **kwargs):
    input.requires_grad = True
    g = GradientShap(model)
    attributions = g.attribute(inputs=input, target=label, baselines=input * 0)
    return attributions

def grad_cam_torchray(model, input, label, **kwargs):
    input.requires_grad = False
    layer = kwargs.get("layer")
    attributions = grad_cam(model, input, label, saliency_layer=layer, resize=True)
    return attributions

def deep_lift(model, input, label, **kwargs):
    input.requires_grad = True
    dls = DeepLift(model)
    attributions = dls.attribute(input, target=label)
    return attributions

def lrp(model, input, label, **kwargs):
    input.requires_grad = False
    l_r_p = LRP(model)
    attributions = l_r_p.attribute(input, target=label)
    return attributions

def lime(model, input, label, **kwargs):
    input.requires_grad = True
    lm = LimeBase(model)
    attributions = lm.attribute(input, target=label)
    return attributions

def saliency(model, input, label, **kwargs):
    input.requires_grad = True
    s = Saliency(model)
    attributions = s.attribute(input, target=label, abs=True)
    return attributions

def occlusion(model, input, label, **kwargs):
    input.requires_grad = False
    ablator = Occlusion(model)
    attributions = ablator.attribute(input, target=label, strides=(input.shape[1], 8, 8), sliding_window_shapes=(input.shape[1], 15,15))
    return attributions

def rise_method(model, input, label, **kwargs):
    input.requires_grad = False
    attributions = rise(model, input, label)
    return attributions

def gradxinput(model, input, label, **kwargs):
    input.requires_grrad = True
    grad = InputXGradient(model)
    attributions = grad.attribute(input, label)
    return attributions

def deconvolution(model, input, label, **kwargs):
    input.requires_grrad = True
    deconv = Deconvolution(model)
    attributions = deconv.attribute(input, label)
    return attributions

def deconvolution(model, input, label, **kwargs):
    attributions = deconvnet(model, input, label, resize=True)
    return attributions

def MWP(model, input, label, **kwargs):
    layer = kwargs.get("layer")
    attributions = excitation_backprop(model, input, label, saliency_layer=layer, resize=True)
    return attributions

def ext_perturbation(model, input, target, **kwargs):
    if not isinstance(target, int):
        target = int(target.detach().cpu().numpy())
    attributions = extremal_perturbation(model, input, target)
    return attributions

