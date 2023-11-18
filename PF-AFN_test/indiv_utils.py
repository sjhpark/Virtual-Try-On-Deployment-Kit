import yaml
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader
import os

def load_yaml(filename):
    # LOAD YAML FILE
    with open(f'config/{filename}.yaml','r') as f:
        output = yaml.safe_load(f)
    return output

def get_layers(model):
    # reference: https://saturncloud.io/blog/pytorch-get-all-layers-of-model-a-comprehensive-guide/
    layers = []
    for name, module in model.named_children():
        if isinstance(module, nn.Sequential):
            layers += get_layers(module)
        elif isinstance(module, nn.ModuleList):
            for m in module:
                layers += get_layers(m)
        else:
            layers.append(module)
    layers = nn.ModuleList(layers)
    return layers

def check_buffers(model):
    """Check buffers in a model.
    After unstructured-pruning the model, you will notice that the model size on disk is doubled after pruning.
    This is because mask buffers (to replace weights with zeros) are stored in addition to the original parameters."""
    print(f"Number of buffers in {model.__class__.__name__}: {len(list(model.named_buffers()))}")
    print(f"Buffers in {model.__class__.__name__}:\n{list(model.named_buffers())}")

def remove_masks(layer):
    """
    Unstructured pruning creates mask buffers to replace weights with zeros. Thus, the model size on disk is doubled after unstructured pruning.
    This function convert an unstructured-pruned model to a sparse model by removing reparametrization (mask buffers).
    The model size on disk will be reduced to the original size.
    Currently, this function only works for weights of Conv2d layers.
    Reference for Unstructured Pruning: https://stackoverflow.com/questions/62326683/prunning-model-doesnt-improve-inference-speed-or-reduce-model-size"""
    # Remove reparameterization (mask buffers)
    if isinstance(layer, nn.Conv2d):
        prune.remove(layer, 'weight')

class Sparsity():
    """
    Compute sparsity of layers in a model.
    Currently, this class only works for weights of Conv2d layers.
    """
    def __init__(self, layers):
        self.layers = layers
    
    def each_layer(self):
        for layer_and_name in self.layers: # layer_and_name is a tuple. Ex) (nn.Conv2d, 'weight') or (nn.Conv2d, 'bias')
            layer = layer_and_name
            # if Conv2d layer
            if isinstance(layer, nn.Conv2d):
                sparisty = torch.sum(layer.weight == 0)
                num_elements = layer.weight.nelement()
                print(f"Sparsity of {layer.__class__.__name__}: {100. * float(sparisty) / float(num_elements)}%")

def global_unstructured_pruning(layers2prune, sparsity_level=0.33):
    """Global Unstructured Pruning"""
    prune.global_unstructured(layers2prune, pruning_method=prune.L1Unstructured, amount=sparsity_level)

def unstructured_pruning(layer, sparsity_level=0.33):
    """Unstructured Pruning"""
    layer = layer[0] # layer[0]: layer itself, layer[1]: layer name such as 'weight' or 'bias'
    if isinstance(layer, nn.Conv2d):
        prune.l1_unstructured(layer, name='weight', amount=sparsity_level)

def custom_filter_pruning(all_layers, layer_idx, filter_idx):
    """Remove a specific filter from a Conv2d layer"""
    # remove a specific filter from a Conv2d layer
    layer = all_layers[layer_idx] 
    layer = layer[0] # layer[0]: layer itself, layer[1]: layer name such as 'weight' or 'bias'

    assert isinstance(layer, nn.Conv2d), "layer must be a Conv2d layer"
    current_weight = layer.weight.data.clone() # shape (out_channels, in_channels, kernel_size[0], kernel_size[1])
    assert filter_idx < current_weight.shape[0], "filter_idx must be smaller than the number of input filters in the conv layer"
    new_weight = torch.cat((current_weight[:filter_idx,:,:,:], current_weight[filter_idx+1:,:,:,:]), 0)
    layer.weight.data = new_weight

    if isinstance(all_layers[layer_idx + 1][0], nn.BatchNorm2d):
        next_bn_layer = all_layers[layer_idx + 1][0]
        current_weight_bn = next_bn_layer.weight.data.clone()
        new_weight_bn = torch.cat((current_weight_bn[:filter_idx], current_weight_bn[filter_idx+1:]), 0)
        next_bn_layer.weight.data = new_weight_bn

        # update running_mean and running_var accordingly
        current_running_mean = next_bn_layer.running_mean.clone()
        new_running_mean = torch.cat((current_running_mean[:filter_idx], current_running_mean[filter_idx+1:]), 0)
        next_bn_layer.running_mean = new_running_mean

        current_running_var = next_bn_layer.running_var.clone()
        new_running_var = torch.cat((current_running_var[:filter_idx], current_running_var[filter_idx+1:]), 0)
        next_bn_layer.running_var = new_running_var

        # update bias accordingly
        current_bias = next_bn_layer.bias.data.clone()
        new_bias = torch.cat((current_bias[:filter_idx], current_bias[filter_idx+1:]), 0)
        next_bn_layer.bias.data = nn.Parameter(new_bias)
    
    if isinstance(all_layers[layer_idx + 2][0], nn.Conv2d):
        next_conv_layer = all_layers[layer_idx + 2][0]
        current_weight_conv = next_conv_layer.weight.data.clone()
        new_weight_conv = torch.cat((current_weight_conv[:,:filter_idx,:,:], current_weight_conv[:,filter_idx+1:,:,:]), 1)
        next_conv_layer.weight.data = new_weight_conv

def size_on_disk(model):
    dir = 'out'
    if not os.path.exists(dir):
        os.makedirs(dir)
    if isinstance(model, dict): # if model is a state dict
        torch.save(model, f"{dir}/temp.p")
        size = os.path.getsize(f"{dir}/temp.p")
        print(f"Model Size on Disk: {size/1e6} MB")
        os.remove(f"{dir}/temp.p")
    elif isinstance(model, nn.Module): # if model is a nn.Module
        torch.save(model.state_dict(), f"{dir}/temp.p")
        size = os.path.getsize(f"{dir}/temp.p")
        print(f"Model Size on Disk: {size/1e6} MB")
        os.remove(f"{dir}/temp.p")
    return size/1e6

def measure_inference_latency(model, test_dataset, device, warmup_itr):
    config = load_yaml('config')
    device = config['device']
    device = torch.device(device)
    print(f"Measuring inference latency of trained {model.__class__.__name__} on {device}...")
    test_dataloader =  DataLoader(test_dataset, batch_size=1, shuffle=False)
    with torch.no_grad():
        inference_latency = []
        for i, data in tqdm(enumerate(test_dataloader)):
            features, _ = data
            features = features.to(device)
            # WARM_UP
            if i == 0:
                print("Warm-up begins...")
                for _ in range(warmup_itr):
                    _ = model(features)
            # MEASURE INFERENCE LATENCY    
            begin = time.time()
            _ = model(features)
            end = time.time()
            inference_latency.append((end - begin)/features.shape[0])
    mean_inference_latency = np.mean(inference_latency)*1000
    print(f"Mean inference latency: {mean_inference_latency:.3f}ms")
    # plot inference latency over iterations and save it as a figure
    if not os.path.exists("out"):
        os.makedirs("out")
    plt.figure(figsize=(12, 8))
    plt.plot(inference_latency)
    plt.scatter(range(len(inference_latency)), inference_latency, s=10, c='r')
    plt.title(f"Inference Latency vs. Iterations in Test Loop\nMean inference latency: {mean_inference_latency:.3f}ms")
    plt.xlabel("Iteration")
    plt.ylabel("Inference latency [s]")
    plt.savefig(f"out/{model.__class__.__name__}_inference_latency.png")
    plt.close()

def param_count(model):
    # COUNT THE NUMBER OF PARAMETERS
    param_dict = {}
    param_count = 0
    for name, param in model.named_parameters():
        zeros = torch.sum(param == 0) # number of zero-mask (usually for pruning) in each weight tensor
        count = param.numel() - zeros
        param_dict[name] = count
        param_count += count
    print(f"Total Parmeter Count in {model.__class__.__name__}: {param_count}")
    # for key, value in sorted(param_dict.items(), key=lambda item: item[1]):
    #     print(f"\t{key}:\t{value}")
    return param_count.item()

def save_model_weights(model, fname):
    if not os.path.exists("out"):
        os.makedirs("out")
    torch.save(model.state_dict(), f"out/{model.__class__.__name__}_weights_{fname}.pth")
    print(f"Saved {model.__class__.__name__}'s weights as out/{model.__class__.__name__}_weights_{fname}.pth.")
