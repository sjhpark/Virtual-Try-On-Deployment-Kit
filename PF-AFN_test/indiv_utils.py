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
    After pruning the model, You will notice that the model size on disk is doubled after pruning.
    This is because mask buffers are stored in addition to the original parameters."""
    print(f"Number of buffers in {model.__class__.__name__}: {len(list(model.named_buffers()))}")
    print(f"Buffers in {model.__class__.__name__}:\n{list(model.named_buffers())}")

def sparse_representation(model, layers):
    """Convert a pruned model to a sparse model by removing reparametrization (mask buffers).
    Currently, this function only works for weights of Conv2d layers."""
    # Remove reparameterization (mask buffers)
    for layer in layers:
        if isinstance(layer, nn.Conv2d):
            prune.remove(layer, 'weight')
    
    # Convert parameters to sparse representation
    sd = model.state_dict() # state dict
    for item in sd:
        sd[item] = model.state_dict()[item].to_sparse()
    return sd

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
