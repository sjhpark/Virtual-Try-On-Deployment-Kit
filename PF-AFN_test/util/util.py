from __future__ import print_function

import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import os
import cv2
import torch
from torch.autograd import Variable
import numpy as np

def tensor2im(image_tensor, imtype=np.uint8, normalize=True):
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy
    image_numpy = image_tensor.cpu().float().numpy()

    image_numpy = (image_numpy + 1) / 2.0
    image_numpy = np.clip(image_numpy, 0, 1)
    if image_numpy.shape[2] == 1 or image_numpy.shape[2] > 3:        
        image_numpy = image_numpy[:,:,0]

    return image_numpy

def tensor2label(label_tensor, n_label, imtype=np.uint8):
    if n_label == 0:
        return tensor2im(label_tensor, imtype)
    label_tensor = label_tensor.cpu().float()    
    if label_tensor.size()[0] > 1:
        label_tensor = label_tensor.max(0, keepdim=True)[1]
    label_tensor = Colorize(n_label)(label_tensor)
    label_numpy = label_tensor.numpy()
    label_numpy = label_numpy / 255.0

    return label_numpy

def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count-1, -1, -1)])

def labelcolormap(N):
    if N == 35: # cityscape
        cmap = np.array([(  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (111, 74,  0), ( 81,  0, 81),
                     (128, 64,128), (244, 35,232), (250,170,160), (230,150,140), ( 70, 70, 70), (102,102,156), (190,153,153),
                     (180,165,180), (150,100,100), (150,120, 90), (153,153,153), (153,153,153), (250,170, 30), (220,220,  0),
                     (107,142, 35), (152,251,152), ( 70,130,180), (220, 20, 60), (255,  0,  0), (  0,  0,142), (  0,  0, 70),
                     (  0, 60,100), (  0,  0, 90), (  0,  0,110), (  0, 80,100), (  0,  0,230), (119, 11, 32), (  0,  0,142)], 
                     dtype=np.uint8)
    else:
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7-j))
                g = g ^ (np.uint8(str_id[-2]) << (7-j))
                b = b ^ (np.uint8(str_id[-3]) << (7-j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
    return cmap

class Colorize(object):
    def __init__(self, n=35):
        self.cmap = labelcolormap(n)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image[0]).cpu()
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image

########################## COUNT THE NUMBER OF PARAMETERS ##########################

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
    return layers

def params_count(model):
    num_params = sum(params.numel() for params in model.parameters())
    print(f"The total number of parameters in {model.__class__.__name__}: {num_params:,}")

########################### SSIM (Structural Similarity) ###########################
def compute_SSIM(gt_images, gen_images):
    # Reference: https://github.com/OFA-Sys/DAFlow/blob/main/utils/test_ssim.py#L11

    SSIMS =[]
    import util.pytorch_ssim as pytorch_ssim
    ssim_loss = pytorch_ssim.SSIM(window_size = 11)

    for i, (gt_image, gen_image) in enumerate(zip(gt_images, gen_images)):
        img1 = cv2.imread(gt_image)
        img2 = cv2.imread(gen_image)

        # Split the image into 3 equal parts and get the 3rd part
        img1 = img1[:, img1.shape[1]//3*2:, :]
        img2 = img2[:, img2.shape[1]//3*2:, :]

        img1 = torch.from_numpy(np.rollaxis(img1, 2)).float().unsqueeze(0)/255.0
        img2 = torch.from_numpy(np.rollaxis(img2, 2)).float().unsqueeze(0)/255.0 
        img1 = img1.cuda()
        img2 = img2.cuda()
    
        img1 = Variable(img1, requires_grad=False)
        img2 = Variable(img2, requires_grad = False)

        SSIM = ssim_loss(img1, img2).data.item()
        print(f"image {i}: SSIM {SSIM}")
        SSIMS.append(SSIM)

    print(f"Average SSIM: {np.mean(SSIMS)}")
