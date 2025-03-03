import time
from options.test_options import TestOptions
from data.data_loader_test import CreateDataLoader
from models.networks import ResUnetGenerator, load_checkpoint
from models.afwm import AFWM
import torch.nn as nn
import os
import numpy as np
import torch
import cv2
import torch.nn.functional as F
from tqdm import tqdm
from util.util import params_count, compute_SSIM
from natsort import natsorted

opt = TestOptions().parse()

start_epoch, epoch_iter = 1, 0

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print(dataset_size)

# Parser Free Warping Model
# args: (opt, input_nc) 
# where input_nc = input number of channels
warp_model = AFWM(opt, 3)
print(warp_model)
warp_model.eval()
warp_model.cuda()
load_checkpoint(warp_model, opt.warp_checkpoint)

# Parser Free Geneator Model
# args: (input_nc, output_nc, num_downs, ngf, norm_layer)
# where input_nc = input number of channels, output_nc = output number of channels, num_downs = number of downsampling layers
gen_model = ResUnetGenerator(7, 4, 5, ngf=64, norm_layer=nn.BatchNorm2d)
print(gen_model)
gen_model.eval()
gen_model.cuda()
load_checkpoint(gen_model, opt.gen_checkpoint)

# Count number of parameters
print('\n-----------------Parameters Count:------------------')
params_count(warp_model)
params_count(gen_model)
print('-----------------------------------------------------\n')

total_steps = (start_epoch-1) * dataset_size + epoch_iter
step = 0
step_per_batch = dataset_size / opt.batchSize

inference_latency = []
for epoch in range(1,2):
    for i, data in enumerate(dataset, start=epoch_iter):

        epoch_iter += opt.batchSize
        real_image = data['image']
        clothes = data['clothes']

        # GPU WARM_UP
        if step == 0:
            print("Warm-up begins...")
            for _ in tqdm(range(10), desc="Warming up..."):
                _ = warp_model(real_image.cuda(), clothes.cuda())
            print("Warm-up complete!")

        begin = time.time()

        edge = data['edge'] # edge is extracted from the clothes image with the built-in function in python
        edge = torch.FloatTensor((edge.detach().numpy() > 0.5).astype(np.int))
        clothes = clothes * edge        

        flow_out = warp_model(real_image.cuda(), clothes.cuda())
        warped_cloth, last_flow, = flow_out
        warped_edge = F.grid_sample(edge.cuda(), last_flow.permute(0, 2, 3, 1),
                          mode='bilinear', padding_mode='zeros')

        gen_inputs = torch.cat([real_image.cuda(), warped_cloth, warped_edge], 1)
        gen_outputs = gen_model(gen_inputs)
        p_rendered, m_composite = torch.split(gen_outputs, [3, 1], 1)
        p_rendered = torch.tanh(p_rendered)
        m_composite = torch.sigmoid(m_composite)
        m_composite = m_composite * warped_edge
        p_tryon = warped_cloth * m_composite + p_rendered * (1 - m_composite)

        path = 'results/' + opt.name
        os.makedirs(path, exist_ok=True)
        sub_path = path + '/PFAFN'
        os.makedirs(sub_path,exist_ok=True)

        if step % 1 == 0:
            a = real_image.float().cuda()
            b= clothes.cuda()
            c = p_tryon
            combine = torch.cat([a[0],b[0],c[0]], 2).squeeze()
            cv_img=(combine.permute(1,2,0).detach().cpu().numpy()+1)/2
            rgb=(cv_img*255).astype(np.uint8)
            bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
            cv2.imwrite(sub_path+'/'+str(step)+'.jpg',bgr)
        
            end = time.time()

            # print inference latency
            inference_latency.append(end - begin)
            print(f"Test Image {step}, Inference latency: {inference_latency[step] * 1000:.3f}ms")

        step += 1
        if epoch_iter >= dataset_size:
            break

print(f"Mean inference latency: {sum(inference_latency) / len(inference_latency) * 1000:.3f}ms")

############## Compute SSIM ##############
# all images
all_images = os.listdir("results/demo/PFAFN/")
# ground truth images (0_gt.jpg, 1_gt.jpg, 2_gt.jpg, ...)
gt_images = [os.path.join("results/demo/PFAFN/", x) for x in all_images if x.endswith("_gt.jpg")]
gt_images = natsorted(gt_images)
# generated images (0.jpg, 1.jpg, 2.jpg, ...)
gen_images = [os.path.join("results/demo/PFAFN/", x) for x in all_images if not x.endswith("_gt.jpg")]
gen_images = natsorted(gen_images)
# compute SSIM
compute_SSIM(gt_images, gen_images)
