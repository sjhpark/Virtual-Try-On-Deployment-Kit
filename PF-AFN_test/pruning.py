from options.test_options import TestOptions
from data.base_dataset import BaseDataset, get_params, get_transform
from PIL import Image
from models.networks import ResUnetGenerator, load_checkpoint
from models.afwm import AFWM
import torch
import torch.nn as nn
import numpy as np
import os
import cv2
import torch.nn.functional as F
import  util.pytorch_ssim as pytorch_ssim
from torch.autograd import Variable
import torchvision.transforms as transforms
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
from PIL import Image
from fvcore.nn import FlopCountAnalysis
from flopth import flopth
import time
from tqdm import tqdm
from natsort import natsorted
from torch.utils.data import Dataset, DataLoader

ssim_loss = pytorch_ssim.SSIM(window_size = 11)

#### hyperparamters A ####
mode = 'resize'
batch_size = 1 # input batch size
########################

if mode == 'resize':
	##### hyperparameters B #####
	H, W = 256,192 # New input image size
	###########################

if mode == 'avgpool':
	##### hyperparameters B #####
	'''
	H or W = 64, 128, 256, 512, or ...
	(k,s) = (2,2), (4,2), or (4,4)
	'''
	H_resized, W_resized = 256, 256
	k = 4 # kernel size
	s = 4 # stride
	###########################

	
	# Ground truth image sample
	image_gt_sample = os.listdir('dataset/VITON_test/ground_truth/')[0]

	# resize
	image_gt_sample = Image.open('dataset/VITON_test/ground_truth/' + image_gt_sample)
	image_gt_sample = transforms.Resize((H_resized,W_resized))(image_gt_sample)
	image_gt_sample = transforms.ToTensor()(image_gt_sample) # PIL image to tensor

	# Average pooling
	avgpool = nn.AvgPool2d(kernel_size=k, stride=s)
	image_gt_sample_pooled = avgpool(image_gt_sample)
	H, W = image_gt_sample_pooled.size()[-2:] # New input image size

# pytorch custom dataset
class CustomDataset(Dataset):
	def __init__(self, mode, stat_flag=False):
		self.opt = TestOptions().parse()
		self.mode = mode
		self.stat_flag = stat_flag
		self.dataset_path = 'dataset/VITON_test/'
		if stat_flag == False: # for evaluating the model
			self.C_list = natsorted(os.listdir(os.path.join(self.dataset_path, 'test_clothes')))
			self.E_list = natsorted(os.listdir(os.path.join(self.dataset_path, 'test_edge')))
			self.I_list = natsorted(os.listdir(os.path.join(self.dataset_path, 'test_img')))
		else: # for evaluating statictics of the model
			self.C_list = ['000020_1.jpg', '000028_1.jpg', '000038_1.jpg', '000048_1.jpg', '000057_1.jpg']
			self.E_list = ['000020_1.jpg', '000028_1.jpg', '000038_1.jpg', '000048_1.jpg', '000057_1.jpg']
			self.I_list  = ['000164_0.jpg', '019454_0.jpg', '019183_0.jpg', '019045_0.jpg', '018746_0.jpg']
	
	def __len__(self):
		return len(self.I_list)

	def __getitem__(self, idx):
		if self.stat_flag:
			groundtruth = os.path.join(self.dataset_path, 'ground_truth', self.I_list[idx])

		I_sample = os.path.join(self.dataset_path, 'test_img', self.I_list[idx])
		C_sample = os.path.join(self.dataset_path, 'test_clothes', self.C_list[idx])
		E_sample = os.path.join(self.dataset_path, 'test_edge', self.E_list[idx])

		if self.mode == "resize":
			I = Image.open(I_sample).convert('RGB').resize((W,H))

		if self.mode == "avgpool":
			I = Image.open(I_sample).convert('RGB').resize((W_resized,H_resized))
			I = transforms.ToTensor()(I)
			I = avgpool(I)
			I = transforms.ToPILImage()(I)

		input_frame = I
		params = get_params(self.opt, I.size)
		transform = get_transform(self.opt, params)
		transform_E = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)

		I = input_frame.convert('RGB')
		params = get_params(self.opt, I.size)
		transform = get_transform(self.opt, params)
		transform_E = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)

		I_tensor = transform(I)

		C = Image.open(C_sample).convert('RGB').resize((W,H))
		C_tensor = transform(C)

		E = Image.open(E_sample).convert('L').resize((W,H))
		E_tensor = transform_E(E)

		data = {'image': I_tensor,'clothes': C_tensor, 'edge': E_tensor}

		real_image = data['image']
		clothes = data['clothes']
		edge = data['edge']
		edge_tmp = edge.clone()
	
		edge = torch.FloatTensor((edge_tmp.detach().numpy() > 0.5).astype(np.int))
		
		clothes = clothes * edge    
		real_image = real_image.reshape(1,real_image.shape[0], real_image.shape[1], real_image.shape[2])
		clothes = clothes.reshape(1, clothes.shape[0], clothes.shape[1], clothes.shape[2] )
		edge = edge.reshape(1, edge.shape[0], edge.shape[1], edge.shape[2] )
		real_image = real_image.cuda()
		clothes = clothes.cuda()
		edge =edge.cuda()

		# drop the first dummy dimension
		real_image = real_image.squeeze(0)
		clothes = clothes.squeeze(0)
		edge = edge.squeeze(0)

		if self.stat_flag:
			return real_image, clothes, edge, groundtruth
		else:
			return real_image, clothes, edge

class dressUpInference():
    def __init__(self):
        self.opt = TestOptions().parse()
        opt = TestOptions().parse()
        self.warp_model = AFWM(opt, 3)
        self.warp_model.eval()
        self.warp_model.cuda()
        self.gen_model = ResUnetGenerator(7, 4, 5, ngf=64, norm_layer=nn.BatchNorm2d)
        # self.gen_model = ResUnetGenerator(7, 4, 5, ngf=128, norm_layer=nn.BatchNorm2d)
        self.gen_model.eval()
        self.gen_model.cuda() 
        load_checkpoint(self.gen_model, opt.gen_checkpoint)
        load_checkpoint(self.warp_model, opt.warp_checkpoint)

    def size_on_disk(self, model):
        '''
        Reference: https://pytorch.org/tutorials/recipes/recipes/dynamic_quantization.html
        '''
        dir = 'out'
        if not os.path.exists(dir):
            os.makedirs(dir)
        torch.save(model.state_dict(), f"{dir}/temp.p")
        size = os.path.getsize(f"{dir}/temp.p")
        print(f"{model.__class__.__name__} Size on Disk: {size/1e6} MB")
        os.remove(f"{dir}/temp.p")
        return size

    def model_statistics(self):
        warp_params = 0
        for k in self.warp_model.parameters():
            warp_params+=k.numel()
        gen_params = 0
        for k in self.gen_model.parameters():
            gen_params+=k.numel()
        print(f'Warp Model Params: {warp_params}')
        print(f'Generator Model Params: {gen_params}')
        print(f'Total Model Params: {warp_params+gen_params}')
        warp_flops, warp_params = flopth(self.warp_model, in_size=((3, H, W),(3, H, W),))
        gen_flops, gen_params = flopth(self.gen_model, in_size=((7, H, W),))
        print(f'Warp Params - Library: {warp_params}')
        print(f'Gen Params - Library: {gen_params}')
        print(f'Warp FLOPS - Library: {warp_flops}')
        print(f'GEN FLOPS - Library: {gen_flops}')

        self.size_on_disk(self.warp_model)
        self.size_on_disk(self.gen_model)

    def infer(self, data):
        real_image = data['image']
        clothes = data['clothes']
        edge = data['edge']
        edge_tmp = edge.clone()

        if edge_tmp.device != torch.device('cpu'):
            edge = torch.FloatTensor((edge_tmp.detach().cpu().numpy() > 0.5).astype(int))
        else:
            edge = torch.FloatTensor((edge_tmp.detach().numpy() > 0.5).astype(int))
        # convert back to cuda
        edge = edge.cuda()

        clothes = clothes * edge
        real_image = real_image.reshape(real_image.shape[0], real_image.shape[1], real_image.shape[2], real_image.shape[3])
        clothes = clothes.reshape(clothes.shape[0], clothes.shape[1], clothes.shape[2], clothes.shape[3] )
        edge = edge.reshape(edge.shape[0], edge.shape[1], edge.shape[2], edge.shape[3] )
        real_image = real_image.cuda()
        clothes = clothes.cuda()
        edge =edge.cuda()

        flow_out = self.warp_model(real_image, clothes)

        warped_cloth, last_flow, = flow_out
        warped_edge = F.grid_sample(edge, last_flow.permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros')
        gen_inputs = torch.cat([real_image, warped_cloth, warped_edge], 1)

        gen_outputs = self.gen_model(gen_inputs)

        p_rendered, m_composite = torch.split(gen_outputs, [3, 1], 1)
        p_rendered = torch.tanh(p_rendered)
        m_composite = torch.sigmoid(m_composite)
        m_composite = m_composite * warped_edge
        p_tryon = warped_cloth * m_composite + p_rendered * (1 - m_composite)
        p_tryon = p_tryon.squeeze()
        cv_img = (p_tryon.permute(1,2,0).detach().cpu().numpy()+1)/2
        # cv_img = cv2.rotate(cv_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        rgb=(cv_img*255).astype(np.uint8)
        return rgb

    def get_statistics(self, img_num):
            MSE = []
            SSIM = []

            dataset_path = 'dataset/VITON_test/'
            dataset = CustomDataset(mode, stat_flag=True)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
            for i, batch in enumerate(dataloader):
                I_tensor, C_tensor, E_tensor, groundtruth = batch
                groundtruth = groundtruth[0]

                if i < img_num:
                    data = { 'image': I_tensor,'clothes': C_tensor, 'edge': E_tensor}
                    img1 = self.infer(data)
                    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
                    img2 = cv2.imread(groundtruth)
                    groundtruth = groundtruth.split('/')[-1]

                    if not os.path.exists(dataset_path+'results/'):
                        os.makedirs(dataset_path+'results/')
                    cv2.imwrite(f"{os.path.join(dataset_path+'results/', mode+'_'+groundtruth)}", img1)

                    mse_score = mean_squared_error(img1, img2)
                    ssim_score = ssim(img1, img2, data_range=img1.max() - img1.min(), multichannel=True)
                    MSE.append(mse_score)
                    SSIM.append(ssim_score)
                    print(f"Image {i} | MSE: {mse_score}, SSIM: {ssim_score}")
            print(f"Generated images are saved in {dataset_path+'results/'}")

    def measure_inference_time(self, warmup_itr=10):
        self.opt = TestOptions().parse()
        opt = TestOptions().parse()
        self.warp_model = AFWM(opt, 3)
        self.warp_model.eval()
        self.warp_model.cuda()
        self.gen_model = ResUnetGenerator(7, 4, 5, ngf=64, norm_layer=nn.BatchNorm2d)
        # self.gen_model = ResUnetGenerator(7, 4, 5, ngf=128, norm_layer=nn.BatchNorm2d)
        self.gen_model.eval()
        self.gen_model.cuda() 
        load_checkpoint(self.gen_model, opt.gen_checkpoint)
        load_checkpoint(self.warp_model, opt.warp_checkpoint)

        dataset = CustomDataset(mode=mode)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # Warmup
        # first img, clothe, edge from first batch
        img, clothes, edge = next(iter(dataloader))
        fake1, fake2, fake3 = torch.zeros_like(img), torch.zeros_like(clothes), torch.zeros_like(edge)
        for i in tqdm(range(warmup_itr), desc="Warming up"):
            _ = self.warp_model(fake1, fake2)

        # Inference of Warpping Model
        real_image, clothes, edge = next(iter(dataloader))
        # Compute inference time
        start_time = time.time()
        flow_out = self.warp_model(real_image, clothes)
        end_time = time.time()
        inference_time = end_time - start_time
        print(f'{self.warp_model.__class__.__name__} per-Image Inference Time: {inference_time/batch_size} seconds')

        warped_cloth, last_flow, = flow_out
        warped_edge = F.grid_sample(edge, last_flow.permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros')
        gen_inputs = torch.cat([real_image, warped_cloth, warped_edge], 1)

        # Inference of Generative Model
        # Compute inference time
        start_time = time.time()
        gen_outputs = self.gen_model(gen_inputs)
        end_time = time.time()
        inference_time = end_time - start_time
        print(f'{self.gen_model.__class__.__name__} per-Image Inference Time: {inference_time/batch_size} seconds')

if __name__ == '__main__':
	obj = dressUpInference()
	obj.model_statistics() # param count & FLOPs count
	obj.measure_inference_time(warmup_itr=10) # measure inference time
	obj.get_statistics(img_num=5) # generate one image and compute accuracy (MSE, SSIM); img_num = either 1, 2, 3, 4, 5 (total 5 groundtruth images)
