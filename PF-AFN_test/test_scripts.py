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

ssim_loss = pytorch_ssim.SSIM(window_size = 11)

mode = 'avgpool'

if mode == 'resize':
	##### hyperparameters #####
	H, W = 64,64 # New input image size
	###########################

if mode == 'avgpool':
	##### hyperparameters #####
	H_resized, W_resized = 128, 128
	###########################
	k = 2 # kernel size
	s = 2 # stride
	
	# Ground truth image sample
	image_gt_sample = os.listdir('dataset/VITON_test/ground_truth/')[0]
	# image_gt_sample = Image.open('dataset/VITON_test/ground_truth/' + image_gt_sample)
	# image_gt_sample = transforms.ToTensor()(image_gt_sample) # PIL image to tensor

	# resize
	image_gt_sample = Image.open('dataset/VITON_test/ground_truth/' + image_gt_sample)
	image_gt_sample = transforms.Resize((H_resized,W_resized))(image_gt_sample)
	image_gt_sample = transforms.ToTensor()(image_gt_sample) # PIL image to tensor

	# Average pooling
	avgpool = nn.AvgPool2d(kernel_size=k, stride=s)
	image_gt_sample_pooled = avgpool(image_gt_sample)
	H, W = image_gt_sample_pooled.size()[-2:] # New input image size

class dressUpInference():
	def __init__(self):
		self.opt = TestOptions().parse()
		opt = TestOptions().parse()
		self.warp_model = AFWM(opt, 3)
		self.warp_model.eval()
		self.warp_model.cuda()
		self.gen_model = ResUnetGenerator(7, 4, 5, ngf=64, norm_layer=nn.BatchNorm2d)
		self.gen_model.eval()
		self.gen_model.cuda() 
		load_checkpoint(self.gen_model, opt.gen_checkpoint)
		load_checkpoint(self.warp_model, opt.warp_checkpoint)

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

	def infer(self, data):
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
		flow_out = self.warp_model(real_image, clothes)
		warped_cloth, last_flow, = flow_out
		warped_edge = F.grid_sample(edge, last_flow.permute(0, 2, 3, 1),
						  mode='bilinear', padding_mode='zeros')

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
	
	def get_statistics(self):
		dataset_path = 'dataset/VITON_test/'
		clothes = ['000020_1.jpg', '000028_1.jpg', '000038_1.jpg', '000048_1.jpg', '000057_1.jpg']
		edges = ['000020_1.jpg', '000028_1.jpg', '000038_1.jpg', '000048_1.jpg', '000057_1.jpg']
		model = ['000164_0.jpg', '019454_0.jpg', '019183_0.jpg', '019045_0.jpg', '018746_0.jpg']
		MSE = []
		SSIM = []
		for i in range(5):
			I_path = os.path.join(dataset_path+'test_img/', model[i])
			C_path = os.path.join(dataset_path+'test_clothes/', clothes[i])
			E_path = os.path.join(dataset_path+'test_edge/', edges[i])
			groundtruth_path = os.path.join(dataset_path+'ground_truth/', model[i])

			if mode == "resize":
				I = Image.open(I_path).convert('RGB').resize((W,H))

			if mode == "avgpool":
				I = Image.open(I_path).convert('RGB').resize((W_resized,H_resized))
				I = transforms.ToTensor()(I)
				I = avgpool(I)
				I = transforms.ToPILImage()(I)

			if i==0:
				print("New Input Image Size", I.size)
			input_frame = I
			params = get_params(self.opt, I.size)
			transform = get_transform(self.opt, params)
			transform_E = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)

			I_tensor = transform(I)

			C = Image.open(C_path).convert('RGB').resize((W,H))
			C_tensor = transform(C)

			E = Image.open(E_path).convert('L').resize((W,H))
			E_tensor = transform_E(E)

			self.data = { 'image': I_tensor,'clothes': C_tensor, 'edge': E_tensor}
			I = input_frame.convert('RGB')
			params = get_params(self.opt, I.size)
			transform = get_transform(self.opt, params)
			transform_E = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)

			I_tensor = transform(I)

			C = Image.open(C_path).convert('RGB')
			C_tensor = transform(C)

			E = Image.open(E_path).convert('L')
			E_tensor = transform_E(E)

			data = { 'image': I_tensor,'clothes': C_tensor, 'edge': E_tensor}
			img1 = self.infer(data)
			img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
			img2 = cv2.imread(groundtruth_path)

			if not os.path.exists(dataset_path+'results/'):
				os.makedirs(dataset_path+'results/')
			cv2.imwrite( f"{os.path.join(dataset_path+'results/', mode+'_'+model[i])}", img1)

			mse_score = mean_squared_error(img1, img2)
			ssim_score = ssim(img1, img2, data_range=img1.max() - img1.min(), multichannel=True)
			MSE.append(mse_score)
			SSIM.append(ssim_score)
			print(f"Image {i} | MSE: {mse_score}, SSIM: {ssim_score}")

		print("Output Image Size", img1.shape)
		print(f"Average MSE: {np.mean(MSE)}\nAverage SSIM: {np.mean(SSIM)}")

if __name__ == '__main__':
	obj = dressUpInference()
	obj.model_statistics()
	obj.get_statistics()
