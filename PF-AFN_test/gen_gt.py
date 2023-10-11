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

#python test.py --name demo --resize_or_crop None --batchSize 1 --gpu_ids 0

class dressUpInference():
    def __init__(self):
        self.opt = TestOptions().parse()
        self.warp_model = AFWM(self.opt, 3)
        self.warp_model.eval()
        self.warp_model.cuda()
        self.gen_model = ResUnetGenerator(7, 4, 5, ngf=64, norm_layer=nn.BatchNorm2d)
        self.gen_model.eval()
        self.gen_model.cuda() 
        load_checkpoint(self.gen_model, self.opt.gen_checkpoint)
        load_checkpoint(self.warp_model, self.opt.warp_checkpoint)

    def gen_gt(self):
        dataset_path = 'dataset/VITON_test/'
        clothes = ['000020_1.jpg', '000028_1.jpg', '000038_1.jpg', '000048_1.jpg', '000057_1.jpg']
        edges = ['000020_1.jpg', '000028_1.jpg', '000038_1.jpg', '000048_1.jpg', '000057_1.jpg']
        model = ['000164_0.jpg', '019454_0.jpg', '019183_0.jpg', '019045_0.jpg', '018746_0.jpg']
        for i in range(5):
            I_path = os.path.join(dataset_path+'test_img/', model[i])
            C_path = os.path.join(dataset_path+'test_clothes/', clothes[i])
            E_path = os.path.join(dataset_path+'test_edge/', edges[i])
            O_path = os.path.join(dataset_path+'ground_truth/', model[i])

            I = Image.open(I_path).convert('RGB')
            input_frame = I
            params = get_params(self.opt, I.size)
            transform = get_transform(self.opt, params)
            transform_E = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)

            I_tensor = transform(I)

            C = Image.open(C_path).convert('RGB')
            C_tensor = transform(C)

            E = Image.open(E_path).convert('L')
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
            result  = self.infer(data)
            result=cv2.cvtColor(result,cv2.COLOR_RGB2BGR)
            result = cv2.rotate(result, cv2.ROTATE_90_CLOCKWISE)

            if not os.path.exists(os.path.dirname(O_path)):
                os.makedirs(os.path.dirname(O_path))
            print('Saving image to:', O_path)
            cv2.imwrite(O_path, result)
            
            print('Done:', i)
        return  
		
    def infer(self, data):
        real_image = data['image']
        clothes = data['clothes']
        ## edge is extracted from the clothes image with the built-in function in python
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
        cv_img = cv2.rotate(cv_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        rgb=(cv_img*255).astype(np.uint8)
        return rgb

if __name__ == '__main__':
    obj = dressUpInference()
    obj.gen_gt()
