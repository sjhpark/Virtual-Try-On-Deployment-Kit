"""Utils for distillation"""
import numpy as np
import os, glob
import subprocess
import time 
import csv
from termcolor import colored, cprint
from tqdm import tqdm
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset

from data.base_dataset import get_params, get_transform
from PIL import Image
from models.networks import ResUnetGenerator, load_checkpoint
from models.afwm import AFWM

def color_print(text, color:str='green', bold:bool=False, underline:bool=False):
    attrs = []
    if bold:
        attrs.append('bold')
    if underline:
        attrs.append('underline')
    cprint(text, color=color, attrs=attrs)

class DistillationLoss(nn.Module):
    def __init__(self, feature_loss_weight=1.0, output_loss_weight=1.0):
        super(DistillationLoss, self).__init__()
        self.feature_loss_weight = feature_loss_weight
        self.output_loss_weight = output_loss_weight
        self.feature_criterion = nn.MSELoss()
        self.output_criterion = nn.L1Loss()

    def forward(self, student_outputs, teacher_outputs, student_p_tryon, teacher_p_tryon):
        assert len(student_outputs) == len(teacher_outputs), "Number of student and teacher outputs must be the same"
        if len(student_outputs) > 1:
            warped_cloth, last_flow = teacher_outputs
            warped_cloth_student, last_flow_student = student_outputs
            feature_loss = 0.1*self.feature_criterion(warped_cloth_student, warped_cloth) + 0.9*self.feature_criterion(last_flow_student, last_flow)
        else:
            feature_loss = self.feature_criterion(student_outputs, teacher_outputs)
        
        output_loss = self.output_criterion(student_p_tryon, teacher_p_tryon)

        total_loss = self.feature_loss_weight * feature_loss + self.output_loss_weight * output_loss
        return total_loss

def teacher_outputs(teacher_warp_model, teacher_gen_model, opt, I_path, C_path, E_path):
    I = Image.open(I_path).convert('RGB')
    C = Image.open(C_path).convert('RGB')
    E = Image.open(E_path).convert('L')

    params = get_params(opt, I.size)
    transform = get_transform(opt, params)
    transform_E = get_transform(opt, params, method=Image.NEAREST, normalize=False)

    """Teacher Warp Module"""
    # Processed edge image
    edge = transform_E(E) # edge image
    edge_tmp = edge.clone()
    edge = torch.FloatTensor((edge_tmp.detach().numpy() > 0.5).astype(np.int))
    edge = edge.reshape(1, edge.shape[0], edge.shape[1], edge.shape[2]).cuda()

    # Input 1 to Warp Model: processed person image
    real_image = transform(I) # person image
    real_image = real_image.reshape(1, real_image.shape[0], real_image.shape[1], real_image.shape[2]).cuda()
    
    # Input 2 to Warp Model: processed cloth image
    cloth = transform(C).cuda() # clothe image
    cloth = cloth * edge
    # cloth = cloth.reshape(1, cloth.shape[0], cloth.shape[1], cloth.shape[2])

    # Outputs (warped clothe & last flow) from Warp Model
    warped_cloth, last_flow, = teacher_warp_model(real_image, cloth)

    """Teacher Gen Module"""
    # Warped edge
    warped_edge = F.grid_sample(edge, last_flow.permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros')

    # Inputs to Gen Model
    gen_inputs = torch.cat([real_image, warped_cloth, warped_edge], 1)

    # Output features from Gen Model
    gen_outputs = teacher_gen_model(gen_inputs)

    # Warp outputs into an image
    p_rendered, m_composite = torch.split(gen_outputs, [3, 1], 1)
    p_rendered = torch.tanh(p_rendered)
    m_composite = torch.sigmoid(m_composite)
    m_composite = m_composite * warped_edge
    p_tryon = warped_cloth * m_composite + p_rendered * (1 - m_composite) # warped image (person trying on clothe)
    
    return edge, real_image, cloth, warped_cloth, last_flow, warped_edge, gen_inputs, gen_outputs, p_tryon

class finetune_VITON_dataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        img_pattern = 'dataset/test_img/0*'
        cloth_pattern = 'dataset/test_clothes/0*'
        img_files = glob.glob(img_pattern)
        cloth_files = glob.glob(cloth_pattern)

        # Collect image, cloth, and edge paths
        self.data = []
        for i in img_files:
            for j in cloth_files:
                self.data.append((i, j, j.replace('test_clothes', 'test_edge')))
        
        # Teacher warp model
        warp_model = AFWM(self.opt, 3).eval()
        warp_model.cuda()
        load_checkpoint(warp_model, self.opt.warp_checkpoint)
        self.teacher_warp_model = warp_model

        # Teacher gen model
        gen_model = ResUnetGenerator(7, 4, 5, ngf=64, norm_layer=nn.BatchNorm2d,  use_dropout=False).eval().to('cuda')
        load_checkpoint(gen_model, self.opt.gen_checkpoint)
        self.teacher_gen_model = gen_model
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        I_path, C_path, E_path = self.data[idx]
        # gen_input, gen_output, warped_edge, warped_cloth, img = outputs(self.warp_model, self.gen_model, self.opt, img, cloth, edge)
        edge, real_image, cloth, warped_cloth, last_flow, warped_edge, gen_inputs, gen_outputs, p_tryon = teacher_outputs(self.teacher_warp_model, self.teacher_gen_model, self.opt, I_path, C_path, E_path)
        # return gen_input, gen_output, warped_edge, warped_cloth, img
        return edge, real_image, cloth, warped_cloth, last_flow, warped_edge, gen_inputs, gen_outputs, p_tryon

def fine_tuning(dataloader, student:str, student_model:nn.Module, optimizer:optim.Optimizer, custom_loss:nn.Module, epochs=10):
    for i in tqdm(range(epochs), desc='Epochs'):
        loss_epoch = []
        for edge, real_image, cloth, warped_cloth, last_flow, warped_edge, gen_inputs, gen_outputs, p_tryon in dataloader:
            if student == 'Gen':
                # Inputs & Outputs from teacher model
                inputs = gen_inputs
                teacher_outputs = gen_outputs.squeeze(1)
                # Zero out gradients
                optimizer.zero_grad()
                # Outputs from student model
                student_outputs = student_model(inputs.squeeze(1)).squeeze(1) # squeeze to remove batch dim
                # Image warping using student model's outputs
                p_rendered, m_composite = torch.split(student_outputs, [3, 1], 1)
                p_rendered = torch.tanh(p_rendered)
                m_composite = torch.sigmoid(m_composite)
                m_composite = m_composite * warped_edge
                p_tryon_student = warped_cloth * m_composite + p_rendered * (1 - m_composite) # warped image using student model's outputs
                # Compute loss
                loss = custom_loss(student_outputs, teacher_outputs, p_tryon_student, p_tryon).float()
                loss_epoch.append(loss.item())
                # Backprop
                loss.backward()
                # Update weights
                optimizer.step()

            elif student == 'Warp':
                # Inputs & Outputs from teacher model
                edge, real_image, cloth = edge.squeeze(1), real_image.squeeze(1), cloth.squeeze(1) # inputs; squeeze to remove batch dim
                teacher_outputs = warped_cloth.squeeze(1), last_flow.squeeze(1) # outputs from teacher model; squeeze to remove batch dim
                gen_outputs = gen_outputs.squeeze(1) # outputs from teacher model; squeeze to remove batch dim
                p_tryon = p_tryon.squeeze(1) # warped image from teacher model; squeeze to remove batch dim
                # Zero out gradients
                optimizer.zero_grad()
                # Outputs from student model
                student_model = student_model.cuda()
                warped_cloth_student, last_flow_student, = student_model(real_image, cloth)
                student_outputs = warped_cloth_student, last_flow_student
                # Warped edge
                warped_edge = F.grid_sample(edge, last_flow_student.permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros')
                # Warp outputs into an image
                p_rendered, m_composite = torch.split(gen_outputs, [3, 1], 1)
                p_rendered = torch.tanh(p_rendered)
                m_composite = torch.sigmoid(m_composite)
                m_composite = m_composite * warped_edge
                # print(torch.sum(m_composite)) # this is for monitoring warped edge (should not be 0)
                p_tryon_student = warped_cloth_student * m_composite + p_rendered * (1 - m_composite) # warped image (person trying on clothe)
                # Compute loss
                loss = custom_loss(student_outputs, teacher_outputs, p_tryon_student, p_tryon).float()
                loss_epoch.append(loss.item())
                # Backprop
                loss.backward()
                # Update weights
                optimizer.step()

        color_print(f'Average Distillation Loss at Epoch {i}: {np.mean(loss_epoch):.4f}', 'yellow')
    return student_model

def test_output(warp_model, gen_model, opt):
    # Set up
    I_path = 'dataset/test_img/000066_0.jpg'
    C_path = 'dataset/test_clothes/019119_1.jpg'
    E_path = 'dataset/test_edge/019119_1.jpg'

    I = Image.open(I_path).convert('RGB')
    C = Image.open(C_path).convert('RGB')
    E = Image.open(E_path).convert('L')

    params = get_params(opt, I.size)
    transform = get_transform(opt, params)
    transform_E = get_transform(opt, params, method=Image.NEAREST, normalize=False)

    real_image = transform(I)
    clothes = transform(C)
    edge = transform_E(E)

    edge_tmp = edge.clone()
    edge = torch.FloatTensor((edge_tmp.detach().numpy() > 0.5).astype(np.int))
    clothes = clothes * edge    
    real_image = real_image.reshape(1,real_image.shape[0], real_image.shape[1], real_image.shape[2])
    clothes = clothes.reshape(1, clothes.shape[0], clothes.shape[1], clothes.shape[2] )
    edge = edge.reshape(1, edge.shape[0], edge.shape[1], edge.shape[2] )
    real_image = real_image.cuda()
    clothes = clothes.cuda()
    edge = edge.cuda()
    warped_cloth, last_flow, = warp_model(real_image, clothes)

    warped_edge = F.grid_sample(edge, last_flow.permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros')
    gen_inputs = torch.cat([real_image, warped_cloth, warped_edge], 1)
    gen_outputs = gen_model(gen_inputs)
    p_rendered, m_composite = torch.split(gen_outputs, [3, 1], 1)
    p_rendered = torch.tanh(p_rendered)
    m_composite = torch.sigmoid(m_composite)
    m_composite = m_composite * warped_edge
    p_tryon = warped_cloth * m_composite + p_rendered * (1 - m_composite)
    p_tryon = p_tryon.squeeze()
    cv_img = (p_tryon.permute(1,2,0).detach().cpu().numpy()+1)/2
    return cv_img

def run_inference(student_model_distilled:nn.Module, teacher_model:nn.Module, opt, warmup_iter=50):
    if opt.student == 'Gen':
        classmate_model = AFWM(opt, input_nc=3).cuda() # original Warp Model
        load_checkpoint(classmate_model, opt.warp_checkpoint)
        dummy_input = [torch.rand(1, 7, 256, 192).cuda()]
        num_filters = opt.student_gen_ngf # ngf of distilled student model
    elif opt.student == 'Warp':
        classmate_model = ResUnetGenerator(7, 4, 5, ngf=64, norm_layer=nn.BatchNorm2d,  use_dropout=False).cuda() # original Gen Model
        load_checkpoint(classmate_model, opt.gen_checkpoint)
        dummy_input = [torch.rand(1, 3, 256, 192).cuda(), torch.rand(1, 3, 256, 192).cuda()]
        num_filters = opt.student_warp_num_filters # num_filters of distilled student model

    # Warmup
    for i in range(warmup_iter):
        student_model_distilled.eval()(*dummy_input)
    
    # Inference of teacher model
    time_container_teacher = []
    for i in range(50):
        start_time = time.time()
        teacher_model.eval()(*dummy_input)
        elapsed = time.time()-start_time
        time_container_teacher.append(elapsed)
    elapsed_teacher = sum(time_container_teacher)/len(time_container_teacher) # average inference time

    # Inference of distilled student model
    time_container_student = []
    for i in range(50):
        start_time = time.time()
        student_model_distilled.eval()(*dummy_input)
        elapsed = time.time()-start_time
        time_container_student.append(elapsed)
    elapsed_student = sum(time_container_student)/len(time_container_student) # average inference time

    color_print(f'Average Inference Time of {opt.student} Teacher Model: {elapsed_teacher:.4f} seconds', color='yellow')
    color_print(f'Average Inference Time of Distilled {opt.student} Student Model: {elapsed_student:.4f} seconds', color='yellow')

    # Save output (warped image)
    if opt.student == 'Gen':
        output = test_output(warp_model=classmate_model.eval(), gen_model=student_model_distilled.eval(), opt=opt)
    elif opt.student == 'Warp':
        output = test_output(warp_model=student_model_distilled.eval(), gen_model=classmate_model.eval(), opt=opt)
    rgb=(output*255).astype(np.uint8)
    rgb=cv2.cvtColor(rgb,cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join('results', f'distilled_{opt.student}_{num_filters}.jpg'), rgb)
    color_print(f'Output saved to results/distilled_{opt.student}_{num_filters}.jpg', bold=True, underline=True)

def count_params(model:nn.Module):
    params = 0
    for i in model.parameters():
        params += i.numel()
    return params

def gpu_power_draw(student_model:nn.Module, opt, run_time=30, warmup_iter=50):
    """Use Nvidia SMI to measure GPU power draw of a student model"""

    if opt.student == 'Gen':
        classmate_model = AFWM(opt, input_nc=3).cuda() # original Warp Model
        load_checkpoint(classmate_model, opt.warp_checkpoint)
        dummy_input = [torch.rand(1, 7, 256, 192).cuda()]
        num_filters = opt.student_gen_ngf # ngf of distilled student model
    elif opt.student == 'Warp':
        classmate_model = ResUnetGenerator(7, 4, 5, ngf=64, norm_layer=nn.BatchNorm2d,  use_dropout=False).cuda() # original Gen Model
        load_checkpoint(classmate_model, opt.gen_checkpoint)
        dummy_input = [torch.rand(1, 3, 256, 192).cuda(), torch.rand(1, 3, 256, 192).cuda()]
        num_filters = opt.student_warp_num_filters # num_filters of distilled student model

    # Warmup
    for i in range(warmup_iter):
        student_model.eval()(*dummy_input)
    
    # Start process
    curr_time_unix = int(time.time())
    filename = f"results/gpu_power_usage_{opt.student}_distilled_{curr_time_unix}.csv"
    command = f"nvidia-smi --query-gpu=gpu_name,power.draw --format=csv -l 1 --filename={filename}" # record GPU power draw per second
    process = subprocess.Popen(command, shell=True)
    
    # Run inference
    time_container_student = []
    time_track = 0
    color_print(f"Running Inference for {run_time} seconds...", bold=True, underline=True)
    while time_track < run_time:
        start_time = time.time()
        student_model.eval()(*dummy_input)
        elapsed = time.time() - start_time
        time_container_student.append(elapsed)
        time_track += elapsed
    
    # Terminate the subprocess
    subprocess.Popen("killall nvidia-smi", shell=True) # kill NVIDA SMI process
    time.sleep(2) # wait for a few seconds to fully save the csv file

    elapsed_student = sum(time_container_student) / len(time_container_student)
    color_print(f"Average Inference Time of student {opt.student}: {elapsed_student:.4f} seconds", color='yellow')

    # Read power draw from the saved csv file
    power_draws = []
    color_print(f"Reading Power Draw from {filename}...", bold=True, underline=True)
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader) # skip header
        for row in reader: # each row (e.g. ['NVIDIA GeForce GTX 1650 with Max-Q Design', ' 35.20 W'])
            row = row[1].split() # ['35.20', 'W']
            power_draw = float(row[0])
            power_draws.append(power_draw)
    average_power_draw = sum(power_draws) / len(power_draws)
    color_print(f"Average Power Draw: {average_power_draw:.4f} W", color='yellow')