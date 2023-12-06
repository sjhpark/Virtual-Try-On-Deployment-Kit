import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.networks import ResUnetGenerator, load_checkpoint
from models.afwm import AFWM
from distill_utils import finetune_VITON_dataset, test_output, DistillationLoss, fine_tuning, run_inference

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--student', type=str, default='Gen', help='Student Model Flag: Gen or Warp')
parser.add_argument('--student_gen_ngf', type=int, default=2, help='student gen model ngf')
parser.add_argument('--student_warp_num_filters', nargs='+', type=int, default=[16,32,64,64,64], help='student warp model num_filters') # command line example: --student_warp_num_filters 16 32 64 64 64
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--ckpt_dir', type=str, default='checkpoints/PFAFN', help='checkpoint directory')
parser.add_argument('--gen_checkpoint', type=str, default='/home/sam/Fall2023/11767/Virtual-Try-On-Deployment-Kit/PF-AFN_test/checkpoints/PFAFN/gen_model_final.pth', help='teacher gen model weights')
parser.add_argument('--warp_checkpoint', type=str, default='/home/sam/Fall2023/11767/Virtual-Try-On-Deployment-Kit/PF-AFN_test/checkpoints/PFAFN/warp_model_final.pth', help='teacher warp model weights')
parser.add_argument('--data_type', type=int, default=32, help='32 or 16 bit data')
parser.add_argument('--dataroot', type=str, default='dataset/', help='dataset directory')
parser.add_argument('--display_winsize', type=int, default=512, help='display window size')
parser.add_argument('--fineSize', type=int, default=512, help='image size')
parser.add_argument('--gpu_ids', type=str, default='1', help='gpu ids: e.g. 0  0,1,2, 0,2.')
parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
parser.add_argument('--isTrain', type=bool, default=False, help='train or test')
parser.add_argument('--loadSize', type=int, default=512, help='scale images to this size')
parser.add_argument('--max_dataset_size', type=float, default=float('inf'), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
parser.add_argument('--nThreads', default=1, type=int, help='# threads for loading data')
parser.add_argument('--name', type=str, default='demo', help='name of the experiment. It decides where to store samples and models')
parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
parser.add_argument('--resize_or_crop', type=str, default='None', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
parser.add_argument('--tf_log', action='store_true', help='if specified, use tensorboard logging. Requires tensorflow installed')
parser.add_argument('--use_dropout', action='store_true', help='use dropout for the generator')
parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
opt = parser.parse_args()
ckpt_dir = opt.ckpt_dir # checkpoint directory

print("Student Model to distill: ", f"{opt.student} Model")

# dataset & dataloader
dataset = finetune_VITON_dataset(opt)
dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

# student model (reduced size of teacher model)
if opt.student == 'Gen':
    warp_model = AFWM(opt, 3).cuda() # original size of warp module
    load_checkpoint(warp_model, opt.warp_checkpoint)
    teacher_model = ResUnetGenerator(7, 4, 5, ngf=64, norm_layer=nn.BatchNorm2d,  use_dropout=False).cuda() # teacher gen model (original size of gen module)
    load_checkpoint(teacher_model, opt.gen_checkpoint)
    student_model = ResUnetGenerator(7, 4, 5, ngf=opt.student_gen_ngf, norm_layer=nn.BatchNorm2d,  use_dropout=False).train().cuda() # original size has ngf=64
elif opt.student == 'Warp':
    gen_model = ResUnetGenerator(7, 4, 5, ngf=64, norm_layer=nn.BatchNorm2d,  use_dropout=False).cuda() # original size of gen module
    load_checkpoint(gen_model, opt.gen_checkpoint)
    teacher_model = AFWM(opt, input_nc=3).cuda() # teacher warp model (original size of warp module)
    load_checkpoint(teacher_model, opt.warp_checkpoint)
    student_model = AFWM(opt, input_nc=3).train().cuda() # where input_nc is the number of input channels (3 for RGB)
    student_model.update_num_filters(new_num_filters=opt.student_warp_num_filters)

# optimizer & criterion
optimizer = optim.Adam(student_model.parameters(), lr=opt.lr)
custom_loss = DistillationLoss()

# distilled student model
student_model_distilled = fine_tuning(dataloader, opt.student, student_model, optimizer, custom_loss, epochs=opt.epochs)

# test output
if opt.student == 'Gen':
    output = test_output(warp_model=warp_model.eval(), gen_model=student_model_distilled.eval(), opt=opt)
elif opt.student == 'Warp':
    output = test_output(warp_model=student_model_distilled.eval(), gen_model=gen_model.eval(), opt=opt)

# save distilled model
print("Saving distilled student model...")
torch.save(student_model_distilled.state_dict(), os.path.join(ckpt_dir, f'distilled_{opt.student}.pth'))
print("Distilled student model saved to {}".format(os.path.join(ckpt_dir, f'distilled_{opt.student}.pth')))

################################ Compare Inference Time (Teacher vs. Student) ################################
# Student model params
student_params = 0
for i in student_model_distilled.parameters():
    student_params += i.numel()

# Teacher model params
teacher_params = 0
for params in teacher_model.parameters():
    teacher_params += params.numel()

# print params
print(f'Student params: {student_params}')
print(f'Teacher params: {teacher_params}')

# Distilled Student Model Inference
load_checkpoint(student_model_distilled, os.path.join(opt.ckpt_dir, f'distilled_{opt.student}.pth'))
run_inference(student_model_distilled=student_model_distilled, teacher_model=teacher_model, opt=opt)
