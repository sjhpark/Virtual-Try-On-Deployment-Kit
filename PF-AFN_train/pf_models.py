from models.networks import ResUnetGenerator
from models.afwm import AFWM
import torch.nn as nn
import os
import sys
sys.path.append('../PF-AFN_train/')
from options.train_options import TrainOptions

opt = TrainOptions().parse()
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')

# Parser Free Warping Model
# args: (opt, input_nc) 
# where input_nc = input number of channels
warp_model = AFWM(opt, 3)

# Parser Free Geneator Model
# args: (input_nc, output_nc, num_downs, ngf, norm_layer)
# where input_nc = input number of channels, output_nc = output number of channels, num_downs = number of downsampling layers
gen_model = ResUnetGenerator(7, 4, 5, ngf=64, norm_layer=nn.BatchNorm2d)