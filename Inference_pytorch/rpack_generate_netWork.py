#%%

#%%
from aimc_tasks.comp_graph.packer_utils import plot_packing_tiled

import argparse
import os
import time
from utee import misc
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from utee import make_path
from utee import wage_util
from models import dataset
import torchvision.models as models
from utee import hook
from aimc_tasks.comp_graph import cgraph,core,splitter
from datetime import datetime
from subprocess import call

parser = argparse.ArgumentParser(description='PyTorch CIFAR-X Example')
parser.add_argument('--dataset', default='cifar10', help='cifar10|cifar100|imagenet')
parser.add_argument('--model', default='VGG8', help='VGG8|DenseNet40|ResNet18')
parser.add_argument('--mode', default='WAGE', help='WAGE|FP')
parser.add_argument('--batch_size', type=int, default=200, help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train (default: 10)')
parser.add_argument('--grad_scale', type=float, default=8, help='learning rate for wage delta calculation')
parser.add_argument('--seed', type=int, default=117, help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=100,  help='how many batches to wait before logging training status')
parser.add_argument('--test_interval', type=int, default=1,  help='how many epochs to wait before another test')
parser.add_argument('--logdir', default='log/default', help='folder to save to the log')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 1e-3)')
parser.add_argument('--decreasing_lr', default='140,180', help='decreasing strategy')
parser.add_argument('--wl_weight', type=int, default=8)
parser.add_argument('--wl_grad', type=int, default=8)
parser.add_argument('--wl_activate', type=int, default=8)
parser.add_argument('--wl_error', type=int, default=8)
# Hardware Properties
# if do not consider hardware effects, set inference=0
parser.add_argument('--inference', type=int, default=0, help='run hardware inference simulation')
parser.add_argument('--subArray', type=int, default=128, help='size of subArray (e.g. 128*128)')
parser.add_argument('--parallelRead', type=int, default=128, help='number of rows read in parallel (<= subArray e.g. 32)')
parser.add_argument('--ADCprecision', type=int, default=5, help='ADC precision (e.g. 5-bit)')
parser.add_argument('--cellBit', type=int, default=1, help='cell precision (e.g. 4-bit/cell)')
parser.add_argument('--onoffratio', type=float, default=10, help='device on/off ratio (e.g. Gmax/Gmin = 3)')
# if do not run the device retention / conductance variation effects, set vari=0, v=0
parser.add_argument('--vari', type=float, default=0., help='conductance variation (e.g. 0.1 standard deviation to generate random variation)')
parser.add_argument('--t', type=float, default=0, help='retention time')
parser.add_argument('--v', type=float, default=0, help='drift coefficient')
parser.add_argument('--detect', type=int, default=0, help='if 1, fixed-direction drift, if 0, random drift')
parser.add_argument('--target', type=float, default=0, help='drift target for fixed-direction drift, range 0-1')
current_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

args = parser.parse_args()

args.logdir = os.path.join(os.path.dirname(__file__), args.logdir)
args = make_path.makepath(args,['log_interval','test_interval','logdir','epochs','gpu','ngpu','debug'])

misc.logger.init(args.logdir, 'test_log' + current_time)
logger = misc.logger.info

misc.ensure_dir(args.logdir)
logger("=================FLAGS==================")
for k, v in args.__dict__.items():
	logger('{}: {}'.format(k, v))
logger("========================================")

args.cuda = torch.cuda.is_available()
    
assert args.model in ['VGG8', 'DenseNet40', 'ResNet18'], args.model
if args.model == 'VGG8':
    from models import VGG
    model_path = './log/VGG8.pth'   # WAGE mode pretrained model
    modelCF = VGG.vgg8(args = args, logger=logger, pretrained = model_path)
elif args.model == 'DenseNet40':
    from models import DenseNet
    model_path = './log/DenseNet40.pth'     # WAGE mode pretrained model
    modelCF = DenseNet.densenet40(args = args, logger=logger, pretrained = model_path)
elif args.model == 'ResNet18':
    from models import ResNet
    # FP mode pretrained model, loaded from 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
    # model_path = './log/xxx.pth'
    # modelCF = ResNet.resnet18(args = args, logger=logger, pretrained = model_path)
    modelCF = ResNet.resnet18(args = args, logger=logger, pretrained = True)
else:
    raise ValueError("Unknown model type")

modelCF.eval()

#%%
def write_tuple_list_csv(csvname,tuple_list):
    with open(csvname,'w') as f:   
        for rect in tuple_list:
            to_write = ''
            for stat in rect:
                to_write += f'{stat},'
            f.write(to_write[:-1]+'\n')

def get_shapes_from_vgg8(modelCF):
    shape_list = [] # shape 
    for layer_name,layer_weights in modelCF.state_dict().items():
        if 'weight' in layer_name:
            if 'bn' not in layer_name:
                if 'downsample.1' not in layer_name:
                    shape_list.append(layer_weights.flatten(start_dim=1).shape)
    return shape_list

def duplicate_cols_by_weightbits(double_list):
    for i,shape in enumerate(double_list):
        double_list[i] = (shape[0],shape[1]*args.wl_weight)
    return double_list

if args.model == 'VGG8':
    # Incomplete impl. due to WAGE mode being incompatible with ONNX serialization
    shape_list = get_shapes_from_vgg8(modelCF)
    shape_list = duplicate_cols_by_weightbits(shape_list)

    core_size = (256,256)
    acc_mapping = core.Aimc_acc(
        inshapes = shape_list,
        core_size = core_size
    )

    rect_list_name = f'rectpack_networks/rectlist_{args.model}_{args.wl_weight}_{args.wl_activate}_{current_time}.csv'
    write_tuple_list_csv(rect_list_name,acc_mapping.packer.rect_list())

if args.model == 'ResNet18':
    
    core_size = (256,256)

    import onnx
    import os

    model_path = '/home/laquizon/lawrence-workspace/DNN_NeuroSim_V1.4/Inference_pytorch/onnx_models/resnet18_Opset16.onnx'
    nx_model = onnx.load(model_path)

    model_cgraph = cgraph.Cgraph.from_onnx_model(nx_model)
    acc_mapping = core.Aimc_acc(model_cgraph,core_size)

    import rpack_nsim.utils
    node_exec_list, rid_exec_list = rpack_nsim.utils.get_cgraph_valid_exec_order(acc_mapping)

    node_exec_order_name = f'rectpack_networks/node_execorder_{args.model}_{args.wl_weight}_{args.wl_activate}_{current_time}.csv'
    rid_exec_order_name = f'rectpack_networks/rid_execorder_{args.model}_{args.wl_weight}_{args.wl_activate}_{current_time}.csv'
    write_tuple_list_csv(node_exec_order_name,node_exec_list)
    write_tuple_list_csv(rid_exec_order_name,rid_exec_list)





#%%
# %run rpack_generate_netWork.py --dataset cifar10 --model ResNet18 --mode FP --inference 1 --cellBit 1 --subArray 256 --parallelRead 256