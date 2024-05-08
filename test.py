from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
from torch.utils.data import DataLoader
from dataset import *
import time
import numpy as np
import torchvision.utils as vutils
from torch.autograd import Variable
from networks import *
from math import log10
import torchvision
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true', help='enables test during training')
parser.add_argument('--mse_avg', action='store_true', help='enables mse avg')
parser.add_argument('--num_layers_res', type=int, help='number of the layers in residual block', default=2)
parser.add_argument('--nrow', type=int, help='number of the rows to save images', default=10)
parser.add_argument('--trainfiles', default="./train.txt", type=str, help='the list of training files')
parser.add_argument('--dataroot', default="./dataset", type=str, help='path to dataset')
parser.add_argument('--testfiles', default="./test.txt", type=str, help='the list of training files')
parser.add_argument('--testroot', default="./test", type=str, help='path to dataset')
parser.add_argument('--trainsize', type=int, help='number of training data', default=18000)
parser.add_argument('--testsize', type=int, help='number of testing data', default=100)
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--test_batchSize', type=int, default=64, help='test batch size')
parser.add_argument('--save_iter', type=int, default=10, help='the interval iterations for saving models')
parser.add_argument('--test_iter', type=int, default=500, help='the interval iterations for testing')
parser.add_argument('--cdim', type=int, default=3, help='the channel-size  of the input image to network')
parser.add_argument('--input_height', type=int, default=128, help='the height  of the input image to network')
parser.add_argument('--input_width', type=int, default=None, help='the width  of the input image to network')
parser.add_argument('--output_height', type=int, default=128, help='the height  of the output image to network')
parser.add_argument('--output_width', type=int, default=None, help='the width  of the output image to network')
parser.add_argument('--crop_height', type=int, default=None, help='the width  of the output image to network')
parser.add_argument('--crop_width', type=int, default=None, help='the width  of the output image to network')
parser.add_argument('--upscale', type=int, default=2, help='the depth of wavelet tranform')
parser.add_argument('--scale_back', action='store_true', help='enables scale_back')
parser.add_argument("--nEpochs", type=int, default=500, help="number of epochs to train for")
parser.add_argument("--start_epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--outf', default='results_test/', help='folder to output images')
parser.add_argument('--manualSeed', type=int, help='manual seed')
# parser.add_argument("--pretrained", default="", type=str, help="path to pretrained model (default: none)")


def forward_parallel(net, input, ngpu):
    if ngpu > 1:
        return nn.parallel.data_parallel(net, input, range(ngpu))
    else:
        return net(input)
            
def save_checkpoint(model, epoch, iteration, prefix=""):
    model_out_path = "model/" + prefix +"model_epoch_{}_iter_{}.pth".format(epoch, iteration)
    state = {"epoch": epoch ,"model": model}
    if not os.path.exists("model/"):
        os.makedirs("model/")

    torch.save(state, model_out_path)
        
    print("Checkpoint saved to {}".format(model_out_path))

def save_images(images, name, path):   
    img = images.cpu()
    im = img.data.numpy().astype(np.float32)
    for i in range(im.shape[0]):
        img_i = im[i].transpose(1, 2, 0)  # 将通道顺序从CHW转换为HWC
        img_i = cv2.cvtColor(img_i, cv2.COLOR_RGB2BGR)  # 转换颜色通道顺序
        cv2.imwrite(os.path.join(path, f"{name}_{i}.jpg"), img_i * 255.0)


  
def merge(images, size):
  #print(images.shape())
  h, w = images.shape[1], images.shape[2]
  img = np.zeros((h * size[0], w * size[1], 3))
  #print(img)
  for idx, image in enumerate(images):
    image = image * 255
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    i = idx % size[1]
    j = idx // size[1]
    img[j*h:j*h+h, i*w:i*w+w, :] = image
  return img

def imsave(images, size, path):
  img = merge(images, size)
  # print(img) 
  return cv2.imwrite(path, img)



test_num = input("=>请输入要测的epoch:")

pretrained = './model'+'/sr_model_epoch_'+test_num+'_iter_0.pth'
global opt, model
opt = parser.parse_args()
mag = int(math.pow(2, opt.upscale))
srnet = NetSR(opt.upscale, num_layers_res=opt.num_layers_res)
# print(srnet)

if opt.scale_back:      
    is_scale_back = True
else:      
    is_scale_back = False

# Load pretrained model if specified
if pretrained:
    if os.path.isfile(pretrained):
        print("=> loading pretrained model '{}'".format(pretrained))
        # srnet.load_state_dict(torch.load(pretrained))
        weights = torch.load(pretrained)
        pretrained_dict = weights['model'].state_dict()
        model_dict = srnet.state_dict()
        # print(model_dict)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        # 3. load the new state dict
        srnet.load_state_dict(model_dict)
        print("=> loaded pretrained model")
    else:
        print("=> no pretrained model found at '{}'".format(pretrained))

wavelet_dec = WaveletTransform(scale=opt.upscale, dec=True)
wavelet_rec = WaveletTransform(scale=opt.upscale, dec=False)          
    
criterion_m = nn.MSELoss(size_average=True)

if opt.cuda:
    srnet = srnet.cuda()      
    wavelet_dec = wavelet_dec.cuda()
    wavelet_rec = wavelet_rec.cuda()
    criterion_m = criterion_m.cuda()


print("=> loading test dataset")
test_list, _ = loadFromFile(opt.testfiles, opt.testsize)
test_set = ImageDatasetFromFile(test_list, opt.testroot, 
                input_height=opt.output_height, input_width=opt.output_width,
                output_height=opt.output_height, output_width=opt.output_width,
                crop_height=None, crop_width=None,
                is_random_crop=False, is_mirror=False, is_gray=False, 
                upscale=mag, is_scale_back=is_scale_back)    
test_data_loader = torch.utils.data.DataLoader(test_set, batch_size=opt.test_batchSize,
                                        shuffle=False, num_workers=int(opt.workers))


srnet.eval()
avg_psnr = 0
epoch = 0
iteration = 0
for titer, batch in enumerate(test_data_loader, 0):
    input, target = Variable(batch[0]), Variable(batch[1])
    if opt.cuda:
        input = input.cuda()
        target = target.cuda()    

    wavelets = forward_parallel(srnet, input, opt.ngpu)                    
    prediction = wavelet_rec(wavelets)
    mse = criterion_m(prediction, target)
    psnr = 10 * log10(1 / (mse.data.item()))
    avg_psnr += psnr
                                    
    save_images(prediction, f"Test_Result_{titer}", path=opt.outf)


print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(test_data_loader)))


