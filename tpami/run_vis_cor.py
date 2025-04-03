'''
test the corruption robustness
'''
import os
import argparse
import torch
from dataset.DrDataset import DrDataset
from dataset.SceneDatasetCor import SceneDatasetCor
from torch.utils.data import DataLoader
import utils.train_utils as utils

from utils.train_and_evaluate import evaluate
import csv
# from metann import Learner

import os

import torch
import torch.nn as nn
import math
import utils.train_utils as utils
from torch.nn import functional as F
from torch import autograd
# import wandb
import numpy as np
import cv2
import wandb



def criterion(inputs, p, e, type='bce'):
    total = []
    # kld = nn.KLDivLoss(reduction='none')
    bce = nn.BCELoss(reduction='none')
    mse = nn.MSELoss(reduction='none')
    ce = nn.CrossEntropyLoss(reduction='none')
    for i in range(len(p)):
        if type == 'bce':
            # p[i] = torch.split(p[i], dim=1, split_size_or_sections=1)[0]
            bce_loss = bce(inputs, p[i])
            loss = bce_loss * torch.exp(-e[i]) + e[i] / 2
            loss = torch.sum(loss, dim=[1, 2, 3])
        elif type == 'mse':
            mse_loss = mse(inputs, p[i])
            loss = mse_loss * torch.exp(-e[i]) / 2 + e[i] / 2
            loss = torch.sum(loss, dim=[1, 2, 3])
        elif type == 'ce':
            ce_loss = ce(inputs, p[i])
            loss = ce_loss * torch.exp(-e[i]) + e[i] / 2
            loss = torch.sum(loss, dim=[1, 2, 3])
        elif type == 'kld':
            kld_loss = kldiv(inputs.squeeze(1), p[i].squeeze(1))
            loss = kld_loss * torch.exp(-e[i]) + e[i] / 2
            loss = torch.sum(loss, dim=[1, 2, 3])
        # loss = [l(inputs[i], p[j][i][0].unsqueeze(0)) * torch.exp(-e[j][i]) + e[j][i] / 2 for i in range(len(inputs))]
        # branch = [l(s[j][i], p[j][i]) * torch.exp(-e[j][i]) + e[j][i] / 2 for i in range(len(inputs))]
        total.append(loss)
    total = sum(total)
    total = total.mean()
    return total


def kldiv(s_map, gt):
    batch_size = s_map.size(0)
    # c = s_map.size(1)
    w = s_map.size(1)
    h = s_map.size(2)

    sum_s_map = torch.sum(s_map.view(batch_size, -1), 1)
    expand_s_map = sum_s_map.view(batch_size, 1, 1).expand(batch_size, w, h)

    assert expand_s_map.size() == s_map.size()

    sum_gt = torch.sum(gt.view(batch_size, -1), 1)
    expand_gt = sum_gt.view(batch_size, 1, 1).expand(batch_size, w, h)

    assert expand_gt.size() == gt.size()

    s_map = s_map / (expand_s_map * 1.0)
    gt = gt / (expand_gt * 1.0)

    s_map = s_map.view(batch_size, -1)
    gt = gt.view(batch_size, -1)

    eps = 1e-8
    result = gt * torch.log(eps + gt / (s_map + eps))
    # print(torch.log(eps + gt/(s_map + eps))   )
    return torch.mean(torch.sum(result, 1))
    # return result.reshape(batch_size, w, h)


def full(pred, gt):
    loss = kldiv(pred, gt)
    return loss


def infer(args, model, data_loader, device):
    # import pdb; pdb.set_trace()
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    with torch.no_grad():
        count = 0
        for images, out_path in metric_logger.log_every(data_loader, 100, header):
            images =  images.to(device)
            # images = images.unsqueeze(0)
            if args.model.find('uncertainty') != -1:
                outputs = model(images)
            else:
                raise(NotImplementedError)
            for i in range(images.shape[0]):
                output = outputs[i]
                out = (output / output.max()).permute(1, 2, 0).cpu().detach().numpy() * 255
                cv2.imwrite(str(out_path[i]), out)
                count += 1
            
            
 
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='uncertainty-m', help="uncertainty-r/-c, resnet, ConvNext")
    parser.add_argument('--data_root', default='./dataset', help='where to store data')
    parser.add_argument('--save_model', default='5_layer')
    parser.add_argument('--input_dim', default=1, type=int)

    # parser.add_argument('--severity', default=0, type=int)
    # parser.add_argument('--cam_subdir', default='camera', type=str)


    # parser.add_argument('--output_folder', default='output', type=str)

    return parser.parse_args()


def main(args):

    results = {}
    cors = [None, 'snow', 'fog', 'gaussian_noise', 'motion_blur', 'impulse_noise', 'jpeg_compression']
    # cors = ['motion_blur']
    
    if args.model == 'uncertainty-m':
        from models.model import Model
        model = Model('mobileViT', input_dim=args.input_dim)
    else: raise NotImplementedError

    checkpoint = torch.load('./save_weights/model_best_{}.pth'.format(args.save_model))
    model.load_state_dict(checkpoint['model'])
    model = model.to('cuda')
    
    for cor in cors:

        dataset = SceneDatasetCor(args.data_root, mode='vis', noise_type=cor, out_folder="ruap_res")
        print(len(dataset))
        data_loader = DataLoader(dataset,
                                batch_size=1,  # must be 1
                                num_workers=8,
                                pin_memory=True)



        infer(args, model, data_loader, device='cuda')

if __name__ == '__main__':
    args = parse_args()
    main(args)
