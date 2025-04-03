'''
test the corruption robustness
'''

import os.path
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import cv2
from pathlib import Path
import random
from tqdm import tqdm


import os
import argparse
import torch
from dataset.StatHard import StatHard
from dataset.SceneDataset import SceneDataset
from torch.utils.data import DataLoader
import utils.train_utils as utils

from utils.train_and_evaluate import evaluate
import csv
import cv2
# from metann import Learner

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='uncertainty-m', help="uncertainty-r/-c, resnet, ConvNext")
    parser.add_argument('--data_root', default='./dataset', help='where to store data')
    parser.add_argument('--save_model', default='5_layer')
    parser.add_argument('--input_dim', default=1, type=int)
    parser.add_argument('--val_aucs', default=False, type=bool)
    parser.add_argument('--p_dic', default=['ml_p', 'unisal_p']  ,nargs='+', help='A list of pseudoss')

   

    return parser.parse_args()

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


def convert(x_path, is_rgb=False):
    transform_with_resize = transforms.Compose([
        transforms.Resize((224, 224)), # Resize the shorter side to 224, maintaining aspect ratio
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    # transform_with_resize = transforms.Compose([transforms.Resize((224, 224)),
    #                                 transforms.ToTensor(),
    #                                 transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    transform_wo_resize = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    x = Image.open(x_path)
    w, h = x.size
    # print(x.size)

    if is_rgb:
        x = x.convert('RGB')
        if ( w == 224 and h == 224):
            x = transform_wo_resize(x)
        else:
            x = transform_with_resize(x)
    else:
        x = x.convert('L')
        x = np.array(x)
        x = x.astype('float')
        
        if ( not (w == 224 and h == 224 ) ):
            x = cv2.resize(x, (224, 224))

        if np.max(x) > 1.0:
            x = x / 255.0
        x = torch.FloatTensor(x).unsqueeze(0)

    return x

def main(args):

  
    dataset = StatHard(args.data_root, mode='test', p_dic=args.p_dic)
    data_loader = DataLoader(dataset,
                            batch_size=1,  # must be 1
                            num_workers=8,
                            pin_memory=True)

    if args.model == 'uncertainty-m':
        from models.model import Model
        model = Model('mobileViT', input_dim=args.input_dim)
    else: raise NotImplementedError


    checkpoint = torch.load('./save_weights/model_best_{}.pth'.format(args.save_model))
    model.load_state_dict(checkpoint['model'])
    model = model.to('cuda')

    avg_path = 'bd_test_average.jpg'
    if not os.path.exists(avg_path):
        gaze_average = None
        for data in data_loader:
            _, gaze, _ = data
            if gaze_average is None:
                gaze_average = gaze
            else:
                gaze_average += gaze
        gaze_average /= len(dataset)
        out = (gaze_average[0] / gaze_average.max()).permute(1, 2, 0).cpu().detach().numpy() * 255
        cv2.imwrite(avg_path, out)
    else:
        gaze_average = convert(avg_path)
        gaze_average = gaze_average.unsqueeze(0)



    dataset = StatHard(args.data_root, mode='test', p_dic=args.p_dic)
    data_loader = DataLoader(dataset,
                            batch_size=1,  
                            num_workers=8,
                            pin_memory=True)

    cnt = 0
    model.eval()
    kld_metric = utils.KLDivergence()
    cc_metric = utils.CC()

    for data in data_loader:
        img, gaze, path = data
        kl = kldiv(gaze.squeeze(1), gaze_average.squeeze(1)).item()
        img, gaze = img.cuda(), gaze.cuda()
        if (kl > 4):
            cnt += 1
            output = model(img)
            kld_metric.update(output, gaze)
            cc_metric.update(output, gaze)

    print('total examples {}'.format(cnt))
    kld_info, cc_info = kld_metric.compute(), cc_metric.compute()
    print(f"val_kld: {kld_info:.3f} val_cc: {cc_info:.3f}")


if __name__ == '__main__':
    args = parse_args()
    main(args)

