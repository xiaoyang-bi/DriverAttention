''''
this script should get 2 type of dataset, 
and use dsu to model them ,
and sample,
and save sample and visualize,
and finally conv or fusion them,
and visualize them 
'''

import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from dataset.DrDataset import DrDataset
import argparse
from torch.utils import data
import os
import cv2
from tqdm import tqdm
torch.manual_seed(3407)


class DistributionUncertainty(nn.Module):
    """
    Distribution Uncertainty Module
        Args:
        p   (float): probabilty of foward distribution uncertainty module, p in [0,1].

    """

    def __init__(self, p=0.5, eps=1e-6):
        super(DistributionUncertainty, self).__init__()
        self.eps = eps
        self.p = p
        self.factor = 1.0

    def _reparameterize(self, mu, std):
        epsilon = torch.randn_like(std) * self.factor
        return mu + epsilon * std

    def sqrtvar(self, x):
        t = (x.var(dim=0, keepdim=True) + self.eps).sqrt()
        t = t.repeat(x.shape[0], 1)
        return t

    def forward(self, x):
        if (not self.training) or (np.random.random()) > self.p:
            return x
        # print('true')
        mean = x.mean(dim=[2, 3], keepdim=False)
        std = (x.var(dim=[2, 3], keepdim=False) + self.eps).sqrt()

        sqrtvar_mu = self.sqrtvar(mean)
        sqrtvar_std = self.sqrtvar(std)

        beta = self._reparameterize(mean, sqrtvar_mu)
        gamma = self._reparameterize(std, sqrtvar_std)

        x = (x - mean.reshape(x.shape[0], x.shape[1], 1, 1)) / std.reshape(x.shape[0], x.shape[1], 1, 1)
        x = x * gamma.reshape(x.shape[0], x.shape[1], 1, 1) + beta.reshape(x.shape[0], x.shape[1], 1, 1)

        return x



def parse_args():
    parser = argparse.ArgumentParser(description="new model training")
    parser.add_argument("--data-path", default="./dataset", help="BDDA root")
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("-b", "--batch-size", default=32, type=int)
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument("--epochs", default=10, type=int, metavar="N",
                        help="number of total epochs to train")
    parser.add_argument("--eval-interval", default=1, type=int, help="validation interval default 10 Epochs")
    parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('--print-freq', default=50, type=int, help='print frequency')
    # parser.add_argument('--resume', default='./save_weights/model_best_kldd3.pth', help='resume from checkpoint')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--model', default='uncertainty-m', help="uncertainty-r/-c/-m, resnet, ConvNext")
    parser.add_argument('--input_channel', default=1, type=int)
    parser.add_argument('--alpha', default=-1, type=float, help="if alpha=-1, without mask")
    parser.add_argument('--name', default='', help="save_name")
    parser.add_argument('--loss_func', default='kld', help='bce/ce')
    parser.add_argument('--val_aucs', default=False, type=bool)
    # Mixed precision training parameters
    parser.add_argument("--amp", action='store_true',
                        help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()

    return args


def mixup_data(x, attention, p=0.5, use_cuda=True):
    '''random select a data in batch and mix it
    attenttion is better not divide the max
    '''
    # if alpha > 0:
    #     lam = np.random.beta(alpha, alpha)
    # else:
    #     lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda() if use_cuda else torch.randperm(batch_size)

    # Generate a sequence for masking
    sequence = torch.rand(batch_size)
    sequence = (sequence < p).int().cuda() if use_cuda else (sequence < p).int()
    # Apply the mask: if the mask is 0, keep the original index order, else use the random permutation
    index = torch.where(sequence == 0, torch.arange(batch_size).cuda() if use_cuda else torch.arange(batch_size), index)
    # print(sequence)
    # print(index)
    random_data = x[index, :]
    random_atten = attention[index, :]

    mix_atten_sum =  attention + random_atten 
    eps = 1e-7
    mix_data = (x * (attention + eps) + random_data * (random_atten + eps)) / (mix_atten_sum + 2*eps)
    # print(index)

    # for i in range(batch_size):
    #     x_i = x[i, :]
    #     atten_i = attention[i, :]
    #     atten_sum = atten_i + mix_atten
    #     atten_i /= atten_sum
    #     mix_atten_i =  mix_atten.copy() / atten_sum
    #     mix_data_i = mix_data.copy()

    #     mixed_x = x_i * atten_i + mix_data_i * mix_atten_i

    return mix_data

def tensor2img(x:torch.Tensor):
    '''return a img so that cv2 can save it'''
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])

    for t, m, s in zip(x, mean, std):
        t.mul_(s).add_(m)
    x = x.permute(1, 2, 0).numpy()
    x = (x * 255).astype('uint8')
    x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)

    return x



if __name__ == '__main__':
    args = parse_args()
    dataset = DrDataset(args.data_path, mode='mix', out_folder='mix', alpha=args.alpha)
    batch_size = 32
    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    device = 'cpu'
    data_loader = data.DataLoader(dataset=dataset,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                shuffle=True,
                                pin_memory=True)
    for image, infer_gaze, out_path in tqdm(data_loader):
        image = image.to(device)
        infer_gaze = infer_gaze.to(device)
        # dsu_module = DistributionUncertainty(p=1)
        # image_mix = image
        # image_mix = dsu_module.forward(image)

        image = mixup_data(image, infer_gaze, use_cuda=False)

        # print(type(image))
        for i in range(image.size()[0]):
            # print(type(image[i]))
            image_saved = tensor2img(image[i, :, :, :])
            cv2.imwrite(str(out_path[i]), image_saved)
            # print(out_path[i])
            # print(image_mix.shape)
        # break
