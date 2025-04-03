import os
import argparse
import torch
# from dataset.CityscapeDataset import CityscapeDataset
from dataset.DrDataset import DrDataset

from torch.utils.data import DataLoader
import utils.train_utils as utils
import cv2

def run(args, model, data_loader, device):
    model.eval()
    model = set_batchnorm_to_train(model)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    # output_folder = './' + args.output_folder
    with torch.no_grad():
        # os.makedirs(output_folder, exist_ok=True)
        count = 0
        for images, out_path in metric_logger.log_every(data_loader, 100, header):
            images =  images.to(device)
            if args.model.find('uncertainty') != -1:
                output = model(images)
            else:
                raise(NotImplementedError)
            out = (output[0] / output.max()).permute(1, 2, 0).cpu().detach().numpy() * 255
            out.astype(int)
            cv2.imwrite(str(out_path[0]), out)
            # print(str(out_path[0]))
            count += 1

import torch.nn as nn
def set_batchnorm_to_train(model):
    """
    This function sets all batch normalization layers in a PyTorch model to train mode.

    Args:
    model (torch.nn.Module): The model whose batch normalization layers need to be set to train mode.

    Returns:
    torch.nn.Module: The model with batch normalization layers set to train mode.
    """
    for module in model.modules():
        if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
            print(module)
            module.train()
    return model

def run_batch(args, model, data_loader, device):
    model.eval()
    model = set_batchnorm_to_train(model)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    # output_folder = './' + args.output_folder
    with torch.no_grad():
        # os.makedirs(output_folder, exist_ok=True)
        count = 0
        for images, out_path in metric_logger.log_every(data_loader, 100, header):
            images =  images.to(device)
            if args.model.find('uncertainty') != -1:
                output = model(images)
            else:
                raise(NotImplementedError)
            batch_size = images.size()[0]
            for i in range(batch_size):
                out = (output[i] / output[i].max()).permute(1, 2, 0).cpu().detach().numpy() * 255
                # out.astype(int)
                cv2.imwrite(str(out_path[i]), out)
                # print(str(out_path[0]))
                count += 1
            

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='uncertainty-m', help="uncertainty-r/-c, resnet, ConvNext")
    parser.add_argument('--data_root', default='./dataset', help='where to store data')
    parser.add_argument('--save_model', default='5_layer')
    parser.add_argument('--input_dim', default=1, type=int)
    parser.add_argument('--val_aucs', default=False, type=bool)
    parser.add_argument('--noise_type', default=None, help='noise type')

    # parser.add_argument('--output_folder', default='output', type=str)

    return parser.parse_args()


def main(args):
   
    dataset = DrDataset(args.data_root, mode='infer', noise_type=args.noise_type)

    data_loader = DataLoader(dataset,
                             batch_size=32,  # must be 1
                             num_workers=8,
                             pin_memory=True)
    print(len(data_loader))
    
    if args.model == 'uncertainty-m':
        from models.model import Model
        model = Model('mobileViT', input_dim=args.input_dim).cuda()
    else : raise NotImplementedError



    checkpoint = torch.load('./save_weights/model_best_{}.pth'.format(args.save_model))
    model.load_state_dict(checkpoint['model'])

    # run(args, model, data_loader, device='cuda')
    run_batch(args, model, data_loader, device='cuda')



if __name__ == '__main__':
    args = parse_args()
    main(args)
