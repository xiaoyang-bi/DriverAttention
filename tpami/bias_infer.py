import os
import argparse
import torch
from dataset.SceneDataset import SceneDataset
from torch.utils.data import DataLoader
import utils.train_utils as utils

from utils.train_and_evaluate import evaluate
import csv
import cv2


def run_infer(args, model, data_loader, device='cuda'):
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
                print(out_path[i])
                out = (output / output.max()).permute(1, 2, 0).cpu().detach().numpy() * 255
                cv2.imwrite(str(out_path[i]), out)
                count += 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='uncertainty-m', help="uncertainty-r/-c, resnet, ConvNext")
    parser.add_argument('--data_root', default='./dataset', help='where to store data')
    parser.add_argument('--save_model', default='5_layer')
    parser.add_argument('--input_dim', default=1, type=int)
    parser.add_argument('--val_aucs', default=False, type=bool)

    # parser.add_argument('--severity', default=0, type=int)
    # parser.add_argument('--cam_subdir', default='camera', type=str)


    # parser.add_argument('--output_folder', default='output', type=str)

    return parser.parse_args()


def main(args):

    dataset = SceneDataset(args.data_root, mode='test_nobias', out_folder=(args.save_model[:10] + 'nobias'))
    print(len(dataset))
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

    run_infer(args, model, data_loader)
if __name__ == '__main__':
    args = parse_args()
    main(args)
