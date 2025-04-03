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

    results = {}
    cors = [None, 'snow', 'fog', 'gaussian_noise', 'motion_blur', 'impulse_noise', 'jpeg_compression']
    # cors = ['motion_blur']

    for cor in cors:

        dataset = SceneDatasetCor(args.data_root, mode='test', noise_type=cor)
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

        kld_metric, cc_metric = evaluate(args, model, data_loader, device='cuda')
        kld_info, cc_info = kld_metric.compute(), cc_metric.compute()
        results[cor] = {'kld': kld_info, 'cc': cc_info}
        print(f"val_kld: {kld_info:.3f} val_cc: {cc_info:.3f}")

    print("\nFinal Results:")
    for cor, metrics in results.items():
        if cor is None:
            print(f"clean: {metrics}")
        else:
            print(f"{cor}: {metrics}")

    with open(f'{args.save_model}.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        # Writing the header
        writer.writerow(['corruption', 'kld', 'cc'])

        for cor, metrics in results.items():
            cor_name = 'clean' if cor is None else cor
            writer.writerow([cor_name, metrics['kld'], metrics['cc']])

    print("Results have been written to results.csv")


if __name__ == '__main__':
    args = parse_args()
    main(args)
