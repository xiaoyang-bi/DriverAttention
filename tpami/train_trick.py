import os
import argparse
from dataset.SceneDataset import SceneDataset
from dataset.MixDataset import MixDataset

from torch.utils import data
from torch.utils.data import  ConcatDataset

from utils.train_and_evaluate import train_one_epoch, evaluate, get_params_groups, create_lr_scheduler
import wandb
import torch
import datetime
import time
from tqdm import tqdm
# from metann import Learner
import utils.train_utils as utils
import cv2
import csv
from pathlib import Path
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from dataset.DrDataset import DrDataset


torch.manual_seed(3407)

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
    parser.add_argument('--alpha', default=0.3, type=float, help="if alpha=-1, without mask")
    parser.add_argument('--project_name', default='prj', help="wandb project name")
    parser.add_argument('--name', default='', help="save_name")
    parser.add_argument('--loss_func', default='kld', help='bce/ce')
    parser.add_argument('--val_aucs', default=False, type=bool)
    # Mixed precision training parameters
    parser.add_argument("--amp", action='store_true',
                        help="Use torch.cuda.amp for mixed precision training")
    parser.add_argument("--use_wandb", action='store_true',
                    help="Use wandb to record")
    parser.add_argument('--p_dic', default=['ml_p', 'unisal_p'], nargs='+', help='A list of pseudoss')
    
    
    args = parser.parse_args()

    return args


def main(args):
    # if(args.alpha<=0): raise NotImplementedError

    if args.use_wandb:
        wandb.init(
            project=args.project_name,
            name = args.name,
            config={
            "lr": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "alpha": args.alpha
        }
)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 16])
    print(args.data_path)
    val_dataset = DrDataset('/data/bxy/MultiModelAD/data/bd_test/', mode='test', cam_subdir='camera', gaze_subdir='gaze')
    # val_noise_dataset_snow = SceneDataset(args.data_path, mode='val_snow', p_dic=args.p_dic,  noise_type='snow')

    
    print('data loader workers number: %d' % num_workers)
    print('length of val dataset: %d' % len(val_dataset))
    print('length of train_batch_size: %d' % batch_size )
    val_data_loader = data.DataLoader(val_dataset,
                                      batch_size=1,  # must be 1
                                      num_workers=num_workers,
                                      pin_memory=True)
    # val_snow_data_loader = data.DataLoader(val_noise_dataset_snow,
    #                                 batch_size=1,  # must be 1
    #                                 num_workers=num_workers,
    #                                 pin_memory=True)

    print(len(val_data_loader))

    if args.model == 'uncertainty-m':
        from models.model import Model
        model = Model('mobileViT', input_dim=args.input_channel)
    else: raise NotImplementedError
    model = model.to(device)

    params_group = get_params_groups(model, weight_decay=args.weight_decay)
    optimizer = torch.optim.AdamW(params_group, lr=args.lr, betas=(0.9, 0.999), eps=1e-8,
                                  weight_decay=args.weight_decay)
    lr_scheduler = create_lr_scheduler(optimizer, args.epochs,
                                    warmup=True)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])

    start_time = time.time()
    train_dataset = SceneDataset(args.data_path, mode='train', p_dic = args.p_dic,  alpha=args.alpha)
    train_data_loader = data.DataLoader(train_dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            shuffle=True,
                            pin_memory=True)
    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])

    start_time = time.time()
    model = model.to(device)

    for epoch in tqdm(range(args.start_epoch, args.epochs)):

        print('length of  train_dataset: %d' % len(train_dataset))
        loss, lr = train_one_epoch(args, model, optimizer, train_data_loader, val_data_loader, None, device, epoch, lr_scheduler,
                                        print_freq=args.print_freq, scaler=None)
        lr = optimizer.param_groups[0]["lr"]
        lr_scheduler.step()


        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args}
        
        
        kld_metric, cc_metric = evaluate(args, model, val_data_loader, device=device)
        kld_info, cc_info = kld_metric.compute(), cc_metric.compute()
        print(f"[epoch: {epoch}] val_kld: {kld_info:.3f} val_cc: {cc_info:.3f}")
  
        torch.save(save_file, "save_weights/model_{}_epoch_{}.pth".format(args.name, epoch) )
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("training time {}".format(total_time_str))


if __name__ == '__main__':
    a = parse_args()

    if not os.path.exists("./save_weights"):
        os.mkdir("./save_weights")

    main(a)
    # wandb.agent(sweep_id, train, count=)
