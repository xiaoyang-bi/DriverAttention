import os
import argparse
from dataset.DrDataset import DrDataset
from torch.utils import data
from utils.train_and_evaluate import train_one_epoch, evaluate, get_params_groups, create_lr_scheduler
# from utils.train_and_evaluate import create_lr_scheduler_2stages
import wandb
import torch
import datetime
import time
from tqdm import tqdm
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np

# from metann import Learner

torch.manual_seed(3407)
# wandb.init(project='fixation')

def convert(x_path, is_rgb=False):
    transform_with_resize = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    transform_wo_resize = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    x = Image.open(x_path)
    w, h = x.size

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
        
        if ( w != 224 or h != 224):
            x = cv2.resize(x, (224, 224))

        if np.max(x) > 1.0:
            x = x / 255.0
        x = torch.FloatTensor(x).unsqueeze(0)

    return x

def generate_colormap_and_blend(raw_image, gray_image, factor=0.7):
    """
    Generate a colored heatmap from a grayscale image, and blend it with a raw image.

    Parameters:
    - raw_image: The original image.
    - gray_image: Grayscale image to be used as a heatmap.
    - factor: Weight of the original image in the blend. 

    Returns:
    - blended_image: Image obtained by blending the original image and the colored heatmap.
    """
    # Normalize the grayscale image
    heatmap_min, heatmap_max = np.min(gray_image), np.max(gray_image)
    norm_heatmap = 255.0 * (gray_image - heatmap_min) / (heatmap_max - heatmap_min)
    # Apply colormap
    color_heatmap = cv2.applyColorMap(norm_heatmap.astype(np.uint8), cv2.COLORMAP_JET)
    # Resize the heatmap to match the original image size
    color_heatmap_resized = cv2.resize(color_heatmap, (raw_image.shape[1], raw_image.shape[0]))
    # Blend the original image and the heatmap
    blended_image = cv2.addWeighted(raw_image, factor, color_heatmap_resized, 1 - factor, 0)

    return blended_image

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
            module.train()
    return model

def visualize_case(args, model):
    '''
    return heatmap
    '''
    model.eval()
    model  = set_batchnorm_to_train(model)
    with torch.no_grad():
        img_path = str(args.data_path + '/47/gaussian_noise_224_224/47_039.jpg')
        # img_path = str(args.data_path + '/47/camera_224_224/47_833.jpg')

        img_raw_path = str(args.data_path + '/47/cam_512_256/47_039.jpg')
        img = convert(img_path, True)
        # print(img.size())
        img = img.unsqueeze(0)
        # print(img.size())
        img = img.to(args.device)
        output = model(img)
        out = (output[0] / output.max()).permute(1, 2, 0).cpu().detach().numpy() * 255
        # out.astype(int)
        raw_img = cv2.imread(img_raw_path)
        # cv2.imwrite('./output/spec.jpg', out)
        heatmap = generate_colormap_and_blend(raw_img, out)
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap_pil = Image.fromarray(heatmap_rgb)
        return heatmap_pil
    
    # out.astype(int)


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
    parser.add_argument('--lr', default=1e-4, type=float, help='initial learning rate')
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
    
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    if(args.alpha<=0): raise NotImplementedError

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
        args.lr = wandb.config.lr
        print(args.lr)
        args.p = wandb.config.p
        print(args.p)


    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 16])
    print(args.data_path)
    train_dataset = DrDataset(args.data_path, mode='train_mix', alpha=args.alpha)
    val_dataset = DrDataset(args.data_path, mode='val')
    val_noise_dataset_gauss = DrDataset(args.data_path, mode='val', noise_type='gaussian_noise')
    val_noise_dataset_motion = DrDataset(args.data_path, mode='val', noise_type='motion_blur')


    
    print('data loader workers number: %d' % num_workers)
    print('length of train_dataset: %d' % len(train_dataset))
    print('length of val dataset: %d' % len(val_dataset))
    print('length of val noise dataset gausss: %d' % len(val_noise_dataset_gauss))
    print('length of val noise dataset motion: %d' % len(val_noise_dataset_motion))
    print('length of train_batch_size: %d' % batch_size )

    train_data_loader = data.DataLoader(train_dataset,
                                        batch_size=batch_size,
                                        num_workers=num_workers,
                                        shuffle=True,
                                        pin_memory=True)
    val_data_loader = data.DataLoader(val_dataset,
                                      batch_size=1,  # must be 1
                                      num_workers=num_workers,
                                      pin_memory=True)
    
    val_noise_data_loader_guass = data.DataLoader(val_noise_dataset_gauss,
                                    batch_size=1,  # must be 1
                                    num_workers=num_workers,
                                    pin_memory=True)
    val_noise_data_loader_motion = data.DataLoader(val_noise_dataset_motion,
                                batch_size=1,  # must be 1
                                num_workers=num_workers,
                                pin_memory=True)

    print(len(val_data_loader))

    
    if args.model == 'uncertainty-m':
        from models.model import Model
        model = Model('mobileViT', input_dim=args.input_channel)
    else: raise NotImplementedError
    model = model.to(device)


    params_group = get_params_groups(model, weight_decay=args.weight_decay)
    optimizer = torch.optim.AdamW(params_group, lr=args.lr, betas=(0.9, 0.999), eps=1e-8,
                                  weight_decay=args.weight_decay)
    # lr_scheduler = create_lr_scheduler(optimizer, len(train_data_loader), args.epochs,
    #                                    warmup=True, warmup_epochs=1)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_data_loader), args.epochs,
                                    warmup=False)

    if args.amp:
        scaler = torch.cuda.amp.GradScaler()

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        optimizer.param_groups[0]["lr"] = args.lr
        # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        # args.start_epoch = checkpoint['epoch'] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    current_kld, current_cc = 10.0, 0.0
    current_kld_noise_gauss, current_cc_noise_gauss = 10.0, 0.0
    current_kld_noise_motion, current_cc_noise_motion = 10.0, 0.0

    # min_loss = 10000000
    start_time = time.time()
    
    # heatmap = visualize_case(args, model)
    # heatmap = wandb.Image(heatmap)
    # wandb.log({'case': heatmap})
    for epoch in tqdm(range(args.start_epoch, args.epochs)):
        loss, lr = train_one_epoch( args, model, optimizer, train_data_loader, device, epoch, lr_scheduler,
                                    aug_p=args.p,
                                    print_freq=args.print_freq, scaler=None)
        # print(mean_loss)


        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args}

        if args.amp:
            save_file["scaler"] = scaler.state_dict()

        if epoch % args.eval_interval == 0 or epoch == args.epochs - 1:
            # raise NotImplementedError
            kld_metric, cc_metric = evaluate(args, model, val_data_loader, device=device)
            kld_metric_noise_gauss, cc_metric_noise_gauss = evaluate(args, model, val_noise_data_loader_guass, device=device)
            kld_metric_noise_motion, cc_metric_noise_motion = evaluate(args, model, val_noise_data_loader_motion, device=device)

            kld_info, cc_info = kld_metric.compute(), cc_metric.compute()
            kld_info_noise_gauss, cc_info_noise_gauss = kld_metric_noise_gauss.compute(), cc_metric_noise_gauss.compute()
            kld_info_noise_motion, cc_info_noise_motion = kld_metric_noise_motion.compute(), cc_metric_noise_motion.compute()
            

            #visualize
            heatmap = visualize_case(args, model)
            heatmap = wandb.Image(heatmap)

            if(args.use_wandb):
                wandb.log({'lr': lr, 
                        'loss_trival': loss, 
                        'cc': cc_info, 
                        'kld': kld_info,
                        'cc_noise_gauss': cc_info_noise_gauss, 
                        'kld_noise_gauss': kld_info_noise_gauss,
                        'cc_noise_motion': cc_info_noise_motion, 
                        'kld_noise_motion': kld_info_noise_motion,
                        'case': heatmap})
            print(f"[epoch: {epoch}] val_kld: {kld_info:.3f} val_cc: {cc_info:.3f}")

            # 当前最佳
            if current_cc <= cc_info:
                torch.save(save_file, "save_weights/model_best_{}_{}_{}.pth".format(args.name, "{:.5f}".format(cc_info), "{:.5f}".format(kld_info) ) )
                current_cc = cc_info
            if current_cc_noise_gauss <= cc_info_noise_gauss:
                torch.save(save_file, "save_weights/model_best_{}_{}_{}.pth".format(args.name, "{:.5f}".format(cc_info), "{:.5f}".format(kld_info) ) )
                current_cc_noise_gauss = cc_info_noise_gauss
            if current_cc_noise_motion <= cc_info_noise_motion:
                torch.save(save_file, "save_weights/model_best_{}_{}_{}.pth".format(args.name, "{:.5f}".format(cc_info), "{:.5f}".format(kld_info) ) )
                current_cc_noise_motion = cc_info_noise_motion
            if current_kld >= kld_info:
                torch.save(save_file, "save_weights/model_best_{}_{}_{}.pth".format(args.name, "{:.5f}".format(cc_info), "{:.5f}".format(kld_info) ) )
                current_kld = kld_info    
            if current_kld_noise_gauss >= kld_info_noise_gauss:
                torch.save(save_file, "save_weights/model_best_{}_{}_{}.pth".format(args.name, "{:.5f}".format(cc_info), "{:.5f}".format(kld_info) ) )
                current_kld_noise_gauss = kld_info_noise_gauss
            if current_kld_noise_motion >= kld_info_noise_motion:
                torch.save(save_file, "save_weights/model_best_{}_{}_{}.pth".format(args.name, "{:.5f}".format(cc_info), "{:.5f}".format(kld_info) ) )
                current_kld_noise_motion = kld_info_noise_motion

        #存aug训练后的第一个
        if args.epochs - epoch < 5:
               torch.save(save_file, "save_weights/model_{}_epoch_{}_{}_{}.pth".format(args.name, epoch, "{:.5f}".format(cc_info), "{:.5f}".format(kld_info) ) )
        
        total_time = time.time() - start_time            
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("training time {}".format(total_time_str))



if __name__ == '__main__':
    if not os.path.exists("./save_weights"):
        os.mkdir("./save_weights")

    sweep_configuration = {
        "method": "bayes",
        "name": "sweep",
        "metric": {"goal": "maximize", "name": "cc"},
        "parameters": {
            "lr": {"max": 0.001, "min": 0.000001},
            'p': {
            'values': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        },
    },
    }

    sweep_id = wandb.sweep(sweep_configuration, project="aug_sweep_resume")
    # main(a)
    wandb.agent(sweep_id, main, count=100)
