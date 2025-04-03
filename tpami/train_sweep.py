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
# from metann import Learner

torch.manual_seed(3407)
# wandb.init(project='fixation')




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
            # "lr": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "alpha": args.alpha
            }
        )
        args.lr = wandb.config.lr
        print(args.lr)


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

    parameters_theta = []
    parameters_phi = []
    for name, param in model.named_parameters():
        if "noise" in name:
            parameters_phi.append(param)
        else:
            parameters_theta.append(param)
    model = model.cuda()


    params_group = get_params_groups(model, weight_decay=args.weight_decay)
    optimizer = torch.optim.AdamW(params_group, lr=args.lr, betas=(0.9, 0.999), eps=1e-8,
                                  weight_decay=args.weight_decay)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_data_loader), args.epochs,
                                       warmup=True, warmup_epochs=1)

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
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    current_kld, current_cc = 10.0, 0.0
    current_kld_noise_gauss, current_cc_noise_gauss = 10.0, 0.0
    current_kld_noise_motion, current_cc_noise_motion = 10.0, 0.0

    # min_loss = 10000000
    start_time = time.time()
    model = model.to(device)
    for epoch in tqdm(range(args.start_epoch, args.epochs)):
        loss, lr = train_one_epoch(args, model, optimizer, train_data_loader, device, epoch, lr_scheduler,
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

            if(args.use_wandb):
                wandb.log({'lr': lr, 
                        'loss_trival': loss, 
                        # 'loss_noise': loss_noise, 
                        # 'loss': loss,
                        # 'aug_loss': aug_loss, 
                        # 'domain_inv_loss': domain_inv_loss,
                        # 'kl_loss': kl_loss,
                        'cc': cc_info, 
                        'kld': kld_info,
                        'cc_noise_gauss': cc_info_noise_gauss, 
                        'kld_noise_gauss': kld_info_noise_gauss,
                        'cc_noise_motion': cc_info_noise_motion, 
                        'kld_noise_motion': kld_info_noise_motion})
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
    # a = parse_args()

    if not os.path.exists("./save_weights"):
        os.mkdir("./save_weights")

    sweep_configuration = {
        "method": "bayes",
        "name": "sweep",
        "metric": {"goal": "maximize", "name": "cc"},
        "parameters": {
            "lr": {"max": 0.001, "min": 0.000001}
    },
}

    sweep_id = wandb.sweep(sweep_configuration, project="aug_sweep")
    # main(a)
    wandb.agent(sweep_id, main, count=30)
