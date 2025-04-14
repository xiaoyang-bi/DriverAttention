import os
import argparse
from dataset.SceneDataset import SceneDataset
from dataset.MixDataset import MixDataset

from torch.utils import data
from torch.utils.data import  ConcatDataset

from utils.train_and_evaluate import train_trival_one_epoch, evaluate, get_params_groups, create_lr_scheduler, create_trival_lr_scheduler
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
import torch.nn as nn

torch.manual_seed(3407)
    
def run_infer(args, model, data_loader, device):
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

def sort_desc_with_index(lst):
    sorted_list_with_index = sorted(enumerate(lst), key=lambda x: x[1], reverse=True)
    sorted_indexes, sorted_values = zip(*sorted_list_with_index)
    return list(sorted_values), list(sorted_indexes)

def tensor2img(x:torch.Tensor):
    '''return a img so that cv2 can save it'''
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])

    for t, m, s in zip(x, mean, std):
        t.mul_(s).add_(m)
    x = x.cpu().detach().permute(1, 2, 0).numpy()
    x = (x * 255).astype('uint8')
    x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)

    return x

def mixup_data(x, p, attention, idx1, idx2):
    '''random select a data in batch and mix it
    attenttion is better not divide the max
    '''
    x1, x2 = x[idx1], x[idx2]
    # p1, p2 = p[idx1], p[idx2]
    atten1, atten2 = attention[idx1], attention[idx2]


    mix_atten_sum =  atten1 + atten2 
    eps = 1e-7
    mix_data = (x1 * (atten1 + eps) + x2 * (atten2 + eps)) / (mix_atten_sum + 2*eps)
    mix_p = []
    for pi in p:
        p1, p2 = pi[idx1], pi[idx2]
        pseudo = (p1 * (atten1 + eps) + p2 * (atten2 + eps)) / (mix_atten_sum + 2*eps)
        mix_p.append(pseudo)
    return mix_data, mix_p


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

def run_infer_mixup(args, model, data_loader, gaze_average, device, topK=4):
    model.eval()
    gaze_average = gaze_average.to(device)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    os.makedirs(str(Path(args.data_path)/'mix_data'), exist_ok=True)
    os.makedirs(str(Path(args.data_path)/'mix_data'/'0'), exist_ok=True)
    os.makedirs(str(Path(args.data_path)/'mix_data'/'1'), exist_ok=True)
    os.makedirs(str(Path(args.data_path)/'mix_data'/'camera_224_224'), exist_ok=True)

    

    with torch.no_grad():
        count = 0
        for images, p, out_path in metric_logger.log_every(data_loader, 100, header):
            batch_size = images.size()[0]
            images =  images.to(device)
            p = [ps.to(device) for ps in p]
            if args.model.find('uncertainty') != -1:
                outputs = model(images)
            else:
                raise(NotImplementedError)
            
            kl_list = []
            attention_list = []
            for i in range(images.shape[0]):
                #trival infer
                output = outputs[i]
                attention_list.append(output)
                out = (output / output.max()).permute(1, 2, 0).cpu().detach().numpy() * 255
                cv2.imwrite(str(out_path[i]), out)

                #cal kl 
                kl = kldiv(output, gaze_average).item()
                kl_list.append(kl)
            if (batch_size != args.batch_size):
                continue
            _,  idxs = sort_desc_with_index(kl_list)
            top_kl_idx = idxs[:topK]
            for idx in top_kl_idx:
                for j in range(batch_size - topK):
                    mix_imgs, mix_ps =  mixup_data(images, p, attention_list, idx, idxs[j+topK])
                    img_out = tensor2img(mix_imgs)
                    for p_idx, mix_p in enumerate(mix_ps):
                        p_out = (mix_p / mix_p.max()).permute(1, 2, 0).cpu().detach().numpy() * 255
                        cv2.imwrite( str(Path(args.data_path)/'mix_data'/'{}'.format(p_idx)/"{}.jpg".format(count)), p_out) 

                    cv2.imwrite( str(Path(args.data_path)/'mix_data'/'camera_224_224'/"{}.jpg".format(count)), img_out)
                    count += 1


def evaluate_batch(args, model, data_loader, device):
    # import pdb; pdb.set_trace()
    model.eval()
    kld_metric = utils.KLDivergence()
    cc_metric = utils.CC()
    if args.val_aucs:
        aucs_metric = utils.SAuc()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    with torch.no_grad():
        os.makedirs('./output', exist_ok=True)
        count = 0
        for images, targets in metric_logger.log_every(data_loader, 100, header):
            images, targets = images.to(device), targets.to(device)
            if args.model.find('uncertainty') != -1:
                output, _ = model(images)
            else:
                raise NotImplementedError   
            #     output, _ = model(images)
            batch_size = images.size(0)
            for i in range(batch_size):
                kld_metric.update(output[i].unsqueeze(0), targets[i].unsqueeze(0))
                cc_metric.update(output[i].unsqueeze(0), targets[i].unsqueeze(0))
                if args.val_aucs:
                    aucs_metric.update(output[i].unsqueeze(0), targets[i].unsqueeze(0))
    if args.val_aucs:
        return kld_metric, cc_metric, aucs_metric
    else:
        return kld_metric, cc_metric
def parse_args():
    parser = argparse.ArgumentParser(description="new model training")
    parser.add_argument("--data-path", default="./dataset", help="BDDA root")
    parser.add_argument("--val-data-path", default="./dataset", help="BDDA root")
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
    parser.add_argument('--backbone', default='mobileViT', help="resnet/ConvNext/mobileViT/vgg/mobilenet/densenet")
    parser.add_argument('--loss_func', default='kld', help='bce/ce')
    parser.add_argument('--val_aucs', default=False, type=bool)
    # Mixed precision training parameters
    parser.add_argument("--amp", action='store_true',
                        help="Use torch.cuda.amp for mixed precision training")
    parser.add_argument("--use_wandb", action='store_true',
                    help="Use wandb to record")
    parser.add_argument('--p_dic', default=['ml_p', 'unisal_p'], nargs='+', help='A list of pseudoss')
    parser.add_argument('--prior', nargs='+', help='A list of pseudoss')


    #===================for ablation study=============================
    parser.add_argument('--use_prior', default=1, type=int)
    parser.add_argument('--use_unc', default=1, type=int)
    parser.add_argument('--use_nonlocal', default=1, type=int)
    
    
    #===================for cod=========================================
    parser.add_argument('--treshold', type=float, default=0.999,
                        help='treshold for the pseudo inverse')
    parser.add_argument('--tradeoff_angle', type=float, default=0.5,
                            help='tradeoff for angle alignment')
    parser.add_argument('--tradeoff_scale', type=float, default=0.003,
                            help='tradeoff for scale alignment')
    parser.add_argument('--tradeoff_kgw', type=float, default=1e-4, help='tradeoff for kgw dist')
    parser.add_argument('--tradeoff_cod', type=float, default=1e-3, help='tradeoff for cod dist')
    parser.add_argument("--save-folder", default="./save_weights", help="BDDA root")
    parser.add_argument("--cor", default="snow", help="cor type for domain adaptation")
    
    

        
    args = parser.parse_args()

    return args

def convert(x_path, is_rgb=False):
    transform_with_resize = transforms.Compose([
        transforms.Resize((224, 224)), # Resize the shorter side to 224, maintaining aspect ratio
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
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
        
        if ( not (w == 224 and h == 224 ) ):
            x = cv2.resize(x, (224, 224))
        if np.max(x) > 1.0:
            x = x / 255.0
        x = torch.FloatTensor(x).unsqueeze(0)
    return x


def write_csv(name, epoch, kl_db):
    csv_file_path = './logs/' + name  + '_{}'.format(epoch) + '.csv'
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Key', 'Value'])  
        writer.writerows(list(kl_db.items())) 
        
        
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



def DARE_GRAM_LOSS(H1, H2, device, args):    
    b,p = H1.shape

    A = torch.cat((torch.ones(b,1).to(device), H1), 1)
    B = torch.cat((torch.ones(b,1).to(device), H2), 1)

    cov_A = (A.t()@A)
    cov_B = (B.t()@B) 

    _,L_A,_ = torch.linalg.svd(cov_A)
    _,L_B,_ = torch.linalg.svd(cov_B)
    
    eigen_A = torch.cumsum(L_A.detach(), dim=0)/L_A.sum()
    eigen_B = torch.cumsum(L_B.detach(), dim=0)/L_B.sum()

    if(eigen_A[1]>args.treshold):
        T = eigen_A[1].detach()
    else:
        T = args.treshold
        
    index_A = torch.argwhere(eigen_A.detach()<=T)[-1]

    if(eigen_B[1]>args.treshold):
        T = eigen_B[1].detach()
    else:
        T = args.treshold

    index_B = torch.argwhere(eigen_B.detach()<=T)[-1]
    
    k = max(index_A, index_B)[0]

    A = torch.linalg.pinv(cov_A ,rtol = (L_A[k]/L_A[0]).detach())
    B = torch.linalg.pinv(cov_B ,rtol = (L_B[k]/L_B[0]).detach())
    
    cos_sim = nn.CosineSimilarity(dim=0,eps=1e-6)
    cos = torch.dist(torch.ones((p+1)).to(device),(cos_sim(A,B)),p=1)/(p+1)
    return args.tradeoff_angle*(cos) + args.tradeoff_scale*torch.dist((L_A[:k]),(L_B[:k]))/k


def mul_guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5): # 混合5个guassian kernel
    batch_size = int(source.size()[0])
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)

    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2) 

    bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]

    kernels = sum(kernel_val)

    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]

    return XX, YY, XY, YX

def COD_Metric(fea_s, fea_t, prob_s, prob_t, device, epsilon=5e-2):
    num_sam_s = fea_s.shape[0]
    num_sam_t = fea_t.shape[0]

    I_s = torch.eye(num_sam_s).to(device)
    I_t = torch.eye(num_sam_t).to(device)

    #====== Kernel Matrix and Centering Matrix =======
    H_s = ( torch.eye(num_sam_s) - torch.ones(num_sam_s)/num_sam_s ).to(device)
    H_t = ( torch.eye(num_sam_t) - torch.ones(num_sam_t)/num_sam_t ).to(device)

    K_ZsZs, K_ZtZt, K_ZsZt, K_ZtZs = mul_guassian_kernel(fea_s, fea_t)
    K_YsYs, K_YtYt, K_YsYt, K_YtYs = mul_guassian_kernel(prob_s, prob_t)

    #################################### CMMD ########################################
    # Inv_K_YsYs = (epsilon *I_s + K_YsYs).inverse() 
    # Inv_K_YtYt = (epsilon* I_t + K_YtYt).inverse()

    Inv_K_YsYs = ((K_YsYs.mm(K_YsYs) + epsilon * I_s).inverse()).mm(K_YsYs) # another kind of regularization
    Inv_K_YtYt = ((K_YtYt.mm(K_YtYt) + epsilon * I_t).inverse()).mm(K_YtYt) 

    C_t = ((K_YtYt.mm(Inv_K_YtYt)).mm(K_ZtZt)).mm(Inv_K_YtYt)
    C_s = ((K_YsYs.mm(Inv_K_YsYs)).mm(K_ZsZs)).mm(Inv_K_YsYs)
    C_st = ((K_YtYs.mm(Inv_K_YsYs)).mm(K_ZsZt)).mm(Inv_K_YtYt)

    CMMD_dist =  (-1) * torch.sqrt((C_s.trace() + C_t.trace() + 2 * C_st.trace()))
    #################################### CKB ######################################## 
    G_Ys = (H_s.mm(K_YsYs)).mm(H_s)
    G_Yt = (H_t.mm(K_YtYt)).mm(H_t)
    G_Zs = (H_s.mm(K_ZsZs)).mm(H_s)
    G_Zt = (H_t.mm(K_ZtZt)).mm(H_t)
    #====== R_{s} and R_{t} =======
    Inv_s = (epsilon*num_sam_s*I_s + G_Ys).inverse() 
    Inv_t = (epsilon*num_sam_t*I_t + G_Yt).inverse()

    R_s = epsilon*G_Zs.mm(Inv_s)
    R_t = epsilon*G_Zt.mm(Inv_t)
       
    #====== R_{st} =======
    B_s = num_sam_s*epsilon*Inv_s
    B_t = num_sam_t*epsilon*Inv_t
    B_s = (B_s + B_s.t())/2 # numerical symmetrize
    B_t = (B_t + B_t.t())/2 # numerical symmetrize

    S_s, U_s = torch.linalg.eigh(B_s)
    S_t, U_t = torch.linalg.eigh(B_t)

    # S_sn = (S_s+1e-4).pow(0.5).diag()
    # S_tn = (S_t+1e-4).pow(0.5).diag()

    #====== optional; some regulization for eigh =======
    if torch.isnan(S_s[0]):
        S_s[0] = 0
    if torch.isnan(S_t[0]):
        S_t[0] = 0
    Ss = torch.maximum(S_s, torch.zeros(S_s.size()).to(device))
    St = torch.maximum(S_t, torch.zeros(S_t.size()).to(device))
    S_sn = (Ss+1e-4).pow(0.5).diag()
    S_tn = (St+1e-4).pow(0.5).diag()

    HC_s = H_s.mm( U_s.mm(S_sn) )
    HC_t = H_t.mm( U_t.mm(S_tn) )
    Nuclear = (HC_t.t().mm(K_ZtZs)).mm(HC_s)

    S_n = torch.linalg.svdvals(Nuclear)    
    
    CKB_dist = (R_s.trace() + R_t.trace() - 2*S_n[:-1].sum()/((num_sam_s*num_sam_t)**0.5))

    return CMMD_dist + CKB_dist

def main(args):
    # if(args.alpha<=0): raise NotImplementedError
    cor = args.cor
    print('use prior :{}'.format(args.use_prior))
    print('use unc: {}'.format(args.use_unc))
    print('use nonlocal: {}'.format(args.use_nonlocal))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 16])
    print(args.data_path)

    # val_dataset = SceneDataset(args.data_path, mode='val')
    # val_dataset = SceneDataset('/data/bxy/MultiModelAD/data/bd_test/', mode='test')
    # val_dataset = DrDataset(args.val_data_path, mode='test', cam_subdir='camera', gaze_subdir='gaze')
    train_dataset = SceneDataset(args.data_path, mode='train', p_dic = args.p_dic,  alpha=args.alpha, use_prior=args.use_prior)
    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    train_data_loader = data.DataLoader(train_dataset,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                shuffle=True,
                                pin_memory=True)
    
    print('data loader workers number: %d' % num_workers)
    print('length of train_batch_size: %d' % batch_size )
    print('prior: {}'.format(args.prior))
    print('use pseudo labels: {}'.format(args.p_dic))
    # print()
    # val_data_loader = data.DataLoader(val_dataset,
    #                                   batch_size=args.batch_size,  # must be 1
    #                                   num_workers=num_workers,
    #                                   pin_memory=True)
    # # val_snow_data_loader = data.DataLoader(val_noise_dataset_snow

    # print(len(val_data_loader))
    
    # import pdb; pdb.set_trace()
    if args.model == 'uncertainty-m':
        from models.model import Model
        model = Model(args.backbone, input_dim=args.input_channel, n=len(args.p_dic), use_unc=args.use_unc, use_nonlocal=args.use_nonlocal)
    else: raise NotImplementedError
    model = model.to(device)
    
    params_group = get_params_groups(model, weight_decay=args.weight_decay)
    optimizer = torch.optim.AdamW(params_group, lr=args.lr, betas=(0.9, 0.999), eps=1e-8,
                                  weight_decay=args.weight_decay)
    
    lr_scheduler = create_trival_lr_scheduler(optimizer, len(train_data_loader), args.epochs,
                                       warmup=True, warmup_epochs=1)
    
    
    #==========================set the dataloader====================================
    # import pdb; pdb.set_trace()
    dataset = {
        "source_train": SceneDataset(args.data_path, mode='train', cam_subdir='camera',  p_dic = args.p_dic, use_prior=args.use_prior),
        "target_train": SceneDataset(args.data_path, mode='test', cam_subdir=cor, p_dic = args.p_dic, use_prior=args.use_prior),
        "target_test": SceneDataset(args.data_path, mode='test', cam_subdir=cor, p_dic = args.p_dic, use_prior=args.use_prior),
        "source_test": SceneDataset(args.data_path, mode='test', p_dic = args.p_dic, use_prior=args.use_prior)
        
    }
    print("target train dataset len: {}".format(len(dataset["target_train"])))
    dataset_loader = {x: torch.utils.data.DataLoader(dataset[x], batch_size=args.batch_size, shuffle=True, num_workers=4)
            for x in ['source_train', 'target_train','target_test', 'source_test']}
    
    len_source = len(dataset_loader["source_train"]) - 1
    len_target = len(dataset_loader["target_train"]) - 1

    iter_source = iter(dataset_loader["source_train"])
    iter_target = iter(dataset_loader["target_train"])

    train_distribution_matching_loss = train_task_loss = train_total_loss = 0.0
    start_time = time.time()
    model = model.to(device)
    
    num_iter = int (len_source * args.epochs)
    warmup_num = int(num_iter / 3)
    test_interval = len_source
    interval_start_t = time.time()
    cc_max = 0.
    
    print('total iter: {}, warmup iter: {}, test interval: {}'.format(num_iter, warmup_num, test_interval))
    for iter_num in range(1, num_iter + 1):
        optimizer.zero_grad()
        model.train()
        
        if iter_num % len_source == 0:
            iter_source = iter(dataset_loader["source_train"])
        if iter_num % len_target == 0:
            iter_target = iter(dataset_loader["target_train"])
            
        data_source = iter_source.next()
        data_target = iter_target.next()

        inputs_target, _ = data_target    
        inputs_source, labels_source_list = data_source
        labels_source_list = [ps.to(device) for ps in labels_source_list]
        labels_source = (labels_source_list[0] + labels_source_list[1])/2
        
        inputs_source = inputs_source.float().to(device)
        inputs_target = inputs_target.float().to(device)
        labels_source = labels_source.float().to(device)    
        
        # here to get the last feats
        # import pdb; pdb.set_trace()
        pred_s, e, feat_s = model(inputs_source, labels_source_list)
        pred_t, feat_t = model(inputs_target)
        # feat_s = feat_s[-1]
        # feat_t = feat_t[-1]
        
        # supp_source = (1e-3) * torch.rand(labels_source.shape[0],4).to(device)
        # supp_target = (1e-3) * torch.rand(labels_target.shape[0],4).to(device)
        # y_source = torch.cat((labels_source, supp_source),dim=1)
        # y_target = torch.cat((pred_t, supp_target),dim=1)
        y_source = labels_source
        y_target = pred_t
        
        
        task_loss = criterion(pred_s, labels_source_list, e, args.loss_func)

        # if args.marg == 'daregram':
        # import pdb; pdb.set_trace()
        marginal_loss = DARE_GRAM_LOSS( feat_s, feat_t, device, args)
            
        cod_dist = COD_Metric(feat_s, feat_t, y_source.view(y_source.size(0), -1), y_target.view(y_target.size(0), -1).detach(), device) 
        conditional_loss = args.tradeoff_cod * cod_dist
        
        
        if iter_num < warmup_num:
            distribution_matching_loss = marginal_loss
        else:
            distribution_matching_loss = marginal_loss + conditional_loss
            
        total_loss = task_loss + distribution_matching_loss
        total_loss.backward()
        optimizer.step()
        
        lr_scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        
        
        ################################ record ###################################
        train_task_loss += task_loss.detach().item()
        train_distribution_matching_loss += distribution_matching_loss.detach().item()
        train_total_loss += total_loss.detach().item()

        if (iter_num % test_interval) == 0:
            interval_end_t = time.time()
            print("Iter {:05d}, Aver. Source: {:.4f}; Aver. Distr.: {:.4f};  Aver. Training Loss: {:.4f};  Time cost: {:.4f}s".format(
                iter_num, train_task_loss / float(test_interval), train_distribution_matching_loss / float(test_interval), 
                train_total_loss / float(test_interval), interval_end_t - interval_start_t))
            interval_start_t = time.time()

            model.eval()
            
            
            source_test_loader = dataset_loader['source_test']
            target_test_loader = dataset_loader['target_test']
            kld_metric, cc_metric = evaluate_batch(args, model, source_test_loader, device)
            kld_info, cc_info = kld_metric.compute(), cc_metric.compute()
            kld_snow_metric, cc_snow_metric = evaluate_batch(args, model, target_test_loader, device)
            kld_snow_info, cc_snow_info = kld_snow_metric.compute(), cc_snow_metric.compute()
            
            print(f"[iter: {iter_num}] val_kld: {kld_info:.3f} val_cc: {cc_info:.3f}")
            print(f"[iter: {iter_num}] snow_val_kld: {kld_snow_info:.3f} snow_val_cc: {cc_snow_info:.3f}")

            train_distribution_matching_loss = train_task_loss = train_total_loss = 0.0
            
    
     
        # loss, lr = train_trival_one_epoch(args, model, optimizer, train_data_loader, val_data_loader, None, device, epoch, lr_scheduler,
        #                                 print_freq=args.print_freq, scaler=None)
        # lr = optimizer.param_groups[0]["lr"]
        # lr_scheduler.step()

            save_file = {"model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "iter": num_iter,
                        "args": args}
    
            torch.save(save_file, Path(args.save_folder) / "model_{}_iter_{}.pth".format(args.name, iter_num))
        
            if cc_max  <= cc_info:
                torch.save(save_file, Path(args.save_folder) / "model_best_{}_{}_{}_{}_{}.pth".format(
                    args.name, "{:.5f}".format(cc_info), "{:.5f}".format(kld_info)  ,
                    "{:.5f}".format(cc_snow_info), "{:.5f}".format(kld_snow_info)))
                cc_max  = cc_info
            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print("training time {}".format(total_time_str))


if __name__ == '__main__':
    a = parse_args()

    if not os.path.exists("./save_weights"):
        os.mkdir("./save_weights")

    main(a)
    # wandb.agent(sweep_id, train, count=)
