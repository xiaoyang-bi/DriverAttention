import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import math
from torchvision import models
from torchvision import transforms
from datetime import datetime
import warnings
import os
import argparse
import time
from PIL import Image

warnings.filterwarnings("ignore")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


parser = argparse.ArgumentParser(description='PyTorch CKW experiment')

parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
parser.add_argument('--dset_name', type=str, default='MPI3D', metavar='S', help='name of the dataset: dSprites, MPI3D or BK')
parser.add_argument('--marg', type=str, default='daregram', metavar='S', help='name of the marginal loss: daregram, rsd, kgw or any other')
parser.add_argument('--src', type=str, default='t', metavar='S', help='source dataset')
parser.add_argument('--tgt', type=str, default='rl', metavar='S', help='target dataset')

parser.add_argument('--lr', type=float, default=1e-1, help='init learning rate for fine-tune')
parser.add_argument('--gamma', type=float, default=0.0001, help='learning rate decay')
parser.add_argument('--epsilon', type=float, default=5e-2, help='regularization for kernel matrix inversion')
parser.add_argument('--warmup_num', type=int, default=3000, help='before this threshold, marginal only; after, marginal + conditional')

parser.add_argument('--tradeoff_angle', type=float, default=0.5,
                        help='tradeoff for angle alignment')
parser.add_argument('--tradeoff_scale', type=float, default=0.003,
                        help='tradeoff for scale alignment')
parser.add_argument('--treshold', type=float, default=0.999,
                        help='treshold for the pseudo inverse')

parser.add_argument('--tradeoff_kgw', type=float, default=1e-4, help='tradeoff for kgw dist')
parser.add_argument('--tradeoff_cod', type=float, default=1e-3, help='tradeoff for cod dist')

parser.add_argument('--num_iter', type=int, default=10000, help='total number of iter')
parser.add_argument('--batch', type=int, default=36, help='batch size')
parser.add_argument('--test_interval', type=int, default=500, help='interval for testing')
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def make_dataset(image_list, labels):
    if labels:
        len_ = len(image_list)
        images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
        if len(image_list[0].split()) > 2:
            images = [(val.split()[0], np.array([float(la) for la in val.split()[1:]])) for val in image_list]
        else:
            images = [(val.split()[0], float(val.split()[1])) for val in image_list]
    return images

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

class ImageList(object):
    """A generic data loader where the images are arranged in this way: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        label_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, image_list, labels=None, transform=None, label_transform=None,
                 loader=pil_loader):
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images !!!"))

        self.imgs = imgs
        self.transform = transform
        self.label_transform = label_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.label_transform is not None:
            target = self.label_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)

def data_transforms(resize_size=(256, 256)):
    return transforms.Compose([
    transforms.Resize(resize_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

class label_transforms:
    def __init__(self, dset_name='MPI3D'):
        """
        dset_name should be dSprites, MPI3D or BK 
        """
        self.dset_name = dset_name

    def __call__(self, label):
        if self.dset_name == 'MPI3D':
            label = torch.tensor(label, dtype=float)
            label = label.float() / 39

        return label

def inv_lr_scheduler(param_lr, optimizer, iter_num, gamma, power, init_lr=0.001, weight_decay=0.0005):
    lr = init_lr * (1 + gamma * iter_num) ** (-power)
    i = 0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_lr[i]
        param_group['weight_decay'] = weight_decay
        i += 1
    return optimizer

def RSD_LOSS(Feature_s, Feature_t):
    u_s, s_s, v_s = torch.svd(Feature_s.t())
    u_t, s_t, v_t = torch.svd(Feature_t.t())
    p_s, cospa, p_t = torch.svd(torch.mm(u_s.t(), u_t))
    sinpa = torch.sqrt(1-torch.pow(cospa,2))
    return args.tradeoff_angle*torch.norm(sinpa,1)+args.tradeoff_scale*torch.norm(torch.abs(p_s) - torch.abs(p_t), 2)

def DARE_GRAM_LOSS(H1, H2):    
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

def KGW_Metric(fea_s, fea_t, device):
    num_sam_s = fea_s.shape[0]
    num_sam_t = fea_t.shape[0]

    one_s = torch.ones(num_sam_s, 1).to(device)
    one_t = torch.ones(num_sam_t, 1).to(device)

    H_s = ( torch.eye(num_sam_s) - torch.ones(num_sam_s)/num_sam_s ).to(device)
    H_t = ( torch.eye(num_sam_t) - torch.ones(num_sam_t)/num_sam_t ).to(device)

    K_XsXs, K_XtXt, K_XsXt, K_XtXs = mul_guassian_kernel(fea_s, fea_t)

    kst_rank = torch.linalg.matrix_rank(K_XsXt)
    if kst_rank < 36: # anomaly detection
        return torch.tensor(0.0).to(device)
    
    nuclear = (H_s.mm(K_XsXt)).mm(H_t)
    nc = torch.norm(nuclear,'nuc')

    KGW = (1/num_sam_s) * K_XsXs.trace() + (1/num_sam_t) * K_XtXt.trace() - (2/(num_sam_s*num_sam_t)) * ((one_s.t()).mm(K_XsXt)).mm(one_t) - (2/(math.sqrt(num_sam_s*num_sam_t))) * nc
    return KGW

def Regression_test(dataloader, model):
    MAE = [0, 0, 0]
    number = 0
    with torch.no_grad():
        for (imgs, labels) in dataloader:
            imgs = imgs.to(device)
            labels = labels.float().to(device)

            _, pred = model(imgs)
            MAE[0] += torch.nn.L1Loss(reduction='sum')(pred[:, 0], labels[:, 0])
            MAE[1] += torch.nn.L1Loss(reduction='sum')(pred[:, 1], labels[:, 1])
            MAE[2] += torch.nn.L1Loss(reduction='sum')(pred, labels)
            number += imgs.size(0)
    for j in range(3):
        MAE[j] = MAE[j] / number
    print("\tMAE : {0},{1}\n".format(MAE[0], MAE[1]))
    print("\tMAEall : {0}\n".format(MAE[2]))

    return MAE[2]

class Resnet18Fc(nn.Module):
    def __init__(self):
        super(Resnet18Fc, self).__init__()
        model_resnet18 = models.resnet18(pretrained=True)
        self.conv1 = model_resnet18.conv1
        self.bn1 = model_resnet18.bn1
        self.relu = model_resnet18.relu
        self.maxpool = model_resnet18.maxpool
        self.layer1 = model_resnet18.layer1
        self.layer2 = model_resnet18.layer2
        self.layer3 = model_resnet18.layer3
        self.layer4 = model_resnet18.layer4
        self.avgpool = model_resnet18.avgpool
        self.__in_features = model_resnet18.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def output_num(self):
        return self.__in_features
    
class Model_Regression(nn.Module):
    def __init__(self, dim=512):
        super(Model_Regression,self).__init__()
        self.backbone = Resnet18Fc()
        self.regressor = nn.Linear(dim, 2)
        self.regressor.weight.data.normal_(0, 0.01)
        self.regressor.bias.data.fill_(0.0)
        self.predictor = nn.Sequential(self.regressor,  nn.Sigmoid())

    def forward(self,x):
        feature = self.backbone(x)
        pred = self.predictor(feature)
        return feature, pred

for exp_iter in {1}: #range(1,10):
    # set dataset
    batch_size = {"source_train": args.batch, "target_train": args.batch,"target_test": args.batch}#, "source_test": args.batch}
    rc="realistic.txt"
    rl="real.txt"
    t="toy.txt"

    rc_t="realistic_test.txt"
    rl_t="real_test.txt"
    t_t="toy_test.txt"

    if args.src =='rl':
        source_path = rl
    elif args.src =='rc':
        source_path = rc
    elif args.src =='t':
        source_path = t

    if args.tgt =='rl':
        target_path = rl
        target_path_t = rl_t
    elif args.tgt =='rc':
        target_path = rc
        target_path_t = rc_t
    elif args.tgt =='t':
        target_path = t
        target_path_t = t_t
       

    dataset = {
        "source_train": ImageList(open(source_path).readlines(), transform=data_transforms(224), label_transform=label_transforms(dset_name=args.dset_name)),
        "target_train": ImageList(open(target_path).readlines(),transform=data_transforms(224), label_transform=label_transforms(dset_name=args.dset_name)),
        "target_test": ImageList( open(target_path_t).readlines(),transform=data_transforms(224), label_transform=label_transforms(dset_name=args.dset_name))
        }

    dataset_loader = {x: torch.utils.data.DataLoader(dataset[x], batch_size=batch_size[x],shuffle=True, num_workers=4)
                for x in ['source_train', 'target_train','target_test']}

    Model_R = Model_Regression()
    Model_R = Model_R.to(device)
    
    optimizer = torch.optim.SGD([
                {'params': Model_R.backbone.parameters(), 'lr': args.lr},
                {'params': Model_R.regressor.parameters(), 'lr': 10*args.lr}
            ],  momentum=0.9, weight_decay=0.0005, nesterov=True)

    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])
        
    len_source = len(dataset_loader["source_train"]) - 1
    len_target = len(dataset_loader["target_train"]) - 1

    iter_source = iter(dataset_loader["source_train"])
    iter_target = iter(dataset_loader["target_train"])

    train_distribution_matching_loss = train_task_loss = train_total_loss = 0.0
    MSE_loss = nn.MSELoss()

    print(args)
    start_time = datetime.now()
    print("************ %1s→ %1s: %1s Start Experiment %1s training ************"%(args.src, args.tgt, start_time, exp_iter))
    interval_start_t = time.time()
    for iter_num in range(1, args.num_iter + 1):
        
        optimizer = inv_lr_scheduler(param_lr, optimizer, iter_num, init_lr=args.lr, gamma=args.gamma, power=0.75,weight_decay=0.0005)
        optimizer.zero_grad()
        
        Model_R.train()

        if iter_num % len_source == 0:
            iter_source = iter(dataset_loader["source_train"])
        if iter_num % len_target == 0:
            iter_target = iter(dataset_loader["target_train"])
            
        data_source = iter_source.next()
        data_target = iter_target.next()

        inputs_target, labels_target = data_target    
        inputs_source, labels_source = data_source
        inputs_source = inputs_source.float().to(device)
        inputs_target = inputs_target.float().to(device)
        labels_source = labels_source.float().to(device)
        
        feat_s, pred_s = Model_R(inputs_source)
        feat_t, pred_t = Model_R(inputs_target)
        # y_source = labels_source 
        # y_target = pred_t

        # optional; relieve the ill-posedness of kernel matrix inversion
        supp_source = (1e-3) * torch.rand(labels_source.shape[0],4).to(device)
        supp_target = (1e-3) * torch.rand(labels_target.shape[0],4).to(device)
        y_source = torch.cat((labels_source, supp_source),dim=1)
        y_target = torch.cat((pred_t, supp_target),dim=1)

        ##################################### loss ######################################
        regression_loss = MSE_loss(labels_source, pred_s)

        if args.marg == 'daregram':
            marginal_loss = DARE_GRAM_LOSS(feat_s, feat_t)
        elif args.marg == 'rsd':
            rsd_dist = RSD_LOSS(feat_s, feat_t, device)
            marginal_loss = args.tradeoff_kgw * rsd_dist
        elif args.marg == 'kgw':
            kgw_dist = KGW_Metric(feat_s, feat_t, device)
            marginal_loss = args.tradeoff_kgw * kgw_dist
        else:
            print("No marginal loss")
            marginal_loss = torch.tensor(0.0).to(device)

        cod_dist = COD_Metric(feat_s, feat_t, y_source, y_target.detach(), device) 
        conditional_loss = args.tradeoff_cod * cod_dist

        if iter_num < args.warmup_num:
            distribution_matching_loss = marginal_loss
        else:
            distribution_matching_loss = marginal_loss + conditional_loss

        total_loss = regression_loss + distribution_matching_loss

        total_loss.backward()
        optimizer.step()
        
        ################################ record ###################################
        train_task_loss += regression_loss.detach().item()
        train_distribution_matching_loss += distribution_matching_loss.detach().item()
        train_total_loss += total_loss.detach().item()

        if (iter_num % args.test_interval) == 0:
            interval_end_t = time.time()
            print("Iter {:05d}, Aver. Source: {:.4f}; Aver. Distr.: {:.4f};  Aver. Training Loss: {:.4f};  Time cost: {:.4f}s".format(
                iter_num, train_task_loss / float(args.test_interval), train_distribution_matching_loss / float(args.test_interval), 
                train_total_loss / float(args.test_interval), interval_end_t - interval_start_t))
            interval_start_t = time.time()

            Model_R.eval()
            mae = Regression_test(dataset_loader["target_test"], Model_R)
            train_distribution_matching_loss = train_task_loss = train_total_loss = 0.0

    end_time = datetime.now()
    seconds = (end_time - start_time).seconds
    minutes = seconds//60
    second = seconds%60
    hours = minutes//60
    minute = minutes%60
    print("************ %1s→ %1s: %1s End Experiment %1s training ************"%(args.src, args.tgt, end_time, exp_iter))
    print('Total TIme Cost: {} Hour {} Minutes {} Seconds'.format(hours,minute,second))