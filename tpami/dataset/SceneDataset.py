import os.path
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import cv2
from pathlib import Path
import random
from torch.utils.data import DataLoader
from tqdm import tqdm


class SceneDataset(Dataset):
    def __init__(self, root: str, mode: str,
                  alpha: float = 0.3,
                  severity:str = None,
                  cam_subdir = 'camera',
                  out_folder = 'infer_gaze',
                  infer_gaze_subdir = 'infer_gaze',
                  gaze_subdir = 'gaze',
                  mask_subdir = 'masks',
                  p_dic = None,
                  sample_num = -1,
                  noise_type:str = None,
                  use_prior=True):
        '''
        mode should include:
        train, val, test, infer, run_example
        '''
        self.mode:str = mode

        self.file_scene_list:list = []
        self.out_folder = out_folder
        self.cam_subdir = cam_subdir
        self.mask_subdir = mask_subdir
        self.gaze_subdir = gaze_subdir
        self.severity = severity
        self.noise_type = noise_type
        self.p_dic:list = p_dic
        # self.use_msk = alpha > 0
        # self.alpha =alpha
        self.infer_gaze_subdir = infer_gaze_subdir
        self.gaze_average = None
        # import pdb; pdb.set_trace()
        # self.prior = prior
        self.use_prior = use_prior
        # self.kl_db:dict = None


        assert os.path.exists(root), f"path '{root}' does not exists."
        self.root = Path(root)
        phase = mode
        if mode.startswith('infer'):
            phase = 'train'
        scene_list_path = self.root/(phase + '.txt') if self.noise_type is None else  self.root/(phase + '_cor.txt')
        # scene_list_path = self.root/('train' + '.txt') 
        
        self.scenes = [self.root/folder.strip()
                    for folder in open(scene_list_path) if not folder.strip().startswith("#")]
        if os.path.exists(str(self.scenes[0]/(self.cam_subdir + '_224_224'))):
                self.cam_subdir = self.cam_subdir + "_224_224"
                print("use resized")
        if os.path.exists(str(self.scenes[0]/(self.gaze_subdir + '_224_224'))):
                self.gaze_subdir = self.gaze_subdir + "_224_224"
                print("use resized")      
        if self.p_dic is not None:
            for p in self.p_dic:
                if os.path.exists(str(self.scenes[0]/(p + '_224_224'))):
                        p += "_224_224"
                        print("use resized")  

        if self.noise_type is  None:
            for scene in self.scenes:
                file_list = sorted(
                    list((scene/self.cam_subdir).glob('*')))
                file_scene_list = [[x.name, scene.name] for x in file_list if x.suffix in ('.png', '.jpg', '.jpeg')]
                self.file_scene_list = self.file_scene_list + file_scene_list
        else:
            if not os.path.exists(str(self.scenes[0]/(self.noise_type))):
                self.noise_type = self.noise_type + "_224_224"
            self.out_folder += ('_' + self.noise_type)

            print("Noise Loader")
            for scene in self.scenes:
                file_list = sorted(
                    list((scene/self.noise_type).glob('*')))
                file_scene_list = [[x.name, scene.name] for x in file_list if x.suffix in ('.png', '.jpg', '.jpeg')]
                self.file_scene_list = self.file_scene_list + file_scene_list

        if sample_num != -1:
            print(len(self.file_scene_list))
            if sample_num > len(self.file_scene_list ):
                raise ValueError("sample_num is larger than the length of the list")
            self.file_scene_list =  random.sample(self.file_scene_list, sample_num)
        # self.file_scene_list = self.file_scene_list[:10] #DEBUG
        self.init_file_scene_list = self.file_scene_list
        # self.file_scene_list = self.file_scene_list[:1000]

    def change_mode(self, mode):
        self.mode = mode
        
        
    def get_mall_by_prior(self, scene, file):
        mall = torch.zeros(1, 224, 224)
        masks_dir=self.root / scene / 'masks_stat' / file
        masks_path = sorted(
                    list((masks_dir).glob('*')))

        # category_sums = {}
        for msk_p in masks_path:
            category = msk_p.stem.split('_')[0]  
            if category in self.prior:
                # import pdb; pdb.set_trace()
                # print(category)
                msk = self.convert(msk_p)
                mall = torch.logical_or(mall, msk)
            else:
                continue
        return mall
    
    def get_weighted_prior(self, scene, file):
        # ratio_dic = {'car': 0.5638977998074998, 'truck': 0.2269732635665115, 'backpack': 0.21311985196308783, 'pedestrian': 0.20617415598709493, 'traffic light': 0.1850667140787043, 'bus': 0.18421525132774222, 'bicycle': 0.1781600717821623, 'tie': 0.1377069540321827, 'motorcycle': 0.13349173879026038, 'handbag': 0.13124596281810588, 'horse': 0.12889326948465574, 'stop sign': 0.12483922368060399, 'dog': 0.10880823520404335, 'boat': 0.10577008263850264, 'snowboard': 0.0892871841788292, 'umbrella': 0.08660523378730556, 'oven': 0.08596869930624962, 'tennis racket': 0.08570528030395508, 'bed': 0.08233067393302917, 'cow': 0.08198763278778642, 'cell phone': 0.07620838226284832, 'kite': 0.0735354639408696, 'baseball bat': 0.0718070297235889, 'suitcase': 0.06551188474555535, 'sports ball': 0.06154689057816372, 'skateboard': 0.06127312399698176, 'banana': 0.060324782971292734, 'fire hydrant': 0.058528549239196134, 'bench': 0.05475237408741351, 'parking meter': 0.05449956074685377, 'elephant': 0.05362171326608708, 'chair': 0.047383366278760754, 'bird': 0.0424957452650365, 'refrigerator': 0.037612475291825834, 'train': 0.03581248259029558, 'surfboard': 0.03540474381297827, 'bottle': 0.03334296714073341, 'potted plant': 0.031974993769836146, 'knife': 0.031571838958188894, 'clock': 0.02853374809737116, 'tv': 0.025223881820224166, 'dining table': 0.024262551218271255, 'bowl': 0.023461395194754004, 'sheep': 0.017924360930919647, 'vase': 0.017169343191199005, 'airplane': 0.015998620631993372, 'broccoli': 0.01333126937970519, 'cup': 0.01049697454760058, 'frisbee': 0.010140785947442055, 'cat': 0.005902743432670832, 'microwave': 0.005191194824874401, 'book': 0.0036249628756195307, 'keyboard': 0.0034132998043787666, 'mouse': 0.003137772936107857, 'toilet': 0.0027635309379547834, 'sink': 0.0025732512502664967, 'fork': 0.0020925351418554783}
        ratio_dic = {'car': 0.0, 'person': 0.18122506607338648, 'stop sign': 0.19527073080923388, 'traffic light': 0.2616087945898801, 'truck': 0.17388554465132544, 'bicycle': 0.16350374183093028, 'bus': 0.16153357425939865, 'motorcycle': 0.19183225919958863, 'bench': 0.10643748839719337, 'dog': 0.05576605004173327, 'backpack': 0.09777960469388935, 'fire hydrant': 0.06474239858484047, 'handbag': 0.08763731467521027, 'parking meter': 0.16562452863722288, 'potted plant': 0.06092319412051039, 'cup': 0.11359222911416274, 'skateboard': 0.09840303970349075, 'train': 0.20691480532578713, 'tv': 0.10791136824738431, 'clock': 0.1144762219590874, 'cell phone': 0.13222140138030689, 'umbrella': 0.11024437257081295, 'bird': 0.08634613909001014, 'boat': 0.18348954395743725, 'kite': 0.03687248143775576, 'suitcase': 0.09635855838609045, 'bowl': 0.0666870674151306, 'chair': 0.05954423152044212, 'frisbee': 0.084945172864205, 'cow': 0.06745842585058782, 'airplane': 0.12468087536177429, 'horse': 0.03908503110024358, 'sink': 0.06261205817910573, 'surfboard': 0.10702988498317394, 'keyboard': 0.0785430556265317, 'mouse': 0.024297037773953804, 'sports ball': 0.14885588348762518, 'elephant': 0.05967972605245516, 'tennis racket': 0.0650258207044329, 'banana': 0.10106796738261853, 'baseball bat': 0.017516029519683153, 'bottle': 0.07283924099191112, 'broccoli': 0.0, 'book': 0.03529503115137245, 'toilet': 0.07208270054966644, 'fork': 0.11649917077285467, 'vase': 0.04720483379038387, 'knife': 0.0703967826049966, 'refrigerator': 0.026514250192185052, 'cat': 0.0, 'microwave': 0.07097810306040395, 'tie': 0.0603520078010561, 'oven': 0.008240488923692621, 'snowboard': 0.11479120579603207, 'bed': 0.03127203895436038, 'dining table': 0.02025312469653382, 'sheep': 0.03619519410151036}
        mall = torch.zeros(1, 224, 224)
        masks_dir=self.root / scene / 'masks_stat' / file
        masks_path = sorted(
                    list((masks_dir).glob('*')))
        
        # import pdb; pdb.set_trace()
        for msk_p in masks_path:
            category = msk_p.stem.split('_')[0]  
            # if category == 'traffic light':
            #     import pdb; pdb.set_trace()
            msk = self.convert(msk_p)
            index = ratio_dic[category] if category in ratio_dic else 0
            mall = mall + msk * index
            # else:
            #     continue
        # mall = (mall - mall.min()) / (mall.max() - mall.min() + 1e-5) 
        # mall = torch.sigmoid(mall)
        return mall
    
    def get_pseudos(self, scene, file):
        p = []
        if self.use_prior:
            mall_path = self.root / scene / 'weighted_mask_wo_norm' / (file.split('.')[0] + '.npy')
            if mall_path.exists():
                mall = torch.from_numpy(np.load(mall_path))
            else:
                mall = self.get_weighted_prior(scene, file)
        
        for p_type in self.p_dic:
            pesudo_path = str(self.root / scene / p_type / file)
            ps = self.convert(pesudo_path)
            if self.use_prior:
                ps = ps * (mall+1)
            ps /= ps.max()
            p.append(ps)
        return p
        
    def __getitem__(self, i):
        file, scene = self.file_scene_list[i]
        if self.noise_type:
            img_path = str(self.root / scene / self.noise_type / file)
        else:
            img_path = str(self.root / scene / self.cam_subdir / file)
        img = self.convert(img_path, True)
        # import pdb; pdb.set_trace()
        if self.mode == 'train':
            # import pdb; pdb.set_trace()
            # mall = self.get_weighted_prior(scene, file)
            # mall_out_dir = self.root / scene / 'weighted_mask_wo_norm' 
            # mall_out_dir.mkdir(parents=True, exist_ok=True)
            # mall_out_path = mall_out_dir / (file.split('.')[0] + '.npy')
            # np.save(mall_out_path, mall.numpy())
            p = self.get_pseudos(scene, file)
            return img, p
        elif self.mode.startswith('val') or self.mode == 'test':
            gaze_path = str(self.root / scene / self.gaze_subdir / file)
            extensions = ['.png', '.jpg', '.jpeg']

            # Iterate through the extensions and check if the file exists
            for ext in extensions:
                gaze_path = str(self.root / scene / self.gaze_subdir / file).rsplit('.', 1)[0] + ext
                if os.path.exists(gaze_path):
                    break
            gaze  = self.convert(gaze_path)
            return img, gaze
        
        elif self.mode == 'run_example' or self.mode =='infer' or self.mode=='test_nobias':
            output_folder = str(self.root / scene / self.out_folder)
            os.makedirs(output_folder, exist_ok=True)
            out_test_path = str(self.root / scene / self.out_folder / file)
            return img, out_test_path
        elif self.mode == 'infer_mix':
            output_folder = str(self.root / scene / self.out_folder)
            os.makedirs(output_folder, exist_ok=True)
            out_test_path = str(self.root / scene / self.out_folder / file)
            p = self.get_pseudos(scene, file)
            # p = []
            # for p_type in self.p_dic:
            #     pesudo_path = str(self.root / scene / p_type / file)
            #     ps = self.convert(pesudo_path)
            #     # if self.use_msk:
            #     mask_path = str(self.root / scene / self.mask_subdir / file)
            #     mall = self.convert(mask_path)
            #     ps = ps * (mall + self.alpha)
            #     ps /= ps.max()
            #     p.append(ps)
            return img, p, out_test_path
        elif self.mode == 'cal':
            gaze_path = str(self.root / scene / self.infer_gaze_subdir / file)
            ext = '.jpg'
            gaze_path = str(self.root / scene / self.infer_gaze_subdir / file).rsplit('.', 1)[0] + ext
            gaze  = self.convert(gaze_path)
            return gaze

        else: 
            raise(NotImplementedError)
        

    def set_gaze_average(self, gaze_average):
        self.gaze_average = gaze_average

    def cal_data_kl(self, name)->dict:
        '''
        only used for the init datasets#TODO modify it to cuda
        '''
        mode_saved = self.mode
        batch_size = 32
        num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 16])
        self.mode = 'cal'
        loader = DataLoader(self, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        # if self.gaze_average is None:
        sum_gaze = 0
        total_samples = 0
        # gaze_average = None
        for gazes in loader:
            gazes = gazes.cuda()
            batch_sum = gazes.sum(dim=0) 
            sum_gaze += batch_sum
            total_samples += gazes.size(0)  
                
        self.gaze_average = sum_gaze / total_samples
        out = self.gaze_average.permute(1, 2, 0).cpu().detach().numpy() * 255
        out = out.astype(np.uint8)
        cv2.imwrite(name + '_gaze_average.jpg', out)

        kl_db = {}

        for gazes in loader:
            gazes = gazes.cuda()
            for gaze in gazes:
                kl = kldiv(gaze, self.gaze_average).item()
                rounded_kl = round(kl, 1)
                if rounded_kl in kl_db:
                    kl_db[rounded_kl] += 1
                else:
                    kl_db[rounded_kl] = 1

        self.mode = mode_saved
        return kl_db, self.gaze_average
    


        # for  file, scene in self.init_file_scene_list:

        #     ext = '.jpg'
        #     gaze_path = str(self.root / scene / self.infer_gaze_subdir / file).rsplit('.', 1)[0] + ext
        #     gaze  = self.convert(gaze_path).cuda()
        #     kl = kldiv(gaze_average, gaze).item()
        #     rounded_kl = round(kl, 1)
        #     if rounded_kl in kl_db  and  kl_db[rounded_kl] < max_value and rounded_kl > key_max:
        #         kl_db[rounded_kl] += 1
        #         selected_file_scene_list.append([file, scene])
        # self.file_scene_list = selected_file_scene_list
        # return kl_db



    def __len__(self):
        # return 100
        return len(self.file_scene_list)

    @staticmethod
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
            # if is_msk:
            #     # print('true')
            #     # x[x == 6] = 255
            #     roi = np.array([6, 7, 11, 17, 18])
            #     x = np.isin(x, roi)
                # x = x.astype('int')
                # print(x)
            x = x.astype('float')
            
            if ( w != 224 or h != 224):
                x = cv2.resize(x, (224, 224))

            if np.max(x) > 1.0:
                x = x / 255.0
            x = torch.FloatTensor(x).unsqueeze(0)

        return x


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

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from torchvision.utils import save_image
    import cv2
    from pathlib import Path
    
    def gen_heatmap(img, gray_img):
        """
        Generate a heatmap from the normalized gray image and overlay it on the original image.

        Args:
            img (numpy.ndarray): Original image of shape [3, h, w].
            gray_img (numpy.ndarray): Normalized gray image of shape [1, h, w].

        Returns:
            numpy.ndarray: Blended image with heatmap overlay of shape [3, h, w].
        """
        gray_img = (gray_img / gray_img.max() * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(gray_img[0], cv2.COLORMAP_JET)
        
        img = (img - img.min()) / (img.max() - img.min()) * 255
        img = img.astype(np.uint8)
        heatmap = cv2.resize(heatmap, (img.shape[2], img.shape[1]))

        blended_image = cv2.addWeighted(img.transpose(1, 2, 0), 0.6, heatmap, 0.4, 0)
        blended_image = blended_image.transpose(2, 0, 1)
        
        return blended_image
    # dataset = SceneDataset("../../atten_data/bd", mode='train',  alpha=0.3, p_dic = ['ml_p', 'unisal_p'], prior=
    #                        [ "motorcycle", "person", "stop sign", "traffic light", "fire hydrant"])
    dataset = SceneDataset("../../atten_data/bd", mode='train', p_dic = ['ml_p', 'unisal_p'])
    # import pdb; pdb.set_trace()
    for i in range(1000):
        _ = dataset[i]
    # dataloader = DataLoader(dataset,
    #                         batch_size=32,
    #                         num_workers=8,
    #                         shuffle=True,
    #                         pin_memory=True)
    # for data in tqdm(dataloader):
    #     continue
    # print(len(dataset))
    # import pdb; pdb.set_trace()
    # output_dir = Path('output')/'norm'
    # output_dir.mkdir(parents=True, exist_ok=True)
    
    # for i in range(100000):
    #     _ = dataset[i]

    # for i in range(100000):
    #     img, ps = dataset[i]
    #     img = img.cpu().numpy()
    #     for p in ps:
    #         p = p.cpu().numpy()
    #         heatmap = gen_heatmap(img, p)
            
    #         cv2.imwrite(output_dir/"{}.jpg".format(i), heatmap.transpose(1, 2, 0))
        
    # for i in range(100000):
    #     img, ps = dataset[i]
    #     img = img.cpu().numpy()
    #     # for p in ps:
        
    #     ps = ps.cpu().numpy()
    #     heatmap = gen_heatmap(img, ps)
        
    #     cv2.imwrite(output_dir/"{}.jpg".format(i), heatmap.transpose(1, 2, 0))
        
    #     # Normalize img if necessary (assuming it's in the range [0, 1])
    #     # img = img * 255
    #     # Normalize p if necessary (assuming it's in the range [0, 1])
    #     for ps in p:
    #         ps = ps * 255
    #         # Convert p from [1, 224, 224] to [3, 224, 224] for JPEG compatibility
    #         # Assuming p is a grayscale image, we replicate the single channel across RGB
    #         ps = ps.repeat(3, 1, 1)

    #     # Save img and p as JPEG
    #     save_image(img, f'datasettest/img_{i}.jpg')
    #     save_image(p[0], f'datasettest/samp_{i}.jpg')
    #     save_image(p[1], f'datasettest/unip_{i}.jpg')
        # save_image(mall, f'datasettest/mall_{i}.jpg')






    # test_dataset = RadiateDataset("/autodl-tmp/atten_dataset/dr/", mode='test')
    # print(len(test_dataset))
    # img = test_dataset[0]

    # i, t = train_dataset[0]
