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
import matplotlib.pyplot as plt

class SceneDataset(Dataset):
    def __init__(self, root: str, mode: str,
                  alpha: float = 0.3,
                  severity:str = None,
                  cam_subdir = 'camera',
                  out_folder = 'infer_gaze',
                  infer_gaze_subdir = 'infer_gaze',
                  gaze_subdir = 'gaze',
                  mask_subdir = 'masks',
                  sample_num = 10,
                  p_dic = None,
                  noise_type:str = None,
                  prior=None):
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
        self.use_msk = alpha > 0
        self.alpha =alpha
        self.infer_gaze_subdir = infer_gaze_subdir
        self.gaze_average = None
        # import pdb; pdb.set_trace()
        self.prior = prior
        # self.kl_db:dict = None


        assert os.path.exists(root), f"path '{root}' does not exists."
        self.root = Path(root)
        phase = mode
        # if mode.startswith('infer'):
        #     phase = 'train'
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
        self.sample_num = sample_num
        # if sample_num != -1:
        #     print(len(self.file_scene_list))
        #     if sample_num > len(self.file_scene_list ):
        #         raise ValueError("sample_num is larger than the length of the list")
        #     self.file_scene_list =  random.sample(self.file_scene_list, sample_num)
        # # self.file_scene_list = self.file_scene_list[:10] #DEBUG
        # self.init_file_scene_list = self.file_scene_list
        # self.file_scene_list = self.file_scene_list[:1000]
        
    def __getitem__(self, i):
        ps_candidates = ['ml_p', 'unisal_p', 'sam_p', 'tased_p']
        file, scene = self.file_scene_list[i]

        p_list = []    
        for p_type in ps_candidates:
            pesudo_path = str(self.root / scene / p_type / file)
            ps = self.convert(pesudo_path)
            ps = ps/ps.max()
            p_list.append(ps)

        num = len(p_list)
        matrix = torch.zeros(num, num)
        for i in range(num):
            for j in range(num):
                if i != j:
                    tensor_i_flat = p_list[i].flatten()
                    tensor_j_flat = p_list[j].flatten()
                    
                    # 计算相关系数
                    mean_i = torch.mean(tensor_i_flat)
                    mean_j = torch.mean(tensor_j_flat)
                    tensor_i_centered = tensor_i_flat - mean_i
                    tensor_j_centered = tensor_j_flat - mean_j
                    
                    covariance = torch.sum(tensor_i_centered * tensor_j_centered)
                    std_i = torch.sqrt(torch.sum(tensor_i_centered ** 2))
                    std_j = torch.sqrt(torch.sum(tensor_j_centered ** 2))
                    
                    cc = covariance / (std_i * std_j)
                    matrix[i, j] = 1 - cc

        return matrix
            
       
    


    def __len__(self):
        # return 1000
        return self.sample_num

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
            x = x.astype('float')
            
            if ( w != 224 or h != 224):
                x = cv2.resize(x, (224, 224))

            if np.max(x) > 1.0:
                x = x / 255.0
            x = torch.FloatTensor(x).unsqueeze(0)

        return x


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from torchvision.utils import save_image
    import cv2
    from pathlib import Path
    
    sample_num = 1000
    dataset = SceneDataset("../../atten_data/bd", mode='train', p_dic = ['ml_p', 'unisal_p'], sample_num=sample_num)
    matrix_res = torch.zeros(4, 4)
    from tqdm import tqdm
    
    dataloader = DataLoader(dataset,
                            batch_size=32,
                            num_workers=8,
                            shuffle=True,
                            pin_memory=True)
    # import pdb; pdb.set_trace()
    for data in tqdm(dataloader):
        matrix_res += data.sum(dim=0)

    # for i in tqdm(range(sample_num)):
    #     matrix = dataset[i]
    #     matrix_res += matrix

    matrix_res /= sample_num
    torch.save(matrix_res, 'matrix_res.pt')

    plt.imshow(matrix_res.numpy(), cmap='hot', interpolation='nearest')
    plt.colorbar()

    plt.xticks(ticks=range(4), labels=['M', 'U', 'S', 'T'])
    plt.yticks(ticks=range(4), labels=['M', 'U', 'S', 'T'])

    plt.title('Average Difference Matrix')
    plt.savefig('matrix_res_heatmap.png')
    plt.close()
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
