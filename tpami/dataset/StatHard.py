import os.path
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import cv2
from pathlib import Path
import random
from tqdm import tqdm


class StatHard(Dataset):
    def __init__(self, root: str, mode: str,
                  alpha: float = 0.3,
                  severity:str = None,
                  cam_subdir = 'camera',
                  out_folder = 'infer_gaze',
                  gaze_subdir = 'gaze',
                  mask_subdir = 'masks',
                  sample_num = -1,):
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
        # self.kl_db:dict = None


        assert os.path.exists(root), f"path '{root}' does not exists."
        self.root = Path(root)
        scene_list_path = self.root/'test.txt'
        self.scenes = [self.root/folder.strip()
                    for folder in open(scene_list_path) if not folder.strip().startswith("#")]
        if os.path.exists(str(self.scenes[0]/(self.cam_subdir + '_224_224'))):
                self.cam_subdir = self.cam_subdir + "_224_224"
                print("use resized")
        if os.path.exists(str(self.scenes[0]/(self.gaze_subdir + '_224_224'))):
                self.gaze_subdir = self.gaze_subdir + "_224_224"
                print("use resized")      

        for scene in self.scenes:
            file_list = sorted(
                list((scene/self.cam_subdir).glob('*')))
            file_scene_list = [[x.name, scene.name] for x in file_list if x.suffix in ('.png', '.jpg', '.jpeg')]
            self.file_scene_list = self.file_scene_list + file_scene_list
               
    
    def __getitem__(self, i):
        file, scene = self.file_scene_list[i]
        ext = '.jpg'
        gaze_path = str(self.root / scene / self.gaze_subdir / file).rsplit('.', 1)[0] + ext
        if self.mode == 'stat':
            gaze = self.convert(gaze_path, resize=False)
            return gaze
        img_path = str(self.root / scene / self.cam_subdir / file)
        img = self.convert(img_path, True)
        gaze  = self.convert(gaze_path)
        return img, gaze, gaze_path



    def __len__(self):
        return len(self.file_scene_list)

    @staticmethod
    def convert(x_path, is_rgb=False, resize=True):
        transform_with_resize = transforms.Compose([
            transforms.Resize((224, 224)), # Resize the shorter side to 224, maintaining aspect ratio
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        # transform_with_resize = transforms.Compose([transforms.Resize((224, 224)),
        #                                 transforms.ToTensor(),
        #                                 transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
        transform_wo_resize = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
        x = Image.open(x_path)
        w, h = x.size
        # print(x.size)

        if is_rgb:
            x = x.convert('RGB')
            if ( w == 224 and h == 224):
                x = transform_wo_resize(x)
            else:
                x = transform_with_resize(x)
        else:
            # print('not rgb')
            x = x.convert('L')
            x = np.array(x)
            x = x.astype('float')
            
            if ( not (w == 224 and h == 224 ) ) and resize==True:
                x = cv2.resize(x, (224, 224))

            if np.max(x) > 1.0:
                x = x / 255.0
            x = torch.FloatTensor(x).unsqueeze(0)

        return x
    


if __name__ == '__main__':
    from torch.utils.data import DataLoader


