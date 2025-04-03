import os.path
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import cv2
from pathlib import Path
import random


class DrDataset(Dataset):
    def __init__(self, root: str, mode: str,
                  alpha: float = -1.,
                  severity:str = None,
                  cam_subdir = 'camera_224_224',
                  out_folder = 'infer_gaze',
                  gaze_subdir = 'gaze_224_224',
                  noise_type:str = None):
        '''
        mode should include:
        train, val, test, infer, run_example
        '''
        self.alpha:float = alpha
        self.mode:str = mode
        self.use_msk:bool = True

        self.file_scene_list:list = []
        self.out_folder = out_folder
        # self.p_dic:list = ['ml_p_224_224', 'unisal_p_224_224']
        self.p_dic:list = ['ml_p', 'unisal_p']
        self.cam_subdir = cam_subdir
        # self.mask_subdir = 'masks_224_224'
        self.mask_subdir = 'masks'
        self.gaze_subdir = gaze_subdir
        self.severity = severity
        self.noise_type = noise_type

        assert os.path.exists(root), f"path '{root}' does not exists."
        self.root = Path(root)


        if self.noise_type is  None:
            scene_list_path = self.root/(mode + '.txt')
            self.scenes = [self.root/folder.strip()
                        for folder in open(scene_list_path) if not folder.strip().startswith("#")]
            for scene in self.scenes:
                file_list = sorted(
                    list((scene/self.cam_subdir).glob('*')))
                file_scene_list = [[x.name, scene.name] for x in file_list if x.suffix in ('.png', '.jpg', '.jpeg')]
                self.file_scene_list = self.file_scene_list + file_scene_list
        else:
            self.noise_type = self.noise_type + "_224_224"
            scene_list_path = self.root/(mode + '_cor.txt')
            self.out_folder += ('_' + self.noise_type)
            print(self.out_folder)
            self.scenes = [self.root/folder.strip()
                        for folder in open(scene_list_path) if not folder.strip().startswith("#")]
            print("Noise Loader")
            # print(self.scenes)
            for scene in self.scenes:
                file_list = sorted(
                    list((scene/self.noise_type).glob('*')))
                file_scene_list = [[x.name, scene.name] for x in file_list if x.suffix in ('.png', '.jpg', '.jpeg')]
                # print(file_scene_list)
                self.file_scene_list = self.file_scene_list + file_scene_list
            
        
    
    def __getitem__(self, i):
        file, scene = self.file_scene_list[i]
        if self.noise_type:
            img_path = str(self.root / scene / self.noise_type / file)
        else:
            img_path = str(self.root / scene / self.cam_subdir / file)
        # print(img_path)

        img = self.convert(img_path, True)
        if self.mode == 'train':

            p = []
            for p_type in self.p_dic:
                pesudo_path = str(self.root / scene / p_type / file)
                # print(pesudo_path)
                ps = self.convert(pesudo_path)

                if self.use_msk:
                    mask_path = str(self.root / scene / self.mask_subdir / file)
                    mall = self.convert(mask_path)
                    ps = ps * (mall + self.alpha)
                    ps /= ps.max()

                p.append(ps)

            return img, p
        elif self.mode == 'val' or self.mode == 'test':
            gaze_path = str(self.root / scene / self.gaze_subdir / file)
            extensions = ['.png', '.jpg', '.jpeg']

            # Iterate through the extensions and check if the file exists
            for ext in extensions:
                gaze_path = str(self.root / scene / self.gaze_subdir / file).rsplit('.', 1)[0] + ext
                if os.path.exists(gaze_path):
                    break
            gaze  = self.convert(gaze_path)
            return img, gaze
        elif self.mode == 'infer':
            output_folder = str(self.root / scene / self.out_folder)
            os.makedirs(output_folder, exist_ok=True)
            out_test_path = str(self.root / scene / self.out_folder / file)

            return img, out_test_path
        elif self.mode == 'train_mix':
            #return image, p, attention
            p = []
            for p_type in self.p_dic:
                pesudo_path = str(self.root / scene / p_type / file)
                # print(pesudo_path)
                ps = self.convert(pesudo_path)
                if self.use_msk:
                    mask_path = str(self.root / scene / self.mask_subdir / file)
                    mall = self.convert(mask_path)
                    ps = ps * (mall + self.alpha)
                    ps /= ps.max()
                p.append(ps)
                
            infer_gaze_path = str(self.root / scene / 'infer_gaze' / file)
            infer_gaze  = self.convert(infer_gaze_path)
            return img, p, infer_gaze
        elif self.mode == 'mix':
            output_folder = str(self.root / scene / self.out_folder)
            os.makedirs(output_folder, exist_ok=True)
            out_test_path = str(self.root / scene / self.out_folder / file)

            infer_gaze_path = str(self.root / scene / 'infer_gaze' / file)
            infer_gaze  = self.convert(infer_gaze_path)
            return img, infer_gaze, out_test_path

        else: 
            raise(NotImplementedError)

    def __len__(self):
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
            x = x.astype('float')
            
            if ( w != 224 or h != 224):
                x = cv2.resize(x, (224, 224))

            if np.max(x) > 1.0:
                x = x / 255.0
            x = torch.FloatTensor(x).unsqueeze(0)

        return x


if __name__ == '__main__':
    # from torch.utils.data import DataLoader
    # train_target = DrDataset("/root/autodl-tmp/atten_dataset/dr", mode='train_target', target_domain = 'rain', alpha=0.3)
    # train_target[0]
    # print(len(train_target))
    

    # train_meta = DrDataset("/root/autodl-tmp/atten_dataset/dr", mode='train_meta', meta_domain = ['night'], sample_num_per_domain = 2048, alpha=0.3)
    # print(len(train_meta))
    # train_meta[0]


    # meta_loader = DataLoader(train_meta,
    #                          batch_size=2048,  # must be 1
    #                          num_workers=8,
    #                          pin_memory=True)
    # print(len(meta_loader))

    # val_dataset = DrDataset("/root/autodl-tmp/atten_dataset/dr", mode='val', alpha=0.3)
    # print(len(val_dataset))
    # val_dataset[0]



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
    dataset = DrDataset("../../atten_data/bd", mode='train', alpha=0.3)
    print(len(dataset))
    import pdb; pdb.set_trace()
    output_dir = Path('output')/'test_pf'
    output_dir.mkdir(parents=True, exist_ok=True)
    for i in range(100000):
        img, ps = dataset[i]
        img = img.cpu().numpy()
        for p in ps:
            p = p.cpu().numpy()
            heatmap = gen_heatmap(img, p)
            
            cv2.imwrite(output_dir/"{}.jpg".format(i), heatmap.transpose(1, 2, 0))