import os.path
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models.detection.mask_rcnn import maskrcnn_resnet50_fpn
from torch.utils.data import Dataset
from PIL import Image
import cv2
from pathlib import Path

class RadiateDataset(Dataset):
    def __init__(self, root: str, mode: str, alpha: float = -1., test_name: str = ''):
        self.alpha:float = alpha
        self.mode:str = mode

        # only when train and alpha !=-1, then use mask
        self.use_msk:bool = alpha !=-1 and mode == 'train'

        #file_scene_list store the ith [png name, scene name],
        #  which match the diretory structure
        self.file_scene_list:list = []
    
        self.stereo_left_folder:str = 'stereo_undistorted/left'
        self.gaze_left_folder:str = 'left_gaze'
        self.mask_left_folder:str = 'left_msk'
        self.p_dic:list = ['left_sam_p', 'left_unisal_p']
        self.out_folder:str = 'infer_gaze'

        assert os.path.exists(root), f"path '{root}' does not exists."
        self.root = Path(root)

        if mode == 'train':
            scene_list_path = self.root/'train.txt'
        elif mode == 'val':
            scene_list_path = self.root/'val.txt'
        elif mode == 'test':
            scene_list_path = self.root/'test.txt'
        else:
            raise(NotImplementedError)
        
        self.scenes = [self.root/folder.strip()
                        for folder in open(scene_list_path) if not folder.strip().startswith("#")]
        # print(self.scenes)
        
        #png in radiate
        for scene in self.scenes:
            file_list = sorted(
                list((scene/self.stereo_left_folder).glob('*.png')))
            file_scene_list = [[x.name, scene.name] for x in file_list]
            self.file_scene_list = self.file_scene_list + file_scene_list
        # print(self.file_scene_list)
        # print(len(self.file_scene_list))
            
        
    
    def __getitem__(self, i):
        file, scene = self.file_scene_list[i]
        img_path = str(self.root / scene / self.stereo_left_folder / file)
        # print(img_path)

        img = self.convert(img_path, True)
        if self.mode == 'train':

            p = []
            for p_type in self.p_dic:
                pesudo_path = str(self.root / scene / p_type / file)
                # print(pesudo_path)
                ps = self.convert(pesudo_path)

                if self.use_msk:
                    # ms = []
                    # for j, mask in enumerate(self.mask_path):
                    #     m = self.convert(os.path.join(self.mask_root, self.mask_dic[j], mask[i]))
                    #     ms.append(m)
                    #     # ps = torch.cat([ps, m], dim=0)
                    mask_path = str(self.root / scene / self.mask_left_folder / file)
                    # print(mask_path)
                    mall = self.convert(mask_path)
                    # ms.append(m)
                    # mall = sum(ms)
                    ps = ps * (mall + self.alpha)
                    ps /= ps.max()


                    # ps_save_path = str(self.root) + '/' + p_type + '_' + str(file)
                    # print(ps_save_path)
                    # print(type(ps))
                    # img_np = ps.numpy()
                    # # 2. Transpose the array if necessary (C, H, W) --> (H, W, C)
                    # if img_np.shape[0] == 3 or img_np.shape[0] == 1:  # 3-channel or 1-channel image
                    #     img_np = np.transpose(img_np, (1, 2, 0))

                    # # 3. Ensure the array has the right datatype and range [0, 255]
                    # img_np = (img_np * 255).astype(np.uint8)
                    # cv2.imwrite(ps_save_path, img_np)

                p.append(ps)

            return img, p
        elif self.mode == 'test':
            output_folder = str(self.root / scene / self.out_folder)
            os.makedirs(output_folder, exist_ok=True)
            out_test_path = str(self.root / scene / self.out_folder / file)
            # print(out_test_path)
            return img, out_test_path
        else: 
            raise(NotImplementedError)
        # gaze = self.convert(self.gaze_path[i])

        # return img, gaze

    def __len__(self):
        return len(self.file_scene_list)

    @staticmethod
    def convert(x_path, is_rgb=False):
        transform = transforms.Compose([transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
        x = Image.open(x_path)
        if is_rgb:
            x = x.convert('RGB')
            x = transform(x)
        else:
            x = x.convert('L')
            x = np.array(x)
            x = x.astype('float')
            x = cv2.resize(x, (224, 224))
            if np.max(x) > 1.0:
                x = x / 255.0
            x = torch.FloatTensor(x).unsqueeze(0)

        return x

#     def collate_fn(self, batch):
#         if self.train:
#             images, targets, p = list(zip(*batch))
#         else:
#             images, targets = list(zip(*batch))
#         batched_imgs = cat_list(images, fill_value=0)
#         batched_targets = cat_list(targets, fill_value=0)
#         if self.train:
#             batched_p = []
#             for index in range(len(p[0])):
#                 temp = [ps[index] for ps in p]
#                 batched_p.append(cat_list(temp, fill_value=0))

#             return batched_imgs, batched_targets, batched_p
#         else:
#             return batched_imgs, batched_targets


# def cat_list(images, fill_value=0):
#     max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
#     batch_shape = (len(images),) + max_size
#     batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
#     for img, pad_img in zip(images, batched_imgs):
#         pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
#     return batched_imgs









if __name__ == '__main__':
    train_dataset = RadiateDataset("/data/bxy/MultiModelAD/data/radiate", mode='train', alpha=0.3)
    # train_dataset.__getitem__(11)
    print(len(train_dataset))

    test_dataset = RadiateDataset("/data/bxy/MultiModelAD/data/radiate", mode='test')
    print(len(test_dataset))
    img = test_dataset[0]

    # i, t = train_dataset[0]
