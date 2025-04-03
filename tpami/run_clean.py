'''
test the corruption robustness
'''
import os
import argparse
import torch
from dataset.DrDataset import DrDataset
from torch.utils.data import DataLoader
import utils.train_utils as utils

from utils.train_and_evaluate import evaluate
from pathlib import Path
import numpy as np
import cv2
from audtorch.metrics.functional import pearsonr

# from metann import Learner

def cal_cc(pred: torch.Tensor, gt: torch.Tensor):
    a = pearsonr(pred.flatten(), gt.flatten())
    a = torch.where(torch.isnan(a), torch.full_like(a, 0), a)
    a = a.mean()
    return a


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='uncertainty-m', help="uncertainty-r/-c, resnet, ConvNext")
    parser.add_argument('--data-path', default='./dataset', help='where to store data')
    parser.add_argument('--dataset', default='dr')
    parser.add_argument('--save_model', default='5_layer')
    parser.add_argument('--input_dim', default=1, type=int)
    parser.add_argument('--val_aucs', default=False, type=bool)
    parser.add_argument('--vis', default=False, type=bool)
    parser.add_argument('--eval_batch', default=False, type=bool)
    
    
    # parser.add_argument('--severity', default=0, type=int)
    # parser.add_argument('--cam_subdir', default='camera', type=str)


    # parser.add_argument('--output_folder', default='output', type=str)

    return parser.parse_args()


def infer(args, model, data_loader, device):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    with torch.no_grad():
        count = 0
        for images, targets in metric_logger.log_every(data_loader, 100, header):
            images = images.to(device)
            targets = targets.to(device)  # Ensure targets are on the same device
            if args.model.find('uncertainty') != -1:
                outputs = model(images)
            else:
                raise NotImplementedError
            
            for i in range(images.shape[0]):
                output = outputs[i]
                target = targets[i]
                cc = cal_cc(output, target)
                # Normalize the output to [0, 255]
                normalized_output = (output / output.max()).permute(1, 2, 0).cpu().detach().numpy() * 255
                normalized_output = normalized_output.astype(np.uint8)
                
                # Convert the normalized output to heatmap
                output_heatmap = cv2.applyColorMap(normalized_output, cv2.COLORMAP_JET)
                
                # Normalize the target to [0, 255]
                normalized_target = (target / target.max()).permute(1, 2, 0).cpu().detach().numpy() * 255
                normalized_target = normalized_target.astype(np.uint8)
                
                # Convert the normalized target to heatmap
                target_heatmap = cv2.applyColorMap(normalized_target, cv2.COLORMAP_JET)
                
                # Convert the original image to numpy array
                original_image = images[i].permute(1, 2, 0).cpu().detach().numpy()
                original_image = (original_image - original_image.min()) / (original_image.max() - original_image.min()) * 255
                original_image = original_image.astype(np.uint8)
                
                # Resize heatmaps to match the original image size
                output_heatmap = cv2.resize(output_heatmap, (original_image.shape[1], original_image.shape[0]))
                target_heatmap = cv2.resize(target_heatmap, (original_image.shape[1], original_image.shape[0]))
                
                # Blend the heatmaps with the original image
                blended_output_image = cv2.addWeighted(original_image, 0.5, output_heatmap, 0.5, 0)
                blended_target_image = cv2.addWeighted(original_image, 0.5, target_heatmap, 0.5, 0)
                
                # Concatenate the blended images horizontally
                concatenated_blended_image = cv2.hconcat([blended_target_image, blended_output_image])
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.5
                color = (0, 0, 255)  # Red color in BGR
                thickness = 2
                text = "cc: {:.2f}".format(cc)
                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                text_x = concatenated_blended_image.shape[1] - text_size[0] - 10
                text_y = text_size[1] + 10
                cv2.putText(concatenated_blended_image, text, (text_x, text_y), font, font_scale, color, thickness)
                
                # Save the concatenated blended image
                output_dir = Path('output')/args.save_model
                output_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(output_dir/"{}.png".format(count), concatenated_blended_image)
                count += 1

def evaluate_batch(args, model, data_loader, device):
    import pdb; pdb.set_trace()
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
                output = model(images)
            else:
                output, _ = model(images)
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


def main(args):
#    cors  = [
#        'brightness',
#        'defocus_blur', 'fog' ,'gaussian_blur', 'impulse_noise','motion_blur','snow',   
#        'contrast', 'frost' , 'gaussian_noise' , 'glass_blur' , 'jpeg_compression' , 'pixelate' ,
#        'shot_noise' ,
#        'zoom_blur'
#    ]
    results = {}
    # cors = ['gaussian_noise', 'motion_blur', 'jpeg_compression', 'fog', 'snow']
    # cors = [ 'fog', 'snow']

    # for cor in cors:
    # if args.dataset == 'dr':
    #     dataset = DrDataset(args.data_path, cam_subdir='camera', mode='test')
    # elif args.dataset == 'bd':
    dataset = DrDataset(args.data_path, mode='test', cam_subdir='camera', gaze_subdir='gaze')
    # else: raise NotImplementedError

    print(len(dataset))
    if args.eval_batch:
        batch_size = 32
    else:
        batch_size = 1
    data_loader = DataLoader(dataset,
                            batch_size=batch_size,  # must be 1
                            num_workers=8,
                            pin_memory=True)



    if args.model == 'uncertainty-m':
        from models.model import Model
        model = Model('mobileViT', input_dim=args.input_dim).cuda()
    else : raise NotImplementedError



    checkpoint = torch.load('./save_weights/model_best_{}.pth'.format(args.save_model))
    model.load_state_dict(checkpoint['model'])
    if args.vis:
        infer(args, model, data_loader, device='cuda')
    else:
        if args.eval_batch:
            kld_metric, cc_metric = evaluate_batch(args, model, data_loader, device='cuda')
        else:
            kld_metric, cc_metric = evaluate(args, model, data_loader, device='cuda')
        kld_info, cc_info = kld_metric.compute(), cc_metric.compute()
        print(f"val_kld: {kld_info:.3f} val_cc: {cc_info:.3f}")
    # if args.val_aucs:
    #     kld_metric, cc_metric, aucs_metric = evaluate(args, model, data_loader, device='cuda')
    #     kld_info, cc_info, aucs_info = kld_metric.compute(), cc_metric.compute(), aucs_metric.compute()

    #     print(f"val_aucs: {aucs_info:.3f} val_kld: {kld_info:.3f} val_cc: {cc_info:.3f}")
    # else:


    # print("\nFinal Results:")
    # for cor, metrics in results.items():
    #     print(f"{cor}: {metrics}")


if __name__ == '__main__':
    args = parse_args()
    main(args)
