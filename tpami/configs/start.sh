# run trival ablation exp
CUDA_VISIBLE_DEVICES=4 nohup python train_trival.py --data-path ../../atten_data/bd --name bd_ab1 --val-data-path /data/bxy/MultiModelAD/data/bd_test/  --use_prior 0 --use_unc 1 --use_nonlocal 0  > logs/bd_ab1.log 2>&1 &
# run aug ablation study
CUDA_VISIBLE_DEVICES=2 nohup python train_iter_mod0s.py --data-path ../../atten_data/bd --name bd_k8_wosf_ab3 --topK 8 --mix_dir mixup_data_ab_3 --use_prior 1 --use_unc 1 --use_nonlocal 1  > logs/bd_k8_wosf_ab3.log 2>&1 &
# run trival backbone
 nohup python train_trival.py --data-path ../../atten_data/bd --name backbone_vgg --backbone vgg --val-data-path /data/bxy/MultiModelAD/data/bd_test > logs/backbone_vgg.log  2>&1 &
