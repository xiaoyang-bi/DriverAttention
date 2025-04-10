import torch
from models.swinv2_regression import SwinV2Regression
if __name__ == '__main__':
    import pdb; pdb.set_trace()
    model = SwinV2Regression()
    model.cuda()
    data = torch.zeros(4, 3,  256, 256).cuda()
    feat_list, out = model(data)
    print('done') 
