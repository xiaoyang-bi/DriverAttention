from dataset.MixDataset import MixDataset

if __name__ == '__main__':
    dataset = MixDataset('../../atten_data/bd', mode='train')
    kl_db = dataset.get_data_by_kl(0, 0)