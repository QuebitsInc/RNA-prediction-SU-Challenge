from utils import seed_everything
import os
import gc
import torch
from dataset import RNA_Dataset
from sampler import LenMatchBatchSampler
from data_loader import DeviceDataLoader
from fastai.vision.all import *
from model import RNA_Model
from error import MAE

fname = 'example0'
PATH = './data/'
OUT = './'
bs = 256
num_workers = 2
SEED = 2023
nfolds = 4
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    seed_everything(SEED)
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_parquet(os.path.join(PATH, 'train_data.parquet'))

    for fold in [0]:
        ds_train = RNA_Dataset(df, mode='train', fold=fold, nfolds=nfolds)
        ds_train_len = RNA_Dataset(df, mode='train', fold=fold,
                                   nfolds=nfolds, mask_only=True)
        sampler_train = torch.utils.data.RandomSampler(ds_train_len)
        len_sampler_train = LenMatchBatchSampler(sampler_train, batch_size=bs,
                                                 drop_last=True)
        dl_train = DeviceDataLoader(torch.utils.data.DataLoader(ds_train,
                                                                batch_sampler=len_sampler_train, num_workers=num_workers,
                                                                persistent_workers=True), device)

        ds_val = RNA_Dataset(df, mode='eval', fold=fold, nfolds=nfolds)
        ds_val_len = RNA_Dataset(df, mode='eval', fold=fold, nfolds=nfolds,
                                 mask_only=True)
        sampler_val = torch.utils.data.SequentialSampler(ds_val_len)
        len_sampler_val = LenMatchBatchSampler(sampler_val, batch_size=bs,
                                               drop_last=False)
        dl_val = DeviceDataLoader(torch.utils.data.DataLoader(ds_val,
                                                              batch_sampler=len_sampler_val, num_workers=num_workers), device)
        gc.collect()

        data = DataLoaders(dl_train, dl_val)
        model = RNA_Model()
        model = model.to(device)
        learn = Learner(data, model, loss_func=loss, cbs=[GradientClip(3.0)],
                        metrics=[MAE()]).to_fp16()

        learn.fit_one_cycle(32, lr_max=5e-4, wd=0.05, pct_start=0.02)
        torch.save(learn.model.state_dict(),
                   os.path.join(OUT, f'{fname}_{fold}.pth'))
        gc.collect()


if __name__ == "__main__":
    main()
