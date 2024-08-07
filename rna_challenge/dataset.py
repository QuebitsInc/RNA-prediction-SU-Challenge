import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import KFold
import torch


class RNA_Dataset(Dataset):
    def __init__(self, df, mode='train', seed=2023, fold=0, nfolds=4,
                 mask_only=False, **kwargs):
        self.seq_map = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
        self.Lmax = 206
        df['L'] = df.sequence.apply(len)
        df_2A3 = df.loc[df.experiment_type == '2A3_MaP']
        df_DMS = df.loc[df.experiment_type == 'DMS_MaP']

        split = list(KFold(n_splits=nfolds, random_state=seed,
                           shuffle=True).split(df_2A3))[fold][0 if mode == 'train' else 1]
        df_2A3 = df_2A3.iloc[split].reset_index(drop=True)
        df_DMS = df_DMS.iloc[split].reset_index(drop=True)

        m = (df_2A3['SN_filter'].values > 0) & (df_DMS['SN_filter'].values > 0)
        df_2A3 = df_2A3.loc[m].reset_index(drop=True)
        df_DMS = df_DMS.loc[m].reset_index(drop=True)

        self.seq = df_2A3['sequence'].values
        self.L = df_2A3['L'].values

        self.react_2A3 = df_2A3[[c for c in df_2A3.columns if
                                 'reactivity_0' in c]].values
        self.react_DMS = df_DMS[[c for c in df_DMS.columns if
                                 'reactivity_0' in c]].values
        self.react_err_2A3 = df_2A3[[c for c in df_2A3.columns if
                                     'reactivity_error_0' in c]].values
        self.react_err_DMS = df_DMS[[c for c in df_DMS.columns if
                                     'reactivity_error_0' in c]].values
        self.sn_2A3 = df_2A3['signal_to_noise'].values
        self.sn_DMS = df_DMS['signal_to_noise'].values
        self.mask_only = mask_only

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        seq = self.seq[idx]
        if self.mask_only:
            mask = torch.zeros(self.Lmax, dtype=torch.bool)
            mask[:len(seq)] = True
            return {'mask': mask}, {'mask': mask}
        seq = [self.seq_map[s] for s in seq]
        seq = np.array(seq)
        mask = torch.zeros(self.Lmax, dtype=torch.bool)
        mask[:len(seq)] = True
        seq = np.pad(seq, (0, self.Lmax-len(seq)))

        react = torch.from_numpy(np.stack([self.react_2A3[idx],
                                           self.react_DMS[idx]], -1))
        react_err = torch.from_numpy(np.stack([self.react_err_2A3[idx],
                                               self.react_err_DMS[idx]], -1))
        sn = torch.FloatTensor([self.sn_2A3[idx], self.sn_DMS[idx]])

        return {'seq': torch.from_numpy(seq), 'mask': mask}, \
               {'react': react, 'react_err': react_err,
                'sn': sn, 'mask': mask}
