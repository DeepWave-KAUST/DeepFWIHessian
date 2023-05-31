
'''  prepare data for pytorch '''

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader




def prepare_data(x, d, y, patch_size, slide, batch_size):
    
    pd = (1, 0, 0, 0)
    pad_replicate = nn.ReplicationPad2d(pd)
    assert  x.shape == y.shape , 'shape should be equal, ' 
    
    x = pad_replicate(x.unsqueeze(0)).squeeze(0).float().cuda()
    d = pad_replicate(d.unsqueeze(0)).squeeze(0).float().cuda()
    y = pad_replicate(y.unsqueeze(0)).squeeze(0).float().cuda()

    k = patch_size
    kk = slide

    X, D, L, Y = [], [], [], []
    for xi in range(int(x.shape[0] // (k/kk))):
        for yi in range(int(x.shape[1] // (k/kk))):
            patch1 = x[xi*(k//kk):xi*(k//kk)+k, yi*(k//kk):yi*(k//kk)+k]
            patchd = d[xi*(k//kk):xi*(k//kk)+k, yi*(k//kk):yi*(k//kk)+k]
            patch2 = y[xi*(k//kk):xi*(k//kk)+k, yi*(k//kk):yi*(k//kk)+k]

            if patch1.shape == (k, k):    
                X.append(patch1)
                D.append(patchd)
                Y.append(patch2)

    X = torch.stack(X)
    D = torch.stack(D)
    Y = torch.stack(Y)
    X = torch.cat([X.unsqueeze(1), D.unsqueeze(1)], dim=1)
    
    dm_dataset = TensorDataset(X, Y)
    train_dataloader = DataLoader(dm_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    return train_dataloader
