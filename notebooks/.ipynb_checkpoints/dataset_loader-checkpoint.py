from torch import Tensor, nn
import torch
import numpy as np

import os
import zarr
import s3fs
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F


with open(os.path.expanduser("~") + "/.keys") as f:
    for line in f:
        if line.startswith("export "):
            key, val = line.strip().split("=", 1)
            os.environ[key.replace("export ","")] = val
fs = s3fs.S3FileSystem(
        key=os.environ["AWS_ACCESS_KEY_ID"],
        secret=os.environ["AWS_SECRET_ACCESS_KEY"],
        client_kwargs={"endpoint_url": "http://localhost:8333"}
    )

class LDCDataset(Dataset):
    def __init__(self, name, resolution, dimT):
        
        store_X = s3fs.S3Map(root=f"ldcdataset/{name}/X", s3=fs, check=False)
        store_Y = s3fs.S3Map(root=f"ldcdataset/{name}/Y", s3=fs, check=False)
        dts_X = zarr.open_array(store_X, mode="r")
        dts_Y = zarr.open_array(store_Y, mode="r")
        x = torch.from_numpy(dts_X[:])

        x = x.permute(0, 3, 1, 2)  # B C X X
        x = F.interpolate(x, size=(resolution, resolution), mode="bilinear", align_corners=False)
        x = x.permute(0, 2, 3, 1)  # B X X C
        self.X = x
        
        y = torch.from_numpy(dts_Y[:, :dimT])

        B, T, X, _, C = y.shape

        y = y.permute(0,1,4,2,3)  # B T C X X
        y = y.reshape(B*T, C, X, X)
        
        y = F.interpolate(y, size=(65,65), mode="bilinear", align_corners=False)
        
        y = y.reshape(B, T, C, 65, 65)
        y = y.permute(0,1,3,4,2)  # B T 65 65 C

        self.Y = y
        self.length = self.X.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]
        return x, y
