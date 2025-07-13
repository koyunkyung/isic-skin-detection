import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import io
import h5py

## HDF5에서 이미지와 메타데이터 불러오는 PyTorch Dataset ##
class SkinLesionDataset(Dataset):
    def __init__(self, df, hdf5_path, transforms=None, use_metadata=True):
        self.df = df.reset_index(drop=True)
        self.hdf5_path = hdf5_path
        self.transforms = transforms
        self.use_metadata = use_metadata
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row['isic_id']
        target = torch.tensor(row['target'], dtype=torch.float32)
        
        # 이미지 로드 (HDF5 -> PIL 이미지 디코딩)
        with h5py.File(self.hdf5_path, "r") as f:
            img_bytes = f[img_id][()]
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        if self.transforms:
            img = self.transforms(img)
            
        if self.use_metadata:
            meta = row.drop(['isic_id', 'target']).values.astype(np.float32)
            meta_tensor = torch.tensor(meta, dtype=torch.float32)
            return img, meta_tensor, target
        else:
            return img, target