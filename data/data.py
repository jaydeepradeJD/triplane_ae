import os
import numpy as np
import torch
from torch.utils.data import Dataset


class TriPlaneDataset(Dataset):
    def __init__(self, data_dir, num_points=500000, num_objs=500, single_obj=False):
        self.data_dir = data_dir
        if single_obj:
            num_objs = 500 # used first 500 files/objects to train decoder
            self.files = np.random.choice(os.listdir(data_dir)[num_objs:], 1, replace=False)[0]
        else:
            self.files = os.listdir(data_dir)[:num_objs] #[f for f in os.listdir(data_dir) if f.endswith(".npy")]
        
        self.num_points = num_points

    def __len__(self):
        return len(self.files)
    
    def normalize_points(self, points):
        min_val = np.min(points, axis=0)
        max_val = np.max(points, axis=0)
        center = (min_val + max_val) / 2.0
        scale = 2.0 / (max_val - min_val).max()
        return (points - center) / scale

    def ToTensor(self, points):
        return torch.tensor(points).float()

    def __getitem__(self, idx):
        file = self.files[idx]
        points = np.load(os.path.join(self.data_dir, file, "models", "model_normalized.npy"))
        # randomly sample num_points points
        points = points[np.random.randint(0, points.shape[0], self.num_points)]
        point_loc = points[:, :3]
        point_occ = points[:, 3]
        
        return idx, self.ToTensor(point_loc), self.ToTensor(point_occ)
    

if __name__ == "__main__":
    dataset = TriPlaneDataset("/work/mech-ai/jrrade/Tri-plane/02691156")
    print(dataset[0])