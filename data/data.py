import os
import numpy as np
import torch
from torch.utils.data import Dataset


class TriPlaneDataset(Dataset):
    def __init__(self, data_dir, num_points):
        self.data_dir = data_dir
        self.files = [f for f in os.listdir(data_dir) if f.endswith(".npy")]
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
        points = np.load(os.path.join(self.data_dir, file))
        # randomly sample num_points points
        points = points[np.random.randint(0, points.shape[0], self.num_points)]
        point_loc = points[:, :3]
        point_occ = points[:, 3]
        
        return self.ToTensor(point_loc), self.ToTensor(point_occ)
    

if __name__ == "__main__":
    dataset = TriPlaneDataset("/work/mech-ai/jrrade/Tri-plane/aeroplane_subset")
    print(dataset[0])