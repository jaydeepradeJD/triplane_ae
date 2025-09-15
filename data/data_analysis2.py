import os
import numpy as np
import json
import trimesh

obj_path = "/work/mech-ai/jrrade/Tri-plane/8bc53bae41a8105f5c7506815f553527/models/model_normalized.obj"
mesh = trimesh.load(obj_path, force='mesh', skip_materials=True)

# get the bounding box of the mesh
print('mesh bounds: ', mesh.bounds)
print('mesh centroid: ', mesh.vertices.mean(axis=0))
# npy_path = "/work/mech-ai/jrrade/Tri-plane/aeroplane_subset/4a7b3bb0f7e4e13af7f031a34b185310.npy"

# data = np.load(npy_path)

# print(data.shape)
# occ = data[:, 3]
# print(np.unique(occ, return_counts=True))

# occ_0_points = data[occ == 0]
# occ1_points = data[occ == 1]
# print(occ_0_points.shape)
# print(occ1_points.shape)
# # save the points with occupancy 1 as ply file
# os.makedirs("/work/mech-ai/jrrade/Tri-plane/aeroplane_subset/plys", exist_ok=True)
# occ_0_ply_path = "/work/mech-ai/jrrade/Tri-plane/aeroplane_subset/plys/4a7b3bb0f7e4e13af7f031a34b185310_0.ply"
# occ_1_ply_path = "/work/mech-ai/jrrade/Tri-plane/aeroplane_subset/plys/4a7b3bb0f7e4e13af7f031a34b185310_1.ply"
# trimesh.PointCloud(occ1_points[:, :3]).export(occ_1_ply_path)
# trimesh.PointCloud(occ_0_points[:, :3]).export(occ_0_ply_path)
