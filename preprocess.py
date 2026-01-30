import json
import numpy as np
import pandas as pd
import os, torch
from glob import glob

norm = {  'mass':        {'mean': 12.0, 'std': 8.0},
        'sersicIndex': {'mean': 4.0, 'std': 1.0},}


def read_data(path):
    all_txt = glob(os.path.join(path, "*.txt"))
    coor_file = [f for f in all_txt if os.path.basename(f).startswith("voronoi_")]
    kine_file = [f for f in all_txt if os.path.basename(f).startswith("kinem_")]
    coor_file.sort()
    kine_file.sort()
    coor_file, kine_file = coor_file[0], kine_file[0]
    coordinate = pd.read_csv(coor_file, sep=r"\s+", header=None, names=["x", "y", "id"])
    with open(kine_file) as f:
        header = f.readline().lstrip("#").split()
    kinematics = pd.read_csv(kine_file, sep=r"\s+", header=None, names=header, comment="#",)
    kinematics["id"] = np.unique(coordinate["id"]) #pd.unique(coordinate["id"]) 
    data = coordinate.join(kinematics.set_index("id"), on="id")
    v_matrix = data.pivot(index="y", columns="x", values="v").values
    sigma_matrix = data.pivot(index="y", columns="x", values="sigma").values
    return data, v_matrix, sigma_matrix


# Normalization of data
def normalization(maps, paras, norm_paras=norm):
    # Normalization
    map_bias = torch.mean(maps, dim=(-2,-1), keepdim=True)
    map_sigma = torch.std(maps, dim=(-2,-1), keepdim=True)
    norm_maps = (maps - map_bias) / (map_sigma + 1e-8) # avoid division by zero

    bias = torch.tensor([norm_paras['mass']['mean'], norm_paras['sersicIndex']['mean']])
    std= torch.tensor([norm_paras['mass']['std'], norm_paras['sersicIndex']['std']])
    norm_paras = (paras - bias)/std # avoid division by zero
    return norm_maps, norm_paras

def denormalization(pred_paras, norm_paras=norm):
    bias = np.array([norm_paras['mass']['mean'], norm_paras['sersicIndex']['mean']])
    std= np.array([norm_paras['mass']['std'], norm_paras['sersicIndex']['std']])
    pred_paras = pred_paras*std + bias
    return pred_paras




