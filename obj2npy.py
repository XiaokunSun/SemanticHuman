import numpy as np
import pytorch3d.io as IO
import os
import glob
import torch
import argparse
from configure.cfgs import cfg
from tqdm import tqdm
# example: python obj2npy.py --save_path /home/sxk/16T/3D_human_representation/DFAUST_male --trainobj_path /home/sxk/16T/data/DFAUST/DFAUST_male/train --testobj_path /home/sxk/16T/data/DFAUST/DFAUST_male/test  --train_start 0 --train_end 980 --test_start 0 --test_end 86
# .obj data ---> .npy data

parser = argparse.ArgumentParser(description='Arguments for dataset split')
parser.add_argument('--save_path', type=str,
            help='Root data directory location, should be same as in neural3dmm.ipynb')
parser.add_argument('--trainobj_path', type=str, 
            help='Dataset name, Default is DFAUST')
parser.add_argument('--testobj_path', type=str, 
            help='Dataset name, Default is DFAUST')
parser.add_argument('--train_start', type=int, 
            help='Number of meshes in validation set, default 100')
parser.add_argument('--train_end', type=int, 
            help='Number of meshes in validation set, default 100')
parser.add_argument('--test_start', type=int, 
            help='Number of meshes in validation set, default 100')
parser.add_argument('--test_end', type=int, 
            help='Number of meshes in validation set, default 100')
args = parser.parse_args()

save_path = args.save_path
os.makedirs(os.path.join(save_path, 'preprocessed'), exist_ok=True)
os.makedirs(os.path.join(save_path, 'template'), exist_ok=True)
train_start = args.train_start
train_end = args.train_end
test_start = args.test_start
test_end = args.test_end
trainobj_path = args.trainobj_path
testobj_path = args.testobj_path
trainobj_list = sorted(os.listdir(trainobj_path))
testobj_list = sorted(os.listdir(testobj_path))
for i in tqdm(range(train_start, train_end)):
    v, f, _ = IO.load_obj(os.path.join(trainobj_path, trainobj_list[i]))
    if i == train_start:
        total_v = v[None]
    else:
        total_v = torch.cat((total_v, v[None]), dim = 0)
    # print(trainobj_list[i])
np.save('{}/preprocessed/train.npy'.format(save_path), total_v)
print(total_v.shape)
for i in tqdm(range(test_start, test_end)):
    v, f, _ = IO.load_obj(os.path.join(testobj_path, testobj_list[i]))
    if i == test_start:
        total_v = v[None]
    else:
        total_v = torch.cat((total_v, v[None]), dim = 0)
    # print(testobj_list[i])
np.save('{}/preprocessed/test.npy'.format(save_path), total_v)
print(total_v.shape)
os.system('cp {} {}'.format(os.path.join(trainobj_path, trainobj_list[0]), os.path.join(save_path, 'template', 'template.obj')))
print('cp {} {}'.format(os.path.join(trainobj_path, trainobj_list[0]), os.path.join(save_path, 'template', 'template.obj')))

def cal_girth(verts, factor_list, edge_point_index_list):
    girth_cp = []
    for i in range(len(factor_list)):
        girth_point_cp = verts[edge_point_index_list[i][:, 0], :] * (1-factor_list[i]) + verts[edge_point_index_list[i][:, 1], :] * factor_list[i]
        girth_part = np.sqrt(np.sum((girth_point_cp[0] - girth_point_cp[-1]) ** 2))
        for ii in range(girth_point_cp.shape[0] - 1):
            girth_part = girth_part + np.sqrt(np.sum((girth_point_cp[ii] - girth_point_cp[ii+1]) ** 2))
        girth_cp.append(girth_part)
    return np.array(girth_cp)

def cal_length(verts, J_regressor, skl_list):
    kps = np.matmul(J_regressor, verts)
    length = np.zeros(len(skl_list))
    for index in range(len(skl_list)):
        if len(skl_list[index]) == 2:
            length[index] = np.sqrt(np.sum((kps[skl_list[index][0], :] -  kps[skl_list[index][1], :]) ** 2))
        elif len(skl_list[index]) == 3:
            length[index] = np.sqrt(np.sum((kps[skl_list[index][0], :] - (kps[skl_list[index][1], :] + kps[skl_list[index][2], :]) / 2) ** 2))
    return length

# Calculate the measurement parameters of human body data

factor_list = np.load(cfg.PATH.factor_list, allow_pickle=True)
edge_point_index_list = np.load(cfg.PATH.edge_point_index_list, allow_pickle=True)
J_regressor = np.load(cfg.PATH.J_regressor, allow_pickle=True)
vert_part_index_dict = np.load(cfg.PATH.vert_part_index_dict, allow_pickle=True).item()
skl_list = cfg.CONSTANTS.skl_list

# The measurement parameters of training data

obj_dir = args.trainobj_path
obj_list = sorted(os.listdir(obj_dir))[train_start:train_end]
measure = np.zeros((len(obj_list), 32))
for k, obj_name in tqdm(enumerate(obj_list)):
    verts, _, _ = IO.load_obj(os.path.join(obj_dir, obj_name))
    girth = cal_girth(verts.numpy(), factor_list, edge_point_index_list)
    length = cal_length(verts.numpy(), J_regressor, skl_list[1:])
    # print(k)
    measure[k,:] = np.concatenate((girth, length))
np.save(os.path.join(obj_dir, '../train_measurements.npy'), measure)
print(measure.shape)
# The measurement parameters of testing data

obj_dir = args.testobj_path
obj_list = sorted(os.listdir(obj_dir))[test_start:test_end]
measure = np.zeros((len(obj_list), 32))
for k, obj_name in tqdm(enumerate(obj_list)):
    verts, _, _ = IO.load_obj(os.path.join(obj_dir, obj_name))
    girth = cal_girth(verts.numpy(), factor_list, edge_point_index_list)
    length = cal_length(verts.numpy(), J_regressor, skl_list[1:])
    # print(k)
    measure[k,:] = np.concatenate((girth, length))
np.save(os.path.join(obj_dir, '../test_measurements.npy'), measure)
print(measure.shape)