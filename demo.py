
import numpy as np
import os
import copy
from configure.cfgs import cfg, update_cfg
import torch
import pytorch3d.io as IO
from glob import glob
from utils_SH import *
from autoencoder_dataset import autoencoder_dataset
from torch.utils.data import DataLoader
from test_funcs import test_autoencoder_dataloader_nonormal

model = init_model(os.path.join(cfg.PATH.root_dir, 'configure', 'testcfg.yaml'))
data = os.path.join(cfg.PATH.root_dir, cfg.TRAIN.dataset, 'preprocessed')
J_regressor = np.load(cfg.PATH.J_regressor, allow_pickle=True)

dataset_test = autoencoder_dataset(root_dir = data, points_dataset = 'test',
                                          shapedata = None,
                                          normalization = cfg.TRAIN.normal_flag, dummy_node = True, J_regressor=J_regressor)

dataloader_test = DataLoader(dataset_test, batch_size=cfg.TRAIN.batchsize_test,\
                                     shuffle = False, num_workers = cfg.TRAIN.num_workers)
predictions, z_s, z_kps_s, tx_s, norm_l1_loss, l2_loss = test_autoencoder_dataloader_nonormal(cfg.TRAIN.device_idx, model, dataloader_test, 
        None, J_regressor, mm_constant = 1000, unnormal_flag = False) 

print(norm_l1_loss, l2_loss)


device = 'cuda:' + str(cfg.TRAIN.device_idx) 
shape_idx = 175
skl_idx = 153
style_idx = 24
obj_path = os.path.join(cfg.PATH.root_dir, cfg.TRAIN.dataset, 'template', 'template.obj')
_, f, _ = IO.load_obj(obj_path)       
save_path = os.path.join('output', 'fig1', cfg.MODEL.model_name, 'part_shape_idx_{}'.format(shape_idx))
os.makedirs(save_path, exist_ok=True)
kps_keep = list(range(len(cfg.CONSTANTS.newskl_list) + 4))
if cfg.TRAIN.kpskeep_flag:
    for i in [3,13,14]:
        kps_keep.remove(i)
skl_keep = list(range(len(cfg.CONSTANTS.newskl_list)))
if cfg.TRAIN.sklkeep_flag:
    skl_keep = [0,1,2,3,4,6,7,8,13,14,15,16,17]
    newskl_keep = list(range(len(cfg.CONSTANTS.newskl_list)))
    for i in [5,9,10]:
        newskl_keep.remove(i)
choosen_skl = [[16,18],[18,20],[20,22],[20,24],[20,26],[2,5],[5,8],[8,11],[8,32],[8,34]]
choosen_skl_index = []
for i in choosen_skl:
    choosen_skl_index.append(cfg.CONSTANTS.newskl_list.index(i))
J_regressor_torch = torch.from_numpy(J_regressor.astype(np.float32)).to(device)
choosen_part_index_in_allpart = []
for i in ['chest','abdomen','hip']:
    choosen_part_index_in_allpart.append(cfg.CONSTANTS.part_list.index(i))
pre_dirpath = os.path.join(cfg.PATH.root_dir, cfg.TRAIN.dataset, 'results', 'multiz+partkps', cfg.MODEL.model_name, 'predictions')
tx_s = torch.from_numpy(np.load(os.path.join(pre_dirpath, 'tx_s.npy'), allow_pickle=True)).to(device).float()
predictions = torch.from_numpy(np.load(os.path.join(pre_dirpath, 'predictions.npy'), allow_pickle=True)).to(device).float()
z_kps_s = torch.from_numpy(np.load(os.path.join(pre_dirpath, 'z_kps_s.npy'), allow_pickle=True)).to(device).float()
z_s = torch.from_numpy(np.load(os.path.join(pre_dirpath, 'z_s.npy'), allow_pickle=True)).to(device).float()



kps_s = torch.matmul(J_regressor_torch, tx_s[:, :-1, :])
skl_s = kps2skl(kps_s, 'ori_m')

newori_skl = copy.deepcopy(skl_s[shape_idx:shape_idx+1, ...])
newlength_skl = copy.deepcopy(skl_s[shape_idx:shape_idx+1, ...])
target_skl = copy.deepcopy(skl_s[skl_idx:skl_idx+1, ...])
newgirth_z = copy.deepcopy(z_s[shape_idx:shape_idx+1, ...])
newstyle_z = copy.deepcopy(z_s[shape_idx:shape_idx+1, ...])
target_z = copy.deepcopy(z_s[style_idx:style_idx+1, ...])

dummy = torch.zeros((1, 1, cfg.MODEL.filter_sizes_enc[0][-1])).to(device)
with torch.no_grad():
    # Edit bone orientation
    for skl_index in choosen_skl_index:  
            newori_skl[:, skl_index, :3] = target_skl[:, skl_index, :3] 
            newori_kps = skl2kps(newori_skl, 'ori_m')
    # Edit bone length
    for skl_index in skl_keep:
        if skl_index in [4,7,15,17]:
            newlength_skl[:, skl_index, 3] = newlength_skl[:, skl_index, 3] * 1.2
        newlength_kps = skl2kps(newlength_skl, 'ori_m')
    # Edit shape size
    newgirth_z[:, choosen_part_index_in_allpart, :] = newgirth_z[:, choosen_part_index_in_allpart, :]  * 1.2
    # Edit shape style
    for part_index in choosen_part_index_in_allpart:
        ori_norm = torch.sqrt(torch.sum(newstyle_z[0, part_index, :] ** 2))
        ori_direct = newstyle_z[0, part_index, :] / ori_norm
        style_norm =torch.sqrt(torch.sum(target_z[0, part_index, :] ** 2))
        style_direct = target_z[0, part_index, :] / style_norm
        newstyle_z[0, part_index, :] = ori_norm * style_direct
    

    rec_editpose = model.decode(z_s[shape_idx:shape_idx+1, ...], model.kps_encode(newori_kps), dummy)
    rec_editlength = model.decode(z_s[shape_idx:shape_idx+1, ...], model.kps_encode(newlength_kps), dummy)
    rec_editgirth = model.decode(newgirth_z, z_kps_s[shape_idx:shape_idx+1, ...], dummy)
    rec_editstyle = model.decode(newstyle_z, z_kps_s[shape_idx:shape_idx+1, ...], dummy)
    
    rec_shape = model.decode(z_s[shape_idx:shape_idx+1, ...], z_kps_s[shape_idx:shape_idx+1, ...], dummy)
    rec_skl = model.decode(z_s[skl_idx:skl_idx+1, ...], z_kps_s[skl_idx:skl_idx+1, ...], dummy)
    rec_style = model.decode(z_s[style_idx:style_idx+1, ...], z_kps_s[style_idx:style_idx+1, ...], dummy)

    save_obj(os.path.join(save_path, 'rec_editpose.obj'), rec_editpose[0,0:-1,:], f.verts_idx)
    save_obj(os.path.join(save_path, 'rec_editlength.obj'), rec_editlength[0,0:-1,:], f.verts_idx)
    save_obj(os.path.join(save_path, 'rec_editgirth.obj'), rec_editgirth[0,0:-1,:], f.verts_idx)
    save_obj(os.path.join(save_path, 'rec_editstyle.obj'), rec_editstyle[0,0:-1,:], f.verts_idx)
    save_obj(os.path.join(save_path, 'rec_shape.obj'), rec_shape[0,0:-1,:], f.verts_idx)
    save_obj(os.path.join(save_path, 'rec_skl.obj'), rec_skl[0,0:-1,:], f.verts_idx)
    save_obj(os.path.join(save_path, 'rec_style.obj'), rec_style[0,0:-1,:], f.verts_idx)
    save_obj(os.path.join(save_path, 'ori_shape.obj'), tx_s[shape_idx,0:-1,:], f.verts_idx)
    save_obj(os.path.join(save_path, 'ori_skl.obj'), tx_s[skl_idx,0:-1,:], f.verts_idx)
    save_obj(os.path.join(save_path, 'ori_style.obj'), tx_s[style_idx,0:-1,:], f.verts_idx)