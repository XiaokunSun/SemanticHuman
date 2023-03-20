import os
import torch
from tqdm import tqdm
import numpy as np
from utils_distance import calc_euclidean_dist_matrix
import random
import copy
import torch.nn.functional as F
from configure.cfgs import cfg
from utils_SH import *

def init_regul(source_vertices, source_faces):
    sommet_A_source = source_vertices[source_faces[:, 0]]
    sommet_B_source = source_vertices[source_faces[:, 1]]
    sommet_C_source = source_vertices[source_faces[:, 2]]
    target = []
    target.append(np.sqrt( np.sum((sommet_A_source - sommet_B_source) ** 2, axis=1)))
    target.append(np.sqrt( np.sum((sommet_B_source - sommet_C_source) ** 2, axis=1)))
    target.append(np.sqrt( np.sum((sommet_A_source - sommet_C_source) ** 2, axis=1)))
    return target

def get_target(vertice, face, size, device):
    target = init_regul(vertice,face)
    target = np.array(target)
    target = torch.from_numpy(target).float().to(device)
    target = target+0.00001
    target = target.unsqueeze(1).expand(3,size,-1)
    return target

def compute_score(points, faces, target):
    score = 0
    sommet_A = points[:,faces[:, 0]]
    sommet_B = points[:,faces[:, 1]]
    sommet_C = points[:,faces[:, 2]]

    score = torch.abs(torch.sqrt(torch.sum((sommet_A - sommet_B) ** 2, dim=2)) / target[0] -1)
    score = score + torch.abs(torch.sqrt(torch.sum((sommet_B - sommet_C) ** 2, dim=2)) / target[1] -1)
    score = score + torch.abs(torch.sqrt(torch.sum((sommet_A - sommet_C) ** 2, dim=2)) / target[2] -1)
    return torch.mean(score)

def Edge_loss(input, rec, edge_verts_index):
    input_edge = torch.sqrt(torch.sum((input[:, edge_verts_index[:, 0], :] - input[:, edge_verts_index[:, 1], :]) ** 2, dim = 2))
    rec_edge = torch.sqrt(torch.sum((rec[:, edge_verts_index[:, 0], :] - rec[:, edge_verts_index[:, 1], :]) ** 2, dim = 2))
    return F.l1_loss(rec_edge, input_edge)

def unnormal(input, mean, std):
    output = input[:,:-1,:]*std + mean
    return torch.cat((output, input[:,-1:,:]), 1)

def normal(input, mean, std):
    output = (input[:,:-1,:] - mean) / std
    return torch.cat((output, input[:,-1:,:]), 1)



def cal_volloss(rec_v, GT_v, faces, vert_part_index, face_part_index, vert_part_index_dict, part_index_in_allpart):
    for i in range(len(vert_part_index_dict)):
        if i in part_index_in_allpart:
            tmp_vert_index = torch.where(vert_part_index == i)[0].long()
            tmp_face_index = torch.where(face_part_index == i)[0].long()
            tmp_f = faces[tmp_face_index].long()
            # rec_v[tmp_vert_index, :] = rec_v[tmp_vert_index, :] - torch.mean(rec_v[tmp_vert_index, :].detach(), dim = 0)[None]
            # GT_v[tmp_vert_index, :] = GT_v[tmp_vert_index, :] - torch.mean(GT_v[tmp_vert_index, :].detach(), dim = 0)[None]
            rec_vol = torch.sum(torch.cross(rec_v[tmp_f[:, 0], :], rec_v[tmp_f[:, 1], :]) * rec_v[tmp_f[:, 2], :])
            GT_vol = torch.sum(torch.cross(GT_v[tmp_f[:, 0], :], GT_v[tmp_f[:, 1], :]) * GT_v[tmp_f[:, 2], :])
            # print(rec_vol, GT_vol)
            if i == part_index_in_allpart[0]:
                vol_loss = F.l1_loss(torch.abs(rec_vol / GT_vol), torch.abs(GT_vol / GT_vol))
            else:
                vol_loss = vol_loss + F.l1_loss(torch.abs(rec_vol / GT_vol), torch.abs(GT_vol / GT_vol))
    return vol_loss / len(part_index_in_allpart)

def train_autoencoder_dataloader_nonormal(dataloader_train, dataloader_val,
                                 device, model, optim, loss_fn, 
                                 start_epoch, n_epochs, eval_freq, dataloader_interp, scheduler,
                                 writer, shapedata,metadata_dir, samples_dir, checkpoint_path,
                                 J_regressor,vert_part_index_dict, partname_list, save_recons):

    f_np = shapedata.reference_mesh.f.astype(np.int32)
    faces = torch.from_numpy(f_np)
    vert_part_index = torch.ones(6890)
    for k,v in enumerate(vert_part_index_dict.values()):
        vert_part_index[v] = k
    face_part_index = torch.ones(13776)
    for k, tmp_face in enumerate(faces):
        if vert_part_index[tmp_face[0]] == vert_part_index[tmp_face[1]] and vert_part_index[tmp_face[0]] == vert_part_index[tmp_face[2]]:
            face_part_index[k] = vert_part_index[tmp_face[0]]
        else:
            face_part_index[k] = 100
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
    if cfg.TRAIN.leafkeep_flag:
        leaf_list = [0,7,10,13,16]
    else:
        leaf_list = []
    edge_verts_index = torch.from_numpy(np.load(os.path.join(cfg.PATH.root_dir, 'asset', 'edge_verts_index.npy'))).long().to(device)
    total_steps = (start_epoch - 1)*len(dataloader_train)
    eval_freq = len(dataloader_train)
    J_regressor = torch.from_numpy(J_regressor.astype(np.float32)).to(device)
    part_index_in_allpart = []
    for i in cfg.CONSTANTS.noleaf_part_list:
        part_index_in_allpart.append(cfg.CONSTANTS.part_list.index(i))
    part_index_in_measure = []
    for i in cfg.CONSTANTS.noleaf_part_list:
        part_index_in_measure.append(cfg.CONSTANTS.measure_part_list.index(i))
    for epoch in range(start_epoch, n_epochs + 1):
        model.train()
        if epoch == start_epoch:
            dataloader_interp_iter = iter(dataloader_interp)
        tloss = []
        rec_loss = torch.zeros(1).to(device)
        edgereg_loss = torch.zeros(1).to(device)
        zpartreg_loss = torch.zeros(1).to(device)
        vol_loss = torch.zeros(1).to(device)
        interp_kps_loss = torch.zeros(1).to(device)
        interp_euc_loss=torch.zeros(1).to(device)
        exc_kps_loss = torch.zeros(1).to(device)
        exc_euc_loss=torch.zeros(1).to(device)

        for b, sample_dict in enumerate(tqdm(dataloader_train)):
            optim.zero_grad()
            tx = copy.deepcopy(sample_dict['verts'].to(device))
            kps_GT = torch.matmul(J_regressor, tx[:, :-1, :]).float()
            cur_bsize = tx.shape[0]
            point_num = tx.shape[1] - 1
            tx_hat, tx_zpart, _ = model(tx, kps_GT[:, kps_keep])
            rec_loss = loss_fn(tx, tx_hat)
            loss = rec_loss
            if epoch > cfg.TRAIN.edgereg_epoch and  cfg.TRAIN.edgereg_w > 0:
                for i in range(tx.shape[0]): 
                    if i == 0:
                        edgereg_loss = compute_score(tx_hat[i].unsqueeze(0),f_np,get_target(tx[i].cpu().numpy(),f_np,1,tx_hat.device))
                    else:
                        edgereg_loss = edgereg_loss + compute_score(tx_hat[i].unsqueeze(0),f_np,get_target(tx[i].cpu().numpy(),f_np,1,tx_hat.device))
                edgereg_loss = edgereg_loss / tx.shape[0]  
                loss = loss + cfg.TRAIN.edgereg_w * edgereg_loss
            if epoch > cfg.TRAIN.zpartreg_epoch and  cfg.TRAIN.zpartreg_w > 0:
                tx_measure = sample_dict['measure'].to(device)
                tx_zpart_m = torch.sqrt(torch.sum(tx_zpart ** 2, dim=2))
                if not cfg.TRAIN.relat_flag:
                    zpartreg_loss = F.l1_loss(tx_zpart_m[:, part_index_in_allpart], tx_measure[:, part_index_in_measure])
                else:
                    zpartreg_loss = F.l1_loss(tx_zpart_m[:, part_index_in_allpart] / tx_measure[:, part_index_in_measure], torch.ones_like(tx_measure[:, part_index_in_measure]).to(device))
                loss = loss + cfg.TRAIN.zpartreg_w * zpartreg_loss
            if epoch > cfg.TRAIN.interp_epoch:
                try:
                    input = dataloader_interp_iter.next()
                except StopIteration:
                    dataloader_interp_iter = iter(dataloader_interp)            
                    input = dataloader_interp_iter.next()
                    input = dataloader_interp_iter.next()
                tx_interp = copy.deepcopy(input['verts'].to(device))
                kps_GT_interp = torch.matmul(J_regressor, tx_interp[:, :-1, :]).float()
                if cfg.TRAIN.edit_mode == 'rand':
                    if cfg.TRAIN.editskl_flag:
                        factor = torch.rand(len(skl_keep)).to(device) * cfg.TRAIN.factor[0] + cfg.TRAIN.factor[1]
                        newkps_GT_interp = torch.matmul(J_regressor, tx_interp[:, :-1, :]).float()
                        newskl_GT_interp = kps2skl(newkps_GT_interp, 'ori_m')
                        newskl_GT_interp[:, skl_keep, 3] = newskl_GT_interp[:, skl_keep, 3] * factor[None]
                        newkps_GT_interp = skl2kps(newskl_GT_interp, 'ori_m')
                    else:
                        newkps_GT_interp = torch.matmul(J_regressor, tx_interp[:, :-1, :]).float()
                        newkps_GT_interp = newkps_GT_interp[:, kps_keep]
                    if cfg.TRAIN.rand_mode == 'rand':
                        part_num = random.randint(1, len(partname_list))
                    elif cfg.TRAIN.rand_mode == 'warm_up':
                        if epoch < 20:
                            part_num = 1
                        elif epoch < 50:
                            part_num = 2
                        elif epoch < 75:
                            part_num = 4
                        elif epoch < 100:
                            part_num = 8
                        else:
                            part_num = random.randint(1, len(partname_list))
                    part_index = random.sample(list(range(len(cfg.CONSTANTS.part_list))), part_num)
                    if cfg.TRAIN.noleaf_flag: 
                        if 0 in part_index:
                            part_index.remove(0)
                            part_num = part_num - 1
                        elif 7 in part_index:
                            part_index.remove(7)
                            part_num = part_num - 1  
                        elif 10 in part_index:
                            part_index.remove(10)
                            part_num = part_num - 1
                        elif 13 in part_index:
                            part_index.remove(13)
                            part_num = part_num - 1
                        elif 16 in part_index:
                            part_index.remove(16)
                            part_num = part_num - 1 
                    a = torch.rand(part_num).to(device) * cfg.TRAIN.factor[0] + cfg.TRAIN.factor[1]
                    a = torch.tile(a[None], (tx_interp.shape[0], 1))
                elif cfg.TRAIN.edit_mode == 'equal':
                    if cfg.TRAIN.editskl_flag:
                        factor = torch.rand(1).to(device) * cfg.TRAIN.factor[0] + cfg.TRAIN.factor[1]
                        newkps_GT_interp = torch.matmul(J_regressor, tx_interp[:, :-1, :]).float()
                        newskl_GT_interp = kps2skl(newkps_GT_interp, 'ori_m')
                        newskl_GT_interp[:, skl_keep, 3] = newskl_GT_interp[:, skl_keep, 3] * factor
                        newkps_GT_interp = skl2kps(newskl_GT_interp, 'ori_m')
                    else:
                        newkps_GT_interp = torch.matmul(J_regressor, tx_interp[:, :-1, :]).float()
                        newkps_GT_interp = newkps_GT_interp[:, kps_keep]
                    part_index = part_index_in_allpart
                    factor = torch.rand(1).to(device) * cfg.TRAIN.factor[0] + cfg.TRAIN.factor[1]
                    a = torch.ones([tx_interp.shape[0], len(cfg.CONSTANTS.noleaf_part_list)], device = device) * factor
                elif cfg.TRAIN.edit_mode == 'exc':
                    newkps_GT_interp = torch.matmul(J_regressor, tx_interp[:, :-1, :]).float()
                    newkps_GT_interp = newkps_GT_interp[:, kps_keep]
                    part_index = part_index_in_allpart
                    tx_measure = input['measure'].to(device)
                    a =  torch.flip(tx_measure, dims = [0]) / tx_measure
                
                latent, latent_kps, dummy = model.encode(tx_interp, newkps_GT_interp)
                for k,v in enumerate(part_index): 
                    latent[:, v, :] = latent[:, v, :]*a[:, k][:, None]    
                rec_interp = model.decode(latent, latent_kps, dummy)    

                if cfg.TRAIN.interp_kps_w > 0:
                    kps_rec_interp = torch.matmul(J_regressor, rec_interp[:, :-1, :]).float()
                    interp_kps_loss = F.l1_loss(kps_rec_interp[:, kps_keep], newkps_GT_interp)
                    loss = loss + cfg.TRAIN.interp_kps_w * interp_kps_loss
                
                if cfg.TRAIN.interp_euc_w > 0:
                    try:
                        len(angle_w)
                    except NameError:
                        pass
                    else:
                        del angle_w
                    angle_w = angle_skl(tx_interp[:, :-1, :], kps_GT_interp, partname_list, vert_part_index_dict, cfg.CONSTANTS.skl_list)

                if cfg.TRAIN.interp_euc_w > 0:
                    for i in range(len(partname_list)):
                        tmp_index_part = vert_part_index_dict[partname_list[i]]
                        De = calc_euclidean_dist_matrix(tx_interp[:, tmp_index_part, :])
                        De_r = calc_euclidean_dist_matrix(rec_interp[:, tmp_index_part, :]) 
                        if i in part_index:
                            De = De * a[:, part_index.index(i)][:, None, None]
                        if cfg.TRAIN.w_part_mode == 'n/N':
                            w_part = tmp_index_part.shape[0]/point_num
                        elif cfg.TRAIN.w_part_mode == '1/K':
                            w_part = 1/len(partname_list)
                        elif cfg.TRAIN.w_part_mode == '1/rand_num':
                            if i in part_index:
                                w_part = 0.99 * (1/len(part_index))
                            else:
                                w_part = 0.01 * (1/(len(partname_list) - len(part_index)))
                        if cfg.TRAIN.w_mode == 'all_one' or i in leaf_list:
                            w = torch.ones_like(angle_w[i].squeeze(-1), device = device)
                        elif cfg.TRAIN.w_mode == 'linear':
                            w = (angle_w[i].squeeze(-1).to(device).float()) / 90
                        elif cfg.TRAIN.w_mode == 'sin':
                            w = torch.sin(angle_w[i].squeeze(-1).float() / 180 * torch.pi).to(device)
                        elif cfg.TRAIN.w_mode == 'threshold':
                            w = (angle_w[i].squeeze(-1).to(device).float()) / 90
                            w = torch.where(w < cfg.TRAIN.w_threshold, torch.full_like(w, 0), w)
                        for batch_idx in range(w.shape[0]):
                            w[batch_idx, ...] = w[batch_idx, ...] - torch.diag_embed(torch.diag(w[batch_idx, ...]))
                            # print(torch.diag(w[batch_idx, ...]))
                        if i == 0:
                            nozero_index = torch.where((w * De) != 0)
                            # print(w[nozero_index].shape, De_r.shape, De.shape)
                            if not cfg.TRAIN.relat_flag:
                                interp_euc_loss = (w_part)*F.l1_loss(w[nozero_index]*De_r[nozero_index].float(), w[nozero_index]*De[nozero_index])
                            else:
                                interp_euc_loss = (w_part)*F.l1_loss(w[nozero_index]*(De_r[nozero_index].float()) / (De[nozero_index]), w[nozero_index]*torch.ones_like(w[nozero_index]).to(device))
                        else:   
                            nozero_index = torch.where((w * De) != 0)
                            if not cfg.TRAIN.relat_flag:
                                interp_euc_loss = interp_euc_loss + (w_part)*F.l1_loss(w[nozero_index]*De_r[nozero_index].float(), w[nozero_index]*De[nozero_index])
                            else:
                                interp_euc_loss = interp_euc_loss + (w_part)*F.l1_loss(w[nozero_index]*(De_r[nozero_index].float()) / (De[nozero_index]), w[nozero_index]*torch.ones_like(w[nozero_index]).to(device))
                    loss = loss + cfg.TRAIN.interp_euc_w*interp_euc_loss

            if epoch > cfg.TRAIN.exc_epoch:
                try:
                    input = dataloader_interp_iter.next()
                except StopIteration:
                    dataloader_interp_iter = iter(dataloader_interp)            
                    input = dataloader_interp_iter.next()
                    input = dataloader_interp_iter.next()

                tx_exc = copy.deepcopy(input['verts']) 
                tx_exc = tx_exc.to(device)
                kps_GT_exc = copy.deepcopy(torch.matmul(J_regressor, tx_exc[:, :-1, :]).float())
                if cfg.TRAIN.exc_mode == 'ori_m':

                    newkps_GT_exc = torch.flip(torch.matmul(J_regressor, tx_exc[:, :-1, :]).float(), dims = [0])
                    newkps_GT_exc = newkps_GT_exc[:, kps_keep]
                elif cfg.TRAIN.exc_mode == 'ori_or_m':
                    newkps_GT_exc = torch.matmul(J_regressor, tx_exc[:, :-1, :]).float()
                    skl_GT_exc = kps2skl(newkps_GT_exc, 'ori_m')
                    if np.random.rand(1) > 0.5:
                        exc_mode = 'ori'
                        skl_GT_exc[:, newskl_keep, :3] = torch.flip(skl_GT_exc[:, newskl_keep, :3], dims = [0])
                    else:
                        exc_mode = 'm'
                        skl_GT_exc[:, skl_keep, 3] = torch.flip(skl_GT_exc[:, skl_keep, 3], dims = [0])
                    newkps_GT_exc = skl2kps(skl_GT_exc, 'ori_m')
                elif cfg.TRAIN.exc_mode == 'ori':
                    exc_mode = 'ori'
                    newkps_GT_exc = torch.matmul(J_regressor, tx_exc[:, :-1, :]).float()
                    skl_GT_exc = kps2skl(newkps_GT_exc, 'ori_m')
                    skl_GT_exc[:, newskl_keep, :3] = torch.flip(skl_GT_exc[:, newskl_keep, :3], dims = [0])
                    newkps_GT_exc = skl2kps(skl_GT_exc, 'ori_m')
                # print(torch.mean(torch.abs(kps_GT_exc[:, kps_keep]-newkps_GT_exc)))
                # print(exc_mode)
                latent, latent_kps, dummy = model.encode(tx_exc, newkps_GT_exc)
                
                rec_exc = model.decode(latent, latent_kps, dummy)  

                if epoch > cfg.TRAIN.vol_epoch and cfg.TRAIN.vol_w > 0:
                    if exc_mode == 'ori':
                        for i in range(rec_exc.shape[0]):
                            if i == 0:
                                vol_loss = cal_volloss(rec_exc[i, :-1, :], tx_exc[i, :-1, :], faces, vert_part_index, face_part_index, vert_part_index_dict, part_index_in_allpart)
                            else:
                                vol_loss = vol_loss + cal_volloss(rec_exc[i, :-1, :], tx_exc[i, :-1, :], faces, vert_part_index, face_part_index, vert_part_index_dict, part_index_in_allpart)
                        vol_loss = vol_loss / rec_exc.shape[0]
                        loss = loss + cfg.TRAIN.vol_w * vol_loss
                    else:
                        vol_loss = torch.zeros(1).to(device)

                if cfg.TRAIN.exc_kps_w > 0:
                    kps_rec_exc = torch.matmul(J_regressor, rec_exc[:, :-1, :]).float()
                    # print(torch.mean(torch.abs(kps_GT_exc[:, kps_keep]-newkps_GT_exc)))
                    # print(torch.mean(torch.abs(kps_rec_exc[:, kps_keep] - kps_GT_exc[:, kps_keep])))
                    # print(torch.mean(torch.abs(kps_rec_exc[:, kps_keep]-newkps_GT_exc)))
                    # print(exc_mode)
                    exc_kps_loss = F.l1_loss(kps_rec_exc[:, kps_keep], newkps_GT_exc)
                    loss = loss + cfg.TRAIN.exc_kps_w * exc_kps_loss

                if cfg.TRAIN.exc_euc_w > 0:
                    try:
                        len(angle_w)
                    except NameError:
                        pass
                    else:
                        del angle_w
                    angle_w = angle_skl(tx_exc[:, :-1, :], kps_GT_exc, partname_list, vert_part_index_dict, cfg.CONSTANTS.skl_list)

                if cfg.TRAIN.exc_euc_w > 0:
                    for i in range(len(partname_list)):
                        tmp_index_part = vert_part_index_dict[partname_list[i]]
                        De = calc_euclidean_dist_matrix(tx_exc[:, tmp_index_part, :])
                        De_r = calc_euclidean_dist_matrix(rec_exc[:, tmp_index_part, :])  
                        if cfg.TRAIN.w_part_mode == 'n/N':
                            w_part = tmp_index_part.shape[0]/point_num
                        elif cfg.TRAIN.w_part_mode == '1/K':
                            w_part = 1/len(partname_list)
                        elif cfg.TRAIN.w_part_mode == '1/rand_num':
                            w_part = 1/len(partname_list)
                        if cfg.TRAIN.w_mode == 'all_one' or i in leaf_list:
                            w = torch.ones_like(angle_w[i].squeeze(-1), device = device)
                        elif cfg.TRAIN.w_mode == 'linear':
                            w = (angle_w[i].squeeze(-1).to(device).float()) / 90
                        elif cfg.TRAIN.w_mode == 'sin':
                            w = torch.sin(angle_w[i].squeeze(-1).float() / 180 * torch.pi).to(device)
                        elif cfg.TRAIN.w_mode == 'threshold':
                            w = (angle_w[i].squeeze(-1).to(device).float()) / 90
                            w = torch.where(w < cfg.TRAIN.w_threshold, torch.full_like(w, 0), w)
                        
                        for batch_idx in range(w.shape[0]):
                            w[batch_idx, ...] = w[batch_idx, ...] - torch.diag_embed(torch.diag(w[batch_idx, ...]))
                
                        if i == 0:
                            nozero_index = torch.where((w * De) != 0)
                            if not cfg.TRAIN.relat_flag:
                                exc_euc_loss = (w_part)*F.l1_loss(w[nozero_index]*De_r[nozero_index].float(), w[nozero_index]*De[nozero_index])
                            else:
                                exc_euc_loss = (w_part)*F.l1_loss(w[nozero_index]*(De_r[nozero_index].float()) / (De[nozero_index]), w[nozero_index]*torch.ones_like(w[nozero_index]).to(device))                    
                        else:      
                            nozero_index = torch.where((w * De) != 0)
                            if not cfg.TRAIN.relat_flag:
                                exc_euc_loss = exc_euc_loss + (w_part)*F.l1_loss(w[nozero_index]*De_r[nozero_index].float(), w[nozero_index]*De[nozero_index])
                            else:
                                exc_euc_loss = exc_euc_loss + (w_part)*F.l1_loss(w[nozero_index]*(De_r[nozero_index].float()) / (De[nozero_index]), w[nozero_index]*torch.ones_like(w[nozero_index]).to(device))
                    loss = loss + cfg.TRAIN.exc_euc_w*exc_euc_loss             

            loss.backward()
            optim.step()
            tloss.append(cur_bsize * loss.item())

            if writer and total_steps % eval_freq == 0:
                writer.add_scalar('loss/loss/data_loss',loss.item(),total_steps)
                writer.add_scalar('loss/loss/rec_loss',rec_loss.item(),total_steps)
                writer.add_scalar('loss/loss/edgereg_loss',edgereg_loss.item(),total_steps)
                writer.add_scalar('loss/loss/zpartreg_loss',zpartreg_loss.item(),total_steps)
                writer.add_scalar('loss/loss/vol_loss',vol_loss.item(),total_steps)
                writer.add_scalar('loss/loss/interp_kps_loss',interp_kps_loss.item(),total_steps)            
                writer.add_scalar('loss/loss/interp_euc_loss',interp_euc_loss.item(),total_steps)
                writer.add_scalar('loss/loss/exc_kps_loss',exc_kps_loss.item(),total_steps)
                writer.add_scalar('loss/loss/exc_euc_loss',exc_euc_loss.item(),total_steps)
             
            total_steps += 1

        # validate
        if True:
            model.eval()
            vloss = []
            with torch.no_grad():
                for b, sample_dict in enumerate(tqdm(dataloader_val)):

                    tx_val = sample_dict['verts'].to(device)
                    tx_idx = sample_dict['idx']
                    cur_bsize = tx_val.shape[0]
                    kps_GT_val = torch.matmul(J_regressor, tx_val[:, :-1, :]).float()
                    

                    tx_hat_val = model(tx_val, kps_GT_val[:, kps_keep])[0]
                    rec_val_loss = loss_fn(tx_val[:, :-1, :], tx_hat_val[:, :-1, :]) 
                    
                    # tx_hat_val = model(tx, kps_GT)[0]               
                    # loss = loss_fn(tx, tx_hat_val)
                    
                    vloss.append(cur_bsize * rec_val_loss.item())
    

        if scheduler:
            scheduler.step()
            
        epoch_tloss = sum(tloss) / float(len(dataloader_train.dataset))
        writer.add_scalar('avg_epoch_train_loss',epoch_tloss,epoch)
        if len(dataloader_val.dataset) > 0:
        # if False:
            epoch_vloss = sum(vloss) / float(len(dataloader_val.dataset))
            writer.add_scalar('avg_epoch_valid_loss', epoch_vloss,epoch)
            print('epoch {0} | tr {1} | val {2}'.format(epoch,epoch_tloss,epoch_vloss))
        else:
            print('epoch {0} | tr {1} '.format(epoch,epoch_tloss))
        model = model.cpu()
  
        # torch.save({'epoch': epoch,
        #     'autoencoder_state_dict': model.state_dict(),
        #     'optimizer_state_dict' : optim.state_dict(),
        #     'scheduler_state_dict': scheduler.state_dict(),
        # },os.path.join(metadata_dir, checkpoint_path+'.pth.tar'))
        
        if epoch % cfg.TRAIN.ck_frequency == 0:
            torch.save({'epoch': epoch,
            'autoencoder_state_dict': model.state_dict(),
            'optimizer_state_dict' : optim.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            },os.path.join(metadata_dir, checkpoint_path+'%s.pth.tar'%(epoch)))

        model = model.to(device)

        if save_recons:
            with torch.no_grad():

                if epoch % 50 == 0:
                    mesh_ind = [0]
                    msh = tx[mesh_ind[0]:1,0:-1,:].detach().cpu().numpy()
                    shapedata.save_meshes(os.path.join(samples_dir,'epoch{0}_GT'.format(epoch)),
                                                     msh, [tx_idx[mesh_ind[0]]])
                    mesh_ind = [0]
                    msh = tx_hat[mesh_ind[0]:1,0:-1,:].detach().cpu().numpy()
                    shapedata.save_meshes(os.path.join(samples_dir,'epoch{0}_rec'.format(epoch)),
                                                    msh, [tx_idx[mesh_ind[0]]])
                
    print('~FIN~')

def train_autoencoder_dataloader(dataloader_train, dataloader_val,
                                 device, model, optim, loss_fn, 
                                 start_epoch, n_epochs, eval_freq, dataloader_interp, scheduler,
                                 writer, shapedata,metadata_dir, samples_dir, checkpoint_path,
                                 J_regressor,vert_part_index_dict, partname_list, save_recons):
    f_np = shapedata.reference_mesh.f.astype(np.int32)
    total_steps = (start_epoch - 1)*len(dataloader_train)
    eval_freq = len(dataloader_train)
    J_regressor = torch.from_numpy(J_regressor.astype(np.float32)).to(device)
    part_index_in_allpart = []
    for i in cfg.CONSTANTS.noleaf_part_list:
        part_index_in_allpart.append(cfg.CONSTANTS.part_list.index(i))
    part_index_in_measure = []
    for i in cfg.CONSTANTS.noleaf_part_list:
        part_index_in_measure.append(cfg.CONSTANTS.measure_part_list.index(i))

    for epoch in range(start_epoch, n_epochs + 1):
        model.train()
        tloss = []
        rec_loss = torch.zeros(1).to(device)
        edgereg_loss = torch.zeros(1).to(device)
        for b, sample_dict in enumerate(tqdm(dataloader_train)):
            optim.zero_grad()
            tx = sample_dict['verts'].to(device)
            kps_GT = torch.matmul(J_regressor, tx[:, :-1, :]).reshape(tx.shape[0], 72).float()
            cur_bsize = tx.shape[0]
            tx_hat = model(tx)[0]
            rec_loss = loss_fn(tx, tx_hat)
            loss = rec_loss
            if epoch > cfg.TRAIN.edgereg_epoch and  cfg.TRAIN.edgereg_w > 0:
                edgereg_loss = torch.zeros(1).to(device)
                for i in range(tx.shape[0]): 
                    edgereg_loss = edgereg_loss + compute_score(tx_hat[i].unsqueeze(0),f_np,get_target(tx[i].cpu().numpy(),f_np,1,tx_hat.device))
                edgereg_loss = edgereg_loss / tx.shape[0]
                loss = loss + cfg.TRAIN.edgereg_w * edgereg_loss       
            loss.backward()
            optim.step()
         
            
            tloss.append(cur_bsize * loss.item())

            if writer and total_steps % eval_freq == 0:
                writer.add_scalar('loss/loss/data_loss',loss.item(),total_steps)
                writer.add_scalar('loss/loss/rec_loss',rec_loss.item(),total_steps)
                writer.add_scalar('loss/loss/edgereg_loss',edgereg_loss.item(),total_steps)
            total_steps += 1

        # validate
        if True:
            model.eval()
            vloss = []
            with torch.no_grad():
                for b, sample_dict in enumerate(tqdm(dataloader_val)):

                    tx = sample_dict['verts'].to(device)
                    tx_idx = sample_dict['idx']
                    cur_bsize = tx.shape[0]
                    kps_GT = torch.matmul(J_regressor, tx[:, :-1, :]).reshape(tx.shape[0], 72).float()
                    

                    tx_hat_val = model(tx)[0]
                    rec_loss = loss_fn(tx[:, :-1, :], tx_hat_val[:, :-1, :]) 
                    
                    # tx_hat_val = model(tx, kps_GT)[0]               
                    # loss = loss_fn(tx, tx_hat_val)
                    
                    vloss.append(cur_bsize * rec_loss.item())
    
        if scheduler:
            scheduler.step()
            
        epoch_tloss = sum(tloss) / float(len(dataloader_train.dataset))
        writer.add_scalar('avg_epoch_train_loss',epoch_tloss,epoch)
        if len(dataloader_val.dataset) > 0:
        # if False:
            epoch_vloss = sum(vloss) / float(len(dataloader_val.dataset))
            writer.add_scalar('avg_epoch_valid_loss', epoch_vloss,epoch)
            print('epoch {0} | tr {1} | val {2}'.format(epoch,epoch_tloss,epoch_vloss))
        else:
            print('epoch {0} | tr {1} '.format(epoch,epoch_tloss))
        model = model.cpu()
  
        # torch.save({'epoch': epoch,
        #     'autoencoder_state_dict': model.state_dict(),
        #     'optimizer_state_dict' : optim.state_dict(),
        #     'scheduler_state_dict': scheduler.state_dict(),
        # },os.path.join(metadata_dir, checkpoint_path+'.pth.tar'))
        
        if epoch % cfg.TRAIN.ck_frequency == 0:
            torch.save({'epoch': epoch,
            'autoencoder_state_dict': model.state_dict(),
            'optimizer_state_dict' : optim.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            },os.path.join(metadata_dir, checkpoint_path+'%s.pth.tar'%(epoch)))

        model = model.to(device)

        if save_recons:
            with torch.no_grad():
                                                
                if epoch % 50 == 0:
                    mesh_ind = [0]
                    msh = tx_hat_val[mesh_ind[0]:1,0:-1,:].detach().cpu().numpy()
                    shapedata.save_meshes(os.path.join(samples_dir,'epoch_val{0}'.format(epoch)),
                                                    msh, [tx_idx[mesh_ind[0]]])
                    mesh_ind = [0]
                    msh = tx_hat[mesh_ind[0]:1,0:-1,:].detach().cpu().numpy()
                    shapedata.save_meshes(os.path.join(samples_dir,'epoch_train{0}'.format(epoch)),
                                                    msh, [tx_idx[mesh_ind[0]]])
    print('~FIN~')