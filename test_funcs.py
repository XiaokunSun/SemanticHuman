import torch
import copy
from tqdm import tqdm
import numpy as np
import os
from configure.cfgs import cfg
from utils_SH import *

def unnormal(input, mean, std):
    output = input[:,:-1,:]*std + mean
    return torch.cat((output, input[:,-1:,:]), 1)

def normal(input, mean, std):
    output = (input[:,:-1,:] - mean) / std
    return torch.cat((output, input[:,-1:,:]), 1)

def test_autoencoder_dataloader(device, model, dataloader_test, shapedata, J_regressor, mm_constant = 1000, unnormal_flag = False):
    model.eval()
    l1_loss = 0
    l2_loss = 0
    J_regressor = torch.from_numpy(J_regressor.astype(np.float32)).to(device)
    with torch.no_grad():
        for i, sample_dict in enumerate(tqdm(dataloader_test)):
            tx = sample_dict['verts'].to(device)
            prediction, z = model(tx)  
            if i==0:
                predictions = copy.deepcopy(prediction)
            else:
                predictions = torch.cat([predictions,prediction],0) 
            if i==0:
                z_s = copy.deepcopy(z)
            else:
                z_s = torch.cat([z_s, z],0) 
            if i==0:
                tx_s = copy.deepcopy(tx)
            else:
                tx_s = torch.cat([tx_s, tx],0) 

            if dataloader_test.dataset.dummy_node:
                x_recon = prediction[:,:-1]
                x = tx[:,:-1]
            else:
                x_recon = prediction
                x = tx      
                          
            l1_loss+= torch.mean(torch.abs(x_recon-x))*x.shape[0]/float(len(dataloader_test.dataset))
            x_recon = (x_recon) * mm_constant
            x = (x) * mm_constant
            l2_loss+= torch.mean(torch.sqrt(torch.sum((x_recon - x)**2,dim=2)))*x.shape[0]/float(len(dataloader_test.dataset))
            
        predictions = predictions.cpu().numpy()
        z_s = z_s.cpu().numpy()
        tx_s = tx_s.cpu().numpy()
        l1_loss = l1_loss.item()
        l2_loss = l2_loss.item()
    
    return predictions, z_s, tx_s, l1_loss, l2_loss



def test_autoencoder_dataloader_nonormal(device, model, dataloader_test, shapedata, J_regressor, mm_constant = 1000, unnormal_flag = False):
    kps_keep = list(range(len(cfg.CONSTANTS.newskl_list) + 4))
    if cfg.TRAIN.kpskeep_flag:
        for i in [3,13,14]:
            kps_keep.remove(i)
    model.eval()
    l1_loss = 0
    l2_loss = 0
    J_regressor = torch.from_numpy(J_regressor.astype(np.float32)).to(device)
    with torch.no_grad():
        for i, sample_dict in enumerate(tqdm(dataloader_test)):
            tx = sample_dict['verts'].to(device)
            kps_GT = torch.matmul(J_regressor, tx[:, :-1, :]).float()
            prediction, z, z_kps = model(tx, kps_GT[:, kps_keep])  
            if i==0:
                predictions = copy.deepcopy(prediction)
            else:
                predictions = torch.cat([predictions,prediction],0) 
            if i==0:
                z_s = copy.deepcopy(z)
            else:
                z_s = torch.cat([z_s, z],0) 
            if i==0:
                z_kps_s = copy.deepcopy(z_kps)
            else:
                z_kps_s = torch.cat([z_kps_s, z_kps],0) 
            if i==0:
                tx_s = copy.deepcopy(tx)
            else:
                tx_s = torch.cat([tx_s, tx],0) 
            if dataloader_test.dataset.dummy_node:
                x_recon = prediction[:,:-1]
                x = tx[:,:-1]
            else:
                x_recon = prediction
                x = tx
            l1_loss+= torch.mean(torch.abs(x_recon-x))*x.shape[0]/float(len(dataloader_test.dataset))

            x_recon = (x_recon) * mm_constant
            x = (x) * mm_constant
            l2_loss+= torch.mean(torch.sqrt(torch.sum((x_recon - x)**2,dim=2)))*x.shape[0]/float(len(dataloader_test.dataset))
            
        predictions = predictions.cpu().numpy()
        z_s = z_s.cpu().numpy()
        z_kps_s = z_kps_s.cpu().numpy()
        tx_s = tx_s.cpu().numpy()
        l1_loss = l1_loss.item()
        l2_loss = l2_loss.item()
    
    return predictions, z_s, z_kps_s, tx_s, l1_loss, l2_loss