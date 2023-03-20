from torch.utils.data import Dataset
import torch
import numpy as np
import os

from configure.cfgs import cfg


class autoencoder_dataset(Dataset):

    def __init__(self, root_dir, points_dataset, shapedata, normalization = 'No', dummy_node = True, measure_flag = False, anglew_flag = False, J_regressor = None):
        
        self.shapedata = shapedata
        self.normalization = normalization
        self.root_dir = root_dir
        self.points_dataset = points_dataset
        self.dummy_node = dummy_node
        self.paths = np.load(os.path.join(root_dir, 'paths_'+points_dataset+'.npy'))
        self.measure_flag = measure_flag
        self.J_regressor = J_regressor
        self.anglew_flag = anglew_flag

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        basename = self.paths[idx]
        verts_init = np.load(os.path.join(self.root_dir,'points'+'_'+self.points_dataset, basename+'.npy'))
        if 'zeromean' in self.normalization:
            verts_init = verts_init -  np.mean(verts_init, axis = 0)
        if 'zeroroot' in self.normalization:
            verts_init = verts_init - np.matmul(self.J_regressor, verts_init)[0]
        if 'onelength' in self.normalization:
            verts_init = verts_init / (np.max(verts_init, axis = 0) - np.min(verts_init, axis = 0))[1] * 1.5
        if 'small' in self.normalization:
            verts_init = verts_init / 1.5
        if 'gass' in self.normalization:
            verts_init = verts_init - self.shapedata.mean
            verts_init = verts_init / self.shapedata.std
        if 'normal' in self.normalization:
            verts_init = verts_init -  self.shapedata.center[idx, :]
            verts_init = verts_init * self.shapedata.scale[idx]
        verts_init[np.where(np.isnan(verts_init))]=0.0
        verts_init = verts_init.astype('float32')
        if self.dummy_node:
            verts = np.zeros((verts_init.shape[0]+1,verts_init.shape[1]),dtype=np.float32)
            verts[:-1,:] = verts_init
            verts_init = verts
        verts = torch.Tensor(verts_init)

        input_dict = {'verts':verts, 'idx':idx}

        if self.measure_flag:
            measure = np.load(os.path.join(self.root_dir,'measure'+'_'+self.points_dataset, basename+'.npy')).astype('float32')
            input_dict['measure'] = torch.from_numpy(measure)

        
        return input_dict