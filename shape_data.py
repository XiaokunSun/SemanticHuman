
### Code obtained and modified from https://github.com/anuragranj/coma, Copyright (c) 2018 Anurag Ranjan, Timo Bolkart, Soubhik Sanyal, Michael J. Black and the Max Planck Gesellschaft

import os
import numpy as np

try:
    import psbody.mesh
    found = True
except ImportError:
    found = False
if found:
    from psbody.mesh import Mesh

from trimesh.exchange.export import export_mesh
import trimesh

import time
from tqdm import tqdm

class ShapeData(object):
    def __init__(self, nVal, train_file, test_file, reference_mesh_file, normalization = 'No', meshpackage = 'mpi-mesh', mean_subtraction_only = False):
        self.nVal = nVal
        self.train_file = train_file
        self.test_file = test_file
        self.vertices_train = None
        self.vertices_val = None
        self.vertices_test = None
        self.n_vertex = None
        self.n_features = None
        self.normalization = normalization
        self.meshpackage = meshpackage
        self.mean_subtraction_only = mean_subtraction_only
        
        if self.meshpackage == 'trimesh':
            self.reference_mesh = trimesh.load(reference_mesh_file, process = False)
        elif self.meshpackage =='mpi-mesh':
            self.reference_mesh = Mesh(filename=reference_mesh_file)

        self.load()
        if self.normalization == 'gass':
            self.mean = np.mean(self.vertices_train, axis=0)
            self.std = np.std(self.vertices_train, axis=0)
        elif self.normalization == 'normal':
            self.center = (np.max(self.vertices_test, axis=1) + np.min(self.vertices_test, axis=1)) / 2
            # self.scale = 1 / np.max(np.max(self.vertices_test, axis=1) - np.min(self.vertices_test, axis=1), axis = 1)
            self.scale = 1 / (np.max(self.vertices_test, axis=1) - np.min(self.vertices_test, axis=1))
        
    def load(self):
        vertices_train = np.load(self.train_file)
        self.vertices_train = vertices_train[:-self.nVal]
        self.vertices_val = vertices_train[-self.nVal:]

        self.n_vertex = self.vertices_train.shape[1]
        self.n_features = self.vertices_train.shape[2]

        if os.path.exists(self.test_file):
            self.vertices_test = np.load(self.test_file)
            self.vertices_test = self.vertices_test

    # def normalize(self):
    #     if self.load_flag:
    #         if self.normalization:
    #             if self.mean_subtraction_only:
    #                 self.std = np.ones_like((self.std))
    #             self.vertices_train = self.vertices_train - self.mean
    #             self.vertices_train = self.vertices_train/self.std
    #             self.vertices_train[np.where(np.isnan(self.vertices_train))]=0.0

    #             self.vertices_val = self.vertices_val - self.mean
    #             self.vertices_val = self.vertices_val/self.std
    #             self.vertices_val[np.where(np.isnan(self.vertices_val))]=0.0

    #             if self.vertices_test is not None:
    #                 self.vertices_test = self.vertices_test - self.mean
    #                 self.vertices_test = self.vertices_test/self.std
    #                 self.vertices_test[np.where(np.isnan(self.vertices_test))]=0.0
                
    #             self.N = self.vertices_train.shape[0]

    #             print('Vertices normalized')
    #         else:
    #             print('Vertices not normalized')


    def save_meshes(self, filename, meshes, mesh_indices):
        for i in range(meshes.shape[0]):
            if self.normalization == 'gass':
                vertices = meshes[i].reshape((self.n_vertex, self.n_features))*self.std + self.mean
            elif self.normalization == 'normal':
                vertices = meshes[i].reshape((self.n_vertex, self.n_features))/self.scale[mesh_indices[i]] + self.center[mesh_indices[i], :]
            elif self.normalization == 'No':
                vertices = meshes[i].reshape((self.n_vertex, self.n_features))
            if self.meshpackage == 'trimesh':
                new_mesh = self.reference_mesh
                if self.n_features == 3:
                    new_mesh.vertices = vertices
                elif self.n_features == 6:
                    new_mesh.vertices = vertices[:,0:3]
                    colors = vertices[:,3:]
                    colors[np.where(colors<0)]=0
                    colors[np.where(colors>1)]=1
                    vertices[:,3:] = colors
                    new_mesh.visual = trimesh.visual.create_visual(vertex_colors = vertices[:,3:])
                else:
                    raise NotImplementedError
                new_mesh.export(filename+'.'+str(mesh_indices[i]).zfill(6)+'.ply','ply')   
            elif self.meshpackage =='mpi-mesh':
                if self.n_features == 3:
                    mesh = Mesh(v=vertices, f=self.reference_mesh.f)
                    mesh.write_obj(filename+'_'+str(mesh_indices[i]).zfill(6)+'.obj')
                    # mesh.write_ply(filename+'.'+str(mesh_indices[i]).zfill(6)+'.ply')
                else:
                    raise NotImplementedError
        return 0

    def save_meshes_withkps(self, filename, meshes, mesh_indices, kps_flag = False, skl_list = None, J_regressor = None):
        for i in range(meshes.shape[0]):
            if self.normalization == 'gass':
                vertices = meshes[i].reshape((self.n_vertex, self.n_features))*self.std + self.mean
            elif self.normalization == 'normal':
                vertices = meshes[i].reshape((self.n_vertex, self.n_features))/self.scale[mesh_indices[i]] + self.center[mesh_indices[i], :]
            elif self.normalization == 'No':
                vertices = meshes[i].reshape((self.n_vertex, self.n_features))
            kps = J_regressor.dot(vertices)
            if self.n_features == 3:
                f=self.reference_mesh.f
                num = 1000
                with open(filename, 'w') as fp:
                    v_i = 0
                    for tmp_v in vertices:
                        fp.write('v %f %f %f %d %d %d\n' % (tmp_v[0], tmp_v[1], tmp_v[2], 127, 127, 127))
                        v_i = v_i + 1
                    if kps_flag:
                        color = [0, 0, 0]
                        for kps_index in skl_list:
                            if len(kps_index) == 2:
                                kps_points = (kps[kps_index[1], :] - kps[kps_index[0], :])[:, None] * np.linspace(0, 0.9, num)[None] + np.tile(kps[kps_index[0], :][:, None], (1, num))
                            elif len(kps_index) == 3:
                                kps_points = ((kps[kps_index[1], :] + kps[kps_index[2], :]) / 2 - kps[kps_index[0], :])[:, None] * np.linspace(0, 0.9, num)[None] + np.tile(kps[kps_index[0], :][:, None], (1, num))
                            for tmp_v in kps_points.T:
                                fp.write('v %f %f %f %d %d %d\n' % (tmp_v[0], tmp_v[1], tmp_v[2], color[0], color[1],color[2]))
                    for tmp_f in f + 1:
                        fp.write('f %d %d %d\n' % (tmp_f[0], tmp_f[1], tmp_f[2]))
        return 0
    