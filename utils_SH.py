from email.policy import strict
import numpy as np
import os
import copy
import math
import torch
torch_version = torch.__version__
from utils_distance import distance_GIH, calc_euclidean_dist_matrix
from configure.cfgs import cfg, update_cfg
import torch.nn.functional as Function
import pickle
from psbody.mesh import Mesh
import mesh_sampling
import trimesh
from shape_data import ShapeData
from utils_spiral import get_adj_trigs, generate_spirals
from models import SpiralAutoencoder_multiz_partkps, SpiralAutoencoder
from sklearn.metrics.pairwise import euclidean_distances
import random

parent_dict = {1: 0, 2: 0, 3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7, \
    11: 8, 12: 9, 13: 9, 14: 9, 15: 12, 16: 13, 17: 14, 18: 16, 19: 17, 20: 18, 21: 19, 22: 20, 23: 21}
child_dict = {0: [1, 2, 3], 1: [4], 2: [5], 3: [6], 4: [7], 5: [8], 6: [9], 7: [10],  \
    8: [11], 9: [12, 13, 14], 12: [15], 13: [16], 14: [17], 16: [18], 17: [19], 18: [20], 19: [21], 20: [22], 21: [23]}

def kps2skl(kps_tmp, skl_mode):
    skl_list = cfg.CONSTANTS.newskl_list # measure_skl_list newskl_list
    # print(skl_list)
    if kps_tmp.shape[1] == len(skl_list) + 4:
        kps = copy.deepcopy(kps_tmp)
    else:
        kps_keep = list(range(len(skl_list) + 4))
        for i in [3,13,14]:
            kps_keep.remove(i)
        kps = torch.zeros((kps_tmp.shape[0], len(skl_list) + 4, 3), device = kps_tmp.device)
        kps[:, kps_keep, :] = kps_tmp
    if skl_mode == 'ori_m' or skl_mode == 'kps_ori_m':
        skl = torch.zeros((kps.shape[0], len(skl_list), 4), device = kps.device)
        for index in range(len(skl_list)):
            if len(skl_list[index]) == 2:
                skl[:, index, :3] = (kps[:, skl_list[index][0], :] - kps[:, skl_list[index][1], :]) / (torch.sqrt(torch.sum((kps[:, skl_list[index][0], :] -  kps[:, skl_list[index][1], :]) ** 2, dim = 1)))[:, None]
                skl[:, index, -1] = torch.sqrt(torch.sum((kps[:, skl_list[index][0], :] -  kps[:, skl_list[index][1], :]) ** 2, dim = 1))
            elif len(skl_list[index]) == 3:
                skl[:, index, :3] = (kps[:, skl_list[index][0], :] - (kps[:, skl_list[index][1], :] + kps[:, skl_list[index][2], :]) / 2) / torch.sqrt(torch.sum((kps[:, skl_list[index][0], :] - (kps[:, skl_list[index][1], :] + kps[:, skl_list[index][2], :]) / 2) ** 2, dim = 1))[:, None]
                skl[:, index, -1] = torch.sqrt(torch.sum((kps[:, skl_list[index][0], :] - (kps[:, skl_list[index][1], :] + kps[:, skl_list[index][2], :]) / 2) ** 2, dim = 1))
    elif skl_mode == 'vec_m':
        skl = torch.zeros((kps.shape[0], len(skl_list), 4), device = kps.device)
        for index in range(len(skl_list)):
            if len(skl_list[index]) == 2:
                skl[:, index, :3] = (kps[:, skl_list[index][0], :] - kps[:, skl_list[index][1], :]) 
                skl[:, index, -1] = torch.sqrt(torch.sum((kps[:, skl_list[index][0], :] -  kps[:, skl_list[index][1], :]) ** 2, dim = 1))
            elif len(skl_list[index]) == 3:
                skl[:, index, :3] = (kps[:, skl_list[index][0], :] - (kps[:, skl_list[index][1], :] + kps[:, skl_list[index][2], :]) / 2) 
                skl[:, index, -1] = torch.sqrt(torch.sum((kps[:, skl_list[index][0], :] - (kps[:, skl_list[index][1], :] + kps[:, skl_list[index][2], :]) / 2) ** 2, dim = 1))
    elif skl_mode == 'vec':
        skl = torch.zeros((kps.shape[0], len(skl_list), 3), device = kps.device)
        for index in range(len(skl_list)):
            if len(skl_list[index]) == 2:
                skl[:, index, :] = (kps[:, skl_list[index][0], :] - kps[:, skl_list[index][1], :])
            elif len(skl_list[index]) == 3:
                skl[:, index, :] = (kps[:, skl_list[index][0], :] - (kps[:, skl_list[index][1], :] + kps[:, skl_list[index][2], :]) / 2)
    elif skl_mode == 'm':
        skl = torch.zeros((kps.shape[0], len(skl_list), 1), device = kps.device)
        for index in range(len(skl_list)):
            if len(skl_list[index]) == 2:
                skl[:, index, -1] = torch.sqrt(torch.sum((kps[:, skl_list[index][0], :] -  kps[:, skl_list[index][1], :]) ** 2, dim = 1))
            elif len(skl_list[index]) == 3:
                skl[:, index, -1] = torch.sqrt(torch.sum((kps[:, skl_list[index][0], :] - (kps[:, skl_list[index][1], :] + kps[:, skl_list[index][2], :]) / 2) ** 2, dim = 1))
    return skl

def skl2kps(skl, skl_mode):
    skl_list = cfg.CONSTANTS.newskl_list
    kps_keep = list(range(len(skl_list) + 4))
    for i in [3,13,14]:
        kps_keep.remove(i)
    kps = torch.zeros((skl.shape[0], len(skl_list) + 4, 3), device = skl.device)
    for k, kps_list in enumerate(skl_list):
        if skl_mode == 'vec':
            kps[:, kps_list[1], :] = - skl[:, k, :] + kps[:, kps_list[0], :]
        elif skl_mode == 'vec_m':
            kps[:, kps_list[1], :] = - skl[:, k, :3] + kps[:, kps_list[0], :3]
        elif skl_mode == 'ori_m' or skl_mode == 'kps_ori_m':
            kps[:, kps_list[1], :] = - (skl[:, k, :3] * skl[:, k, 3:]) + kps[:, kps_list[0], :]
    return kps[:, kps_keep, :]
    
def cal_length(kps, skl_list):
    '''
    kps: tensor [N_kps, 3]
    skl_list: list [N_part]
    return length: tensor [N_part]
    '''
    length =torch.zeros(len(skl_list), device = kps.device)
    for index in range(len(skl_list)):
        if len(skl_list[index]) == 2:
            length[index] = torch.sqrt(torch.sum((kps[skl_list[index][0], :] -  kps[skl_list[index][1], :]) ** 2))
        elif len(skl_list[index]) == 3:
            length[index] = torch.sqrt(torch.sum((kps[skl_list[index][0], :] - (kps[skl_list[index][1], :] + kps[skl_list[index][2], :]) / 2) ** 2))
    return length

def cal_girth(face_point, face_normal, points):
    '''
    face_point: tensor [3]
    face_normal: tensor [3]
    points: tensor [N, 2, 3]
    return girth: tensor [1], X: tensor [N_x, 3], order: list [N_x] 
    '''
    A = torch.zeros((points.shape[0], 3, 3), device = points.device)
    B = torch.zeros((points.shape[0], 3), device = points.device)
    line_point = points[:, 0, :]
    line_ori = points[:, 0, :] - points[:, 1, :]
    line_ori[torch.where(line_ori==0)] = 1e-6
    A[:, 0, :] = face_normal[None].repeat(points.shape[0], 1)
    A[:, 1, 0] = 1 / line_ori[:, 0]
    A[:, 1, 1] = -1 / line_ori[:, 1]
    A[:, 1, 2] = 0
    A[:, 2, 0] = 1 / line_ori[:, 0]
    A[:, 2, 1] = 0
    A[:, 2, 2] = -1 / line_ori[:, 2]
    B[:, 0] = torch.sum(face_point * face_normal)
    B[:, 1] = torch.sum(line_point * torch.cat((1/line_ori[:,0:1], -1/line_ori[:,1:2], torch.zeros((points.shape[0], 1), device = points.device)), dim = 1), dim = 1)
    B[:, 2] = torch.sum(line_point * torch.cat((1/line_ori[:,0:1], torch.zeros((points.shape[0], 1), device = points.device), -1/line_ori[:,2:]), dim = 1), dim = 1)
    X = torch.linalg.solve(A, B)
    X_mean = torch.mean(X, dim=0)
    X_vec = X - X_mean
    X_vec_m = torch.sqrt(torch.sum(X_vec * X_vec, dim=1))
    cos_theta = torch.sum(X_vec[0:1,:].repeat(X_vec.shape[0]-1, 1) * X_vec[1:,:], dim=1) / (X_vec_m[1:]*X_vec_m[0])
    if torch_version == '1.5.0':
        theta = torch.acos(cos_theta)/math.pi*180
    else :
        theta = torch.arccos(cos_theta)/math.pi*180
    cross_mul = torch.cross(X_vec[0:1,:].repeat(X_vec.shape[0]-1, 1), X_vec[1:,:])
    flag = torch.where((cross_mul[:,0] * cross_mul[:,1] * cross_mul[:,2]) > 0, torch.ones(X_vec.shape[0]-1).to(points.device), torch.ones(X_vec.shape[0]-1).to(points.device) * -1)
    _, order = torch.sort(torch.cat((torch.tensor([0]).to(points.device), theta * flag)))
    girth = torch.sqrt(torch.sum((X[order[0]] - X[order[-1]]) ** 2))
    for ii in range(X.shape[0] - 1):
        # print(girth_part)
        girth = girth + torch.sqrt(torch.sum((X[order[ii]] - X[order[ii+1]]) ** 2))
    # fin_order = [0] + (order+1).tolist()
    # girth = euc[order[0], order[-1]]
    # for i in range(len(order)-1):
    #     girth = girth + euc[order[i], order[i+1]]
    return girth, X, order

def measure_body_quick(v, kps, skl_list, factor_list, edge_point_index_list):
    '''
    v: tensor [N_v, 3]
    kps: tensor [N_kps, 3]
    skl_list: list [N_part]
    factor_list: list [N_part]
    edge_point_index_list: list [N_part]
    return girth_cp: tensor [N_part], length: tensor [N_part]
    '''
    girth_cp = []
    for i in range(len(factor_list)):
        girth_point_cp = v[edge_point_index_list[i][:, 0], :] * (1-factor_list[i]) + v[edge_point_index_list[i][:, 1], :] * factor_list[i]
        girth_part = torch.sqrt(torch.sum((girth_point_cp[0] - girth_point_cp[-1]) ** 2))
        for ii in range(girth_point_cp.shape[0] - 1):
            girth_part = girth_part + torch.sqrt(torch.sum((girth_point_cp[ii] - girth_point_cp[ii+1]) ** 2))
        girth_cp.append(girth_part)
    length = cal_length(kps, skl_list)
    return torch.tensor(girth_cp, device = v.device), length

def save_obj(obj_path, v, f, partcolor_list = None, vert_part_index = None, skl_list = None, kps = None):
    '''
    obj_path: str save_path
    v: array [N_v, 3]
    f: array [N_f, 3]
    partcolor_list: list [N_color]
    vert_part_index: array [N_v]
    skl_list: list [N_part]
    kps: array [N_kps, 3]
    '''
    num = 1000
    with open(obj_path, 'w') as fp:
        v_i = 0
        for tmp_v in v:
            if partcolor_list == None and vert_part_index == None:
                color = [192,192,192]
            else:
                color = partcolor_list[int(vert_part_index[v_i])]
            fp.write('v %f %f %f %d %d %d\n' % (tmp_v[0], tmp_v[1], tmp_v[2], color[0], color[1],color[2]))
            v_i = v_i + 1
        if skl_list != None and kps != None:
            for kps_index in skl_list:
                kps_points = (kps[kps_index[1], :] - kps[kps_index[0], :])[:, None] * np.linspace(0, 0.99, num)[None] + np.tile(kps[kps_index[0], :][:, None], (1, num))
                for tmp_v in kps_points.T:
                    color = [0, 0, 0]
                    fp.write('v %f %f %f %d %d %d\n' % (tmp_v[0], tmp_v[1], tmp_v[2], color[0], color[1],color[2]))
        if skl_list == None and kps != None:
                for tmp_v in kps:
                    color = [0, 0, 0]
                    fp.write('v %f %f %f %d %d %d\n' % (tmp_v[0], tmp_v[1], tmp_v[2], color[0], color[1],color[2]))
        for tmp_f in f + 1:
            fp.write('f %d %d %d\n' % (tmp_f[0], tmp_f[1], tmp_f[2]))


def save_skl(obj_path, kps_tmp, skl_list):
    
    tmp_skl_list = cfg.CONSTANTS.newskl_list # measure_skl_list newskl_list
    kps_keep = list(range(len(tmp_skl_list) + 4))
    for i in [3,13,14]:
        kps_keep.remove(i)
    if kps_tmp.shape[0] == len(tmp_skl_list) + 4:
        kps = copy.deepcopy(kps_tmp)
    else:
        kps = torch.zeros((len(tmp_skl_list) + 4, 3), device = kps_tmp.device)
        kps[kps_keep, :] = copy.deepcopy(kps_tmp)

    num = 100
    scale = 0.01
    with open(obj_path, 'w') as fp:
        for kps_index in skl_list:
            if len(kps_index) == 2:
                kps_points = (kps[kps_index[1], :] - kps[kps_index[0], :])[:, None] * np.linspace(0, 1, num)[None] + np.tile(kps[kps_index[0], :][:, None], (1, num))
            elif len(kps_index) == 3:
                kps_points = ((kps[kps_index[1], :] + kps[kps_index[2], :]) / 2 - kps[kps_index[0], :])[:, None] * np.linspace(0, 1, num)[None] + np.tile(kps[kps_index[0], :][:, None], (1, num))
            for tmp_v in kps_points.T:
                color = [0, 0, 0]
                fp.write('v %f %f %f %d %d %d\n' % (tmp_v[0], tmp_v[1], tmp_v[2], color[0], color[1],color[2]))
            for tmp_v in kps[kps_keep]:
                color = [0, 0, 0]
                for i in range(num):
                    fp.write('v %f %f %f %d %d %d\n' % (tmp_v[0] + (np.random.rand(1) - 0.5) * scale, tmp_v[1] + (np.random.rand(1) - 0.5) * scale, tmp_v[2] + (np.random.rand(1) - 0.5) * scale, color[0], color[1],color[2]))

def init_model(config_path):
    '''
    config_path: str 
    return: model
    '''
    update_cfg(config_path)
    J_regressor = np.load(cfg.PATH.J_regressor, allow_pickle=True)
    vert_part_index_dict = np.load(cfg.PATH.vert_part_index_dict, allow_pickle=True).item()
    partname_list = list(vert_part_index_dict.keys())
    torch.cuda.get_device_name(cfg.TRAIN.device_idx)
    downsample_method = 'COMA_downsample' # choose'COMA_downsample' or 'meshlab_downsample'

    # below are the arguments for the DFAUST run
    reference_mesh_file = os.path.join(cfg.PATH.root_dir, cfg.TRAIN.dataset, 'template', 'template.obj')
    downsample_directory = os.path.join(cfg.PATH.root_dir, cfg.TRAIN.dataset,'template', downsample_method)

    reference_points = [[414]]  # [[3567,4051,4597]] used for COMA with 3 disconnected components

    if not os.path.exists(downsample_directory):
        os.makedirs(downsample_directory)

    data = os.path.join(cfg.PATH.root_dir, cfg.TRAIN.dataset, 'preprocessed')

    if cfg.CONSTANTS.seed != None:
        random.seed(cfg.CONSTANTS.seed)
        np.random.seed(cfg.CONSTANTS.seed)
        torch.manual_seed(cfg.CONSTANTS.seed)
        torch.cuda.manual_seed(cfg.CONSTANTS.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print("Loading data .. ")

    shapedata =  ShapeData(nVal=cfg.TRAIN.Val_num, 
                            train_file=os.path.join(data, 'train.npy'), 
                            test_file=os.path.join(data, 'test.npy'), 
                            reference_mesh_file=reference_mesh_file,
                            normalization = 'No',
                            meshpackage = cfg.TRAIN.meshpackage)

    if not os.path.exists(os.path.join(downsample_directory,'downsampling_matrices{}{}{}{}.pkl'.format(cfg.MODEL.ds_factors[0], cfg.MODEL.ds_factors[1], cfg.MODEL.ds_factors[2], cfg.MODEL.ds_factors[3]))):
        if shapedata.meshpackage == 'trimesh':
            raise NotImplementedError('Rerun with mpi-mesh as meshpackage')
        print("Generating Transform Matrices ..")
        if downsample_method == 'COMA_downsample':
            M,A,D,U,F = mesh_sampling.generate_transform_matrices(shapedata.reference_mesh, cfg.MODEL.ds_factors)
        with open(os.path.join(downsample_directory,'downsampling_matrices{}{}{}{}.pkl'.format(cfg.MODEL.ds_factors[0], cfg.MODEL.ds_factors[1], cfg.MODEL.ds_factors[2], cfg.MODEL.ds_factors[3])), 'wb') as fp:
            M_verts_faces = [(M[i].v, M[i].f) for i in range(len(M))]
            pickle.dump({'M_verts_faces':M_verts_faces,'A':A,'D':D,'U':U,'F':F}, fp)
    else:
        print("Loading Transform Matrices ..")
        with open(os.path.join(downsample_directory,'downsampling_matrices{}{}{}{}.pkl'.format(cfg.MODEL.ds_factors[0], cfg.MODEL.ds_factors[1], cfg.MODEL.ds_factors[2], cfg.MODEL.ds_factors[3])), 'rb') as fp:
            #downsampling_matrices = pickle.load(fp,encoding = 'latin1')
            downsampling_matrices = pickle.load(fp)
                
        M_verts_faces = downsampling_matrices['M_verts_faces']
        if shapedata.meshpackage == 'mpi-mesh':
            M = [Mesh(v=M_verts_faces[i][0], f=M_verts_faces[i][1]) for i in range(len(M_verts_faces))]
        elif shapedata.meshpackage == 'trimesh':
            M = [trimesh.base.Trimesh(vertices=M_verts_faces[i][0], faces=M_verts_faces[i][1], process = False) for i in range(len(M_verts_faces))]
        A = downsampling_matrices['A']
        D = downsampling_matrices['D']
        U = downsampling_matrices['U']
        F = downsampling_matrices['F']

    for i in range(len(D)):
        if i == 0:
            D_ = D[i].todense()
        else:
            D_ = np.matmul(D[i].todense(), D_)
    a,b = D_.nonzero()
    downsamplevert_part_index_dict = {}
    for k in vert_part_index_dict.keys():
        downsamplevert_part_index_dict[k] = np.where(vert_part_index_dict[k] == b[:,None])[0]

    vert_part_index = np.ones(M_verts_faces[0][0].shape[0])
    for k,v in enumerate(vert_part_index_dict.values()):
        vert_part_index[v] = k



    print('partname_list of dict:', partname_list)
    print('The number of vertices in each part of the body:', [len(i) for i in list(vert_part_index_dict.values())])
    print('The number of downsample vertices in each part of the body:', [len(i) for i in list(downsamplevert_part_index_dict.values())])
    print('partname_list of cfg:', cfg.CONSTANTS.part_list)
    print('skl_list of cfg:', cfg.CONSTANTS.skl_list)



    # Needs also an extra check to enforce points to belong to different disconnected component at each hierarchy level
    print("Calculating reference points for downsampled versions..")
    for i in range(len(cfg.MODEL.ds_factors)):
        if shapedata.meshpackage == 'mpi-mesh':
            dist = euclidean_distances(M[i+1].v, M[0].v[reference_points[0]])
        elif shapedata.meshpackage == 'trimesh':
            dist = euclidean_distances(M[i+1].vertices, M[0].vertices[reference_points[0]])
        reference_points.append(np.argmin(dist,axis=0).tolist())

    if shapedata.meshpackage == 'mpi-mesh':
        sizes = [x.v.shape[0] for x in M]
    elif shapedata.meshpackage == 'trimesh':
        sizes = [x.vertices.shape[0] for x in M]
    Adj, Trigs = get_adj_trigs(A, F, shapedata.reference_mesh, meshpackage = shapedata.meshpackage)

    spirals_np, spiral_sizes,spirals = generate_spirals(cfg.MODEL.step_sizes, 
                                                        M, Adj, Trigs, 
                                                        reference_points = reference_points, 
                                                        dilation = cfg.MODEL.dilation, random = False, 
                                                        meshpackage = shapedata.meshpackage, 
                                                        counter_clockwise = True)
    # exit(101)
    bU = []
    bD = []
    for i in range(len(D)):
        d = np.zeros((1,D[i].shape[0]+1,D[i].shape[1]+1))
        u = np.zeros((1,U[i].shape[0]+1,U[i].shape[1]+1))
        d[0,:-1,:-1] = D[i].todense()
        u[0,:-1,:-1] = U[i].todense()
        d[0,-1,-1] = 1
        u[0,-1,-1] = 1
        bD.append(d)
        bU.append(u)

    torch.manual_seed(cfg.CONSTANTS.seed)

    if cfg.TRAIN.GPU:
        device = torch.device("cuda:"+str(cfg.TRAIN.device_idx) if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    print(device)

    tspirals = [torch.from_numpy(s).long().to(device) for s in spirals_np]
    tD = [torch.from_numpy(s).float().to(device) for s in bD]
    tU = [torch.from_numpy(s).float().to(device) for s in bU]
    if cfg.TRAIN.model_type == 'multiz+partkps':
        model = SpiralAutoencoder_multiz_partkps(kps_index_list = cfg.CONSTANTS.kps_index_list, vert_part_index_dict=downsamplevert_part_index_dict, 
        filters_enc = cfg.MODEL.filter_sizes_enc,   
                            filters_dec = cfg.MODEL.filter_sizes_dec,
                            latent_size = cfg.MODEL.part_shape_latent_size,
                            part_kps_latent_size=cfg.MODEL.part_kps_latent_size,
                            sizes=sizes,
                            spiral_sizes=spiral_sizes,
                            spirals=tspirals,
                            D=tD, U=tU,device=device).to(device)
        print('--------------------------init_multiz+partkps--------------------------')
    elif cfg.TRAIN.model_type == 'neural3DMM':
        model = SpiralAutoencoder( filters_enc = cfg.MODEL.filter_sizes_enc,   
                            filters_dec = cfg.MODEL.filter_sizes_dec,
                            latent_size = cfg.MODEL.nz,
                            sizes=sizes,
                            spiral_sizes=spiral_sizes,
                            spirals=tspirals,
                            D=tD, U=tU,device=device).to(device)
        print('--------------------------init_neural3DMM--------------------------')
        
    optim = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.lr, weight_decay=cfg.TRAIN.regularization)
    if cfg.TRAIN.scheduler[0]:
        scheduler=torch.optim.lr_scheduler.StepLR(optim, cfg.TRAIN.scheduler[1], gamma= cfg.TRAIN.scheduler[2])
    else:
        scheduler = None

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # if hasattr(model, 'fc_latent_enc_list'):
    #     for m in model.fc_latent_enc_list:
    #         for p in m.parameters():
    #             if p.requires_grad:
    #                 params = params + p.numel()
    # if hasattr(model, 'fc_latent_dec_list'):
    #     for m in model.fc_latent_dec_list:
    #         for p in m.parameters():
    #             if p.requires_grad:
    #                 params = params + p.numel()   
    print("Total number of parameters is: {}".format(params)) 
    print(model)


        
    if cfg.TRAIN.resume[0]:
            print('loading checkpoint from file %s'%(cfg.TRAIN.resume[1]))
            checkpoint_dict = torch.load(cfg.TRAIN.resume[1],map_location=device)
            start_epoch = checkpoint_dict['epoch'] + 1
            model.load_state_dict(checkpoint_dict['autoencoder_state_dict'], strict = False)
            print('Resuming from epoch %s'%(str(start_epoch)))     
    else:
        start_epoch = 1

    return model

def edit_skl(kps, kps_index, edit_length):
    '''
    kps: tensor [N_num, N_kps, 3]
    kps_index: int
    edit_length: float
    return: new_kps tensor [N_num, N_kps, 3]
    '''
    parent_dict = {1: 0, 2: 0, 3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7, \
    11: 8, 12: 9, 13: 9, 14: 9, 15: 12, 16: 13, 17: 14, 18: 16, 19: 17, 20: 18, 21: 19, 22: 20, 23: 21}
    child_dict = {0: [1, 2, 3], 1: [4], 2: [5], 3: [6], 4: [7], 5: [8], 6: [9], 7: [10],  \
        8: [11], 9: [12, 13, 14], 12: [15], 13: [16], 14: [17], 16: [18], 17: [19], 18: [20], 19: [21], 20: [22], 21: [23]}
    parent_kps = kps[:, parent_dict[kps_index], :]
    edit_kps = kps[:, kps_index, :]
    new_kps = copy.deepcopy(kps)
    dirc_vec = (edit_kps - parent_kps) 
    edit_length = (edit_length - 1)[:, None]
    # new_kps[kps_index, :] = parent_kps + dirc_vec * edit_length
    edit_list = []
    def dfs(index):
        edit_list.append(index)
        if index in list(child_dict.keys()):
            for child_index in child_dict[index]: 
                dfs(child_index)
        else:
            return 0
    dfs(kps_index)
    for i in edit_list:
        new_kps[:, i, :] = new_kps[:, i, :] + (dirc_vec * edit_length)
    return new_kps

def angle_skl(v, kps, partname_list, vert_part_index_dict, skl_list):    
    # v_direct = v[:, :, None].repeat(1,1,v.shape[1],1) - v[:, None].repeat(1,v.shape[1],1,1)
    euc_list = []
    for part_index in range(len(partname_list)):
        v_index = vert_part_index_dict[partname_list[part_index]]
        # v_direct_part = v_direct[:,v_index[:, None], v_index, :]
        v_direct_part = v[:, v_index, :][:, :, None].repeat(1,1,len(v_index),1) - v[:, v_index, :][:, None].repeat(1,len(v_index),1,1)
        if len(skl_list[part_index]) == 2:
            kps_direct = (kps[:,skl_list[part_index][0],:] - kps[:,skl_list[part_index][1], :])[:,None, None].repeat(1,v_index.shape[0],v_index.shape[0],1)
        elif len(skl_list[part_index]) == 3:
                kps_direct = (kps[:,skl_list[part_index][0],:] - (kps[:,skl_list[part_index][1], :] + kps[:,skl_list[part_index][2], :]) / 2)[:,None, None].repeat(1,v_index.shape[0],v_index.shape[0],1)
        v_direct_m = torch.sqrt(torch.sum(torch.mul(v_direct_part, v_direct_part), dim = -1, keepdim=True))
        # print(v_direct_m.shape)
        # for idx in range(v.shape[0]):
        #     v_direct_m[idx, :, :, 0] = v_direct_m[idx, :, :, 0] - torch.diag_embed(torch.diag(v_direct_m[idx, :, :, 0]))
        kps_direct_m = torch.sqrt(torch.sum(torch.mul(kps_direct, kps_direct), dim = -1, keepdim=True))
        dot = torch.sum(torch.mul(v_direct_part, kps_direct), dim = -1, keepdim=True)
        cos_ = torch.abs(dot / (v_direct_m*kps_direct_m))

        newcos = torch.where(torch.isnan(cos_), torch.full_like(cos_, 1), cos_).float()
        newcos = torch.where(newcos > 1, torch.full_like(newcos, 1), newcos)
        newcos = torch.where(newcos < 0, torch.full_like(newcos, 0), newcos)
        if torch_version == '1.5.0':
            newarccos = torch.acos(newcos)*180/math.pi
        else:
            newarccos = torch.arccos(newcos)*180/math.pi
        
        if not torch.all(torch.isfinite(newarccos)):
            print('inf', partname_list[part_index])
            print('newcos', newcos[torch.where(torch.isfinite(newarccos) == False)])
            print('newarccos', newarccos[torch.where(torch.isfinite(newarccos) == False)])
        if torch.any(torch.isnan(newarccos)):
            print('nan', partname_list[part_index])
            print('newcos', newcos[torch.where(torch.isnan(newarccos) == True)])
            print('newarccos', newarccos[torch.where(torch.isnan(newarccos) == True)])
        euc_list.append(newarccos)
    return euc_list
    
def euc_dist(tx, rec, vert_part_index_dict, partname_list, angle_w, w_mode, w_threshold, print_flag = False):
    point_num = tx.shape[1]
    De = calc_euclidean_dist_matrix(tx)
    De_r = calc_euclidean_dist_matrix(rec) 
    euc_loss = torch.zeros(len(partname_list), device = tx.device)
    euc_1 = torch.zeros(len(partname_list), device = tx.device)
    euc_2 = torch.zeros(len(partname_list), device = tx.device)
    for i in range(len(partname_list)):
        if w_mode == 'all_one':
            w = torch.ones_like(angle_w[i].squeeze(-1), device = tx.device)
        elif w_mode == 'linear':
            w = (angle_w[i].squeeze(-1).to(tx.device).float()) / 90
        elif w_mode == 'sin':
            w = torch.sin(angle_w[i].squeeze(-1).float() / 180 * math.pi).to(tx.device)
        elif w_mode == 'threshold':
            w = (angle_w[i].squeeze(-1).to(tx.device).float()) / 90
            w = torch.where(w < w_threshold, torch.full_like(w, 0), w) 
        tmp_index_part = vert_part_index_dict[partname_list[i]]
        nozero_index = torch.where(w != 0) 
        euc_1[i] = torch.mean(w[nozero_index]*De[:, tmp_index_part[:, None], tmp_index_part][nozero_index].float())
        euc_2[i] = torch.mean(w[nozero_index]*De_r[:, tmp_index_part[:, None], tmp_index_part][nozero_index].float())
        euc_loss[i] = Function.l1_loss(w[nozero_index]*De_r[:, tmp_index_part[:, None], tmp_index_part][nozero_index].float(), w[nozero_index]*De[:, tmp_index_part[:, None], tmp_index_part][nozero_index])  
    if print_flag:
        print('euc_1:', euc_1)
        print('euc_2:', euc_2)
    return euc_loss

def write_txt(txt_path, str):
    with open(txt_path, "a+") as f: 
        f.write(str + '\n')


