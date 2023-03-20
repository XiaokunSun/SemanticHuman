import numpy as np
import os

import pickle
from psbody.mesh import Mesh
import mesh_sampling
import trimesh
from shape_data import ShapeData
from autoencoder_dataset import autoencoder_dataset
from torch.utils.data import DataLoader
from utils_spiral import get_adj_trigs, generate_spirals
from models import  SpiralAutoencoder, SpiralAutoencoder_multiz_partkps
from train_funcs import train_autoencoder_dataloader_nonormal, train_autoencoder_dataloader
from test_funcs import test_autoencoder_dataloader_nonormal, test_autoencoder_dataloader
from configure.cfgs import cfg, update_cfg
import torch
from tensorboardX import SummaryWriter
from sklearn.metrics.pairwise import euclidean_distances
import random
import torch.nn.functional as Function
import sys

# Load the configuration file and initialize
config_path = os.path.join(cfg.PATH.root_dir, 'configure', 'traincfg.yaml')
update_cfg(config_path)

J_regressor = np.load(cfg.PATH.J_regressor, allow_pickle=True)
vert_part_index_dict = np.load(cfg.PATH.vert_part_index_dict, allow_pickle=True).item()
partname_list = list(vert_part_index_dict.keys())
print('train ' + cfg.MODEL.model_name)
torch.cuda.get_device_name(cfg.TRAIN.device_idx)


generative_model = 'autoencoder'
downsample_method = 'COMA_downsample' 


if cfg.TRAIN.model_type == 'multiz+partkps':
    dir_name = 'multiz+partkps'
    dummy_flag = True
elif cfg.TRAIN.model_type == 'neural3DMM':
    dir_name = 'neural3DMM'  
    dummy_flag = True



reference_mesh_file = os.path.join(cfg.PATH.root_dir, cfg.TRAIN.dataset, 'template', 'template.obj')
downsample_directory = os.path.join(cfg.PATH.root_dir, cfg.TRAIN.dataset,'template', downsample_method)

reference_points = [[414]]  

results_folder = os.path.join(cfg.PATH.root_dir, cfg.TRAIN.dataset, 'results', dir_name, cfg.MODEL.model_name)  
if not os.path.exists(os.path.join(results_folder)):
    os.makedirs(os.path.join(results_folder))

summary_path = os.path.join(results_folder,'summaries')
if not os.path.exists(summary_path):
    os.makedirs(summary_path)  
    
checkpoint_path = os.path.join(results_folder,'checkpoints')
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)
    
samples_path = os.path.join(results_folder,'samples')
if not os.path.exists(samples_path):
    os.makedirs(samples_path)
    
prediction_path = os.path.join(results_folder,'predictions')
if not os.path.exists(prediction_path):
    os.makedirs(prediction_path)

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

# Initialize spiral convolution
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

with open('{}/color.obj'.format(prediction_path), 'w') as fp:
    v_i = 0
    for v in M_verts_faces[0][0]:
        color = cfg.CONSTANTS.partcolor_list[int(vert_part_index[v_i])]
        fp.write('v %f %f %f %d %d %d\n' % (v[0], v[1], v[2], color[0], color[1],color[2]))
        v_i = v_i + 1
    for f in M_verts_faces[0][1] + 1:
        fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

print('partname_list of dict:', partname_list)
print('The number of vertices in each part of the body:', [len(i) for i in list(vert_part_index_dict.values())])
print('The number of downsample vertices in each part of the body:', [len(i) for i in list(downsamplevert_part_index_dict.values())])
print('partname_list of cfg:', cfg.CONSTANTS.part_list)
print('skl_list of cfg:', cfg.CONSTANTS.skl_list)

downsamplevert_part_index = np.ones(a.shape[0])
for k,v in enumerate(downsamplevert_part_index_dict.values()):
    downsamplevert_part_index[v] = k
downsamplepartname_list = list(downsamplevert_part_index_dict.keys())
with open('{}/downsamplecolor.obj'.format(prediction_path), 'w') as fp:
    v_i = 0
    for v in M_verts_faces[-1][0]:
        color = cfg.CONSTANTS.partcolor_list[int(downsamplevert_part_index[v_i])]
        fp.write('v %f %f %f %d %d %d\n' % (v[0], v[1], v[2], color[0], color[1],color[2]))
        v_i = v_i + 1
    for f in M_verts_faces[-1][1] + 1:
        fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))


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

# Initialize training datasets

dataset_train = autoencoder_dataset(root_dir = data, points_dataset = 'train',
                                           shapedata = shapedata,
                                           normalization = cfg.TRAIN.normal_flag, dummy_node = dummy_flag, measure_flag = cfg.TRAIN.measure_flag, J_regressor = J_regressor)

dataloader_train = DataLoader(dataset_train, batch_size=cfg.TRAIN.batchsize_train,\
                                     shuffle = cfg.TRAIN.shuffle, num_workers = cfg.TRAIN.num_workers)

dataset_val = autoencoder_dataset(root_dir = data, points_dataset = 'val', 
                                         shapedata = shapedata,
                                         normalization = cfg.TRAIN.normal_flag, dummy_node = dummy_flag, J_regressor = J_regressor)

dataloader_val = DataLoader(dataset_val, batch_size=cfg.TRAIN.batchsize_test,\
                                     shuffle = False, num_workers = cfg.TRAIN.num_workers)


dataset_interp = autoencoder_dataset(root_dir = data, points_dataset = 'train',
                                        shapedata = shapedata,
                                        normalization = cfg.TRAIN.normal_flag, dummy_node = dummy_flag, 
                                         measure_flag = cfg.TRAIN.measure_flag, J_regressor = J_regressor)

dataloader_interp = DataLoader(dataset_interp, batch_size=cfg.TRAIN.batchsize_interp,\
                                    shuffle = cfg.TRAIN.shuffle, num_workers = cfg.TRAIN.num_workers)

dataset_test = autoencoder_dataset(root_dir = data, points_dataset = 'test',
                                          shapedata = shapedata,
                                          normalization = cfg.TRAIN.normal_flag, dummy_node = dummy_flag, J_regressor = J_regressor)

dataloader_test = DataLoader(dataset_test, batch_size=cfg.TRAIN.batchsize_test,\
                                     shuffle = False, num_workers = cfg.TRAIN.num_workers)

# Initialize the network

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
print("Total number of parameters is: {}".format(params)) 
print(model)

writer = SummaryWriter(summary_path)
with open(os.path.join(results_folder, 'checkpoints', 'train_params.txt'), 'w') as f:
    f.write('------------------------------config------------------------------\n')
    print(cfg, file = f)
    
if cfg.TRAIN.resume[0]:
        print('loading checkpoint from file %s'%(cfg.TRAIN.resume[1]))
        if cfg.TRAIN.resume[2]:
            checkpoint_dict = torch.load(cfg.TRAIN.resume[1],map_location=device)
            start_epoch = 1
            model.load_state_dict(checkpoint_dict['autoencoder_state_dict'])
            print('finetune from epoch %s'%(str(start_epoch)))     
        else:
            checkpoint_dict = torch.load(cfg.TRAIN.resume[1],map_location=device)
            start_epoch = checkpoint_dict['epoch'] + 1
            model.load_state_dict(checkpoint_dict['autoencoder_state_dict'])
            optim.load_state_dict(checkpoint_dict['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint_dict['scheduler_state_dict'])
            print('Resuming from epoch %s'%(str(start_epoch)))     
else:
    start_epoch = 1

if cfg.TRAIN.model_type == 'multiz+partkps':
    train_autoencoder_dataloader_nonormal(dataloader_train, dataloader_test,
                        device, model, optim, Function.l1_loss,
                        start_epoch = start_epoch,
                        n_epochs = cfg.TRAIN.n_epochs,
                        eval_freq = cfg.TRAIN.eval_frequency,
                        dataloader_interp = dataloader_interp,
                        scheduler = scheduler,
                        writer = writer,
                        shapedata=shapedata,
                        metadata_dir=checkpoint_path, samples_dir=samples_path,
                        checkpoint_path = cfg.TRAIN.ck_name, J_regressor = J_regressor,
                        vert_part_index_dict = vert_part_index_dict, partname_list = partname_list,
                        save_recons=True
                        )
else:
    train_autoencoder_dataloader(dataloader_train, dataloader_test,
                        device, model, optim, Function.l1_loss,
                        start_epoch = start_epoch,
                        n_epochs = cfg.TRAIN.n_epochs,
                        eval_freq = cfg.TRAIN.eval_frequency,
                        dataloader_interp = dataloader_interp,
                        scheduler = scheduler,
                        writer = writer,
                        shapedata=shapedata,
                        metadata_dir=checkpoint_path, samples_dir=samples_path,
                        checkpoint_path = cfg.TRAIN.ck_name, J_regressor = J_regressor,
                        vert_part_index_dict = vert_part_index_dict, partname_list = partname_list,
                        save_recons=True
                        )

if cfg.TRAIN.eval_flag:
    checkpoint_dict = torch.load(os.path.join(checkpoint_path, cfg.TRAIN.ck_name+'%s.pth.tar'%(cfg.TRAIN.n_epochs)),map_location=device)
    model.load_state_dict(checkpoint_dict['autoencoder_state_dict'])
    
    if cfg.TRAIN.model_type == 'multiz+partkps':
        predictions, z_s, z_kps_s, tx_s, norm_l1_loss, l2_loss = test_autoencoder_dataloader_nonormal(device, model, dataloader_test, 
                                                                    shapedata, J_regressor, mm_constant = 1000, unnormal_flag = cfg.TRAIN.normal_flag)  
        np.save(os.path.join(prediction_path,'predictions'), predictions)   
        np.save(os.path.join(prediction_path,'z_s'), z_s) 
        np.save(os.path.join(prediction_path,'z_kps_s'), z_kps_s)    
        np.save(os.path.join(prediction_path,'tx_s'), tx_s) 
        print('autoencoder: L1 loss', norm_l1_loss)
        print('autoencoder: euclidean distance in mm=', l2_loss)
        with open(os.path.join(results_folder, 'checkpoints', 'train_params.txt'), 'a') as f:
            f.write(f'autoencoder: L1 loss {norm_l1_loss}')
            f.write('\n')
            f.write(f'autoencoder: euclidean distance in mm {l2_loss}')
    else:
        predictions, z_s, tx_s, norm_l1_loss, l2_loss = test_autoencoder_dataloader(device, model, dataloader_test, 
                                                                    shapedata, J_regressor, mm_constant = 1000, unnormal_flag = cfg.TRAIN.normal_flag)  
        np.save(os.path.join(prediction_path,'predictions'), predictions)   
        np.save(os.path.join(prediction_path,'z_s'), z_s) 
        np.save(os.path.join(prediction_path,'tx_s'), tx_s) 
        print('autoencoder: L1 loss', norm_l1_loss)
        print('autoencoder: euclidean distance in mm=', l2_loss)
        with open(os.path.join(results_folder, 'checkpoints', 'train_params.txt'), 'a') as f:
            f.write(f'autoencoder: L1 loss {norm_l1_loss}' )
            f.write('\n')
            f.write(f'autoencoder: euclidean distance in mm {l2_loss}')