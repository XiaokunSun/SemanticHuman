import os

from yacs.config import CfgNode as CN

cfg = CN(new_allowed=True)

cfg.MODEL = CN(new_allowed=True)
cfg.MODEL.model_name = 'SMPL_normal_multiz+kps128_l1+kps_2222'
cfg.MODEL.ds_factors = [2, 2, 2, 2]
cfg.MODEL.step_sizes = [2, 2, 1, 1, 1]
cfg.MODEL.filter_sizes_enc = [[3, 16, 32, 64, 128],[[],[],[],[],[]]]
cfg.MODEL.filter_sizes_dec = [[128, 64, 32, 32, 16],[[],[],[],[],3]]
cfg.MODEL.dilation = [2, 2, 1, 1, 1]  
cfg.MODEL.part_shape_latent_size = 8
cfg.MODEL.part_kps_latent_size = 8

cfg.CONSTANTS = CN(new_allowed=True)
cfg.CONSTANTS.skl_list = [[15,12],[15,12],[12,9],[6,0],[0,1,2],[1,4],\
[4,7],[7,10],[2,5],[5,8],[8,11],[16,18],\
[18,20],[20,22],[17,19],[19,21],[21,23]]
cfg.CONSTANTS.newskl_list = [[0,1],[0,2],[0,6],[1,4],[2,5],[6,9],[4,7],\
[5,8],[9,12],[9,16],[9,17],[7,10],[8,11],[12,15],[16,18],[17,19],\
[18,20],[19,21],[20,22],[21,23],[20,24],[21,25],[20,26],[21,27],[15,28],[15,29],[15,30]] # ,[7,31],[8,32],[7,33],[8,34]]
cfg.CONSTANTS.kps_index_list = [[15,28,29,30],[15,12],[12,9],[6,0],[0,1,2],[1,4],\
[4,7],[7,10,31,33],[2,5],[5,8],[8,11,32,34],[16,18],\
[18,20],[20,22,24,26],[17,19],[19,21],[21,23,25,27]]
cfg.CONSTANTS.noleaf_skl_list = [[15,12],[12,9],[6,0],[0,1,2],[1,4],\
[4,7],[2,5],[5,8],[16,18],[18,20],[17,19],[19,21]]
cfg.CONSTANTS.measure_skl_list = [[15,12],[12,9],[6,0],[0,1,2],[1,4],\
[4,7],[7,10],[2,5],[5,8],[8,11],[16,18],\
[18,20],[20,22],[17,19],[19,21],[21,23]]
cfg.CONSTANTS.skl_list_total = [[0, 2], [2, 5], [5, 8], [8, 11], \
    [0, 1], [1, 4], [4, 7], [7, 10], \
    [0, 3], [3, 6], [6, 9], [9, 12], [12, 15], \
    [9, 14], [14, 17], [17, 19], [19, 21], [21, 23], \
    [9, 13], [13, 16], [16, 18], [18, 20], [20, 22]]
cfg.CONSTANTS.part_list = ['head','neck','chest','abdomen','hip','left_ham','left_shank',\
    'left_feet','right_ham','right_shank','right_feet','left_arm','left_forearm','left_hand',\
        'right_arm','right_forearm','right_hand']
cfg.CONSTANTS.leaf_part_list = ['head','left_feet','right_feet','left_hand','right_hand']
cfg.CONSTANTS.noleaf_part_list = ['neck','chest','abdomen','hip','left_ham','left_shank','right_ham',\
    'right_shank','left_arm','left_forearm','right_arm','right_forearm']
cfg.CONSTANTS.measure_part_list = ['neck','chest','abdomen','hip','left_ham','left_shank',\
    'left_feet','right_ham','right_shank','right_feet','left_arm','left_forearm','left_hand',\
        'right_arm','right_forearm','right_hand']
cfg.CONSTANTS.seed = 2
cfg.CONSTANTS.partcolor_list = [[0, 191, 255], [255, 0, 191], [255, 0, 63], [0, 127, 255], \
[255, 0, 254], [0, 254, 255], [255, 0, 127], [255, 127, 0], [0, 0, 255], [255, 191, 0], \
[63, 0, 255], [191, 255, 0], [0, 255, 0], [0, 63, 255], [127, 255, 0], [127, 0, 255], \
[255, 63, 0], [191, 0, 255], [0, 255, 63], [254, 255, 0], [63, 255, 0], [255, 0, 0], \
[0, 255, 191], [0, 255, 127]]

cfg.PATH = CN(new_allowed=True)
cfg.PATH.root_dir = '/home/sxk/16T/SemanticHuman'
cfg.PATH.J_regressor = os.path.join(cfg.PATH.root_dir, 'asset', 'J_regressor.npy')
cfg.PATH.vert_part_index_dict = os.path.join(cfg.PATH.root_dir, 'asset', 'vert_part_index_dict.npy')
cfg.PATH.factor_list = os.path.join(cfg.PATH.root_dir, 'asset', 'factor_list.npy')
cfg.PATH.edge_point_index_list = os.path.join(cfg.PATH.root_dir, 'asset', 'edge_point_index_list.npy')
cfg.PATH.edge_verts_index = os.path.join(cfg.PATH.root_dir, 'asset', 'edge_verts_index.npy')


cfg.TRAIN = CN(new_allowed=True)
cfg.TRAIN.meshpackage = 'mpi-mesh'
cfg.TRAIN.n_epochs = 300
cfg.TRAIN.Val_num = 10
cfg.TRAIN.dataset = 'SMPL'
cfg.TRAIN.dataset_interp = 'SMPL'
cfg.TRAIN.ck_name = 'checkpoint'
cfg.TRAIN.batchsize_train = 16
cfg.TRAIN.batchsize_test = 16
cfg.TRAIN.batchsize_interp = 4
cfg.TRAIN.eval_frequency = 10
cfg.TRAIN.normal_flag = 'No'
cfg.TRAIN.model_type = ''
cfg.TRAIN.skl_mode = 'm'
cfg.TRAIN.exc_mode = 'm'
cfg.TRAIN.kpskeep_flag = True
cfg.TRAIN.sklkeep_flag = True
cfg.TRAIN.leafkeep_flag = True
cfg.TRAIN.editskl_flag = False
cfg.TRAIN.noleaf_flag = False
cfg.TRAIN.GPU = True
cfg.TRAIN.device_idx = 1
cfg.TRAIN.num_workers = 4
cfg.TRAIN.shuffle = True
cfg.TRAIN.measure_flag = True
cfg.TRAIN.eval_flag = True
cfg.TRAIN.relat_flag = True
cfg.TRAIN.lr = 1e-3
cfg.TRAIN.regularization = 5e-5
cfg.TRAIN.scheduler = [True, 1,0.99]
cfg.TRAIN.resume = [False, '', False]
cfg.TRAIN.w_mode = 'linear'
cfg.TRAIN.w_threshold = 0.8
cfg.TRAIN.w_part_mode = '1/K'
cfg.TRAIN.edit_mode = 'equal'
cfg.TRAIN.rand_mode = 'rand'
cfg.TRAIN.factor = [0.4, 0.8]
cfg.TRAIN.edgereg_epoch = 0
cfg.TRAIN.edgereg_w = 1e0
cfg.TRAIN.zpartreg_epoch = 0
cfg.TRAIN.zpartreg_w = 1e0
cfg.TRAIN.vol_epoch = 0
cfg.TRAIN.vol_w = 1e0
cfg.TRAIN.interp_epoch = 0
cfg.TRAIN.interp_kps_w = 1e0
cfg.TRAIN.interp_euc_w = 1e0
cfg.TRAIN.exc_epoch = 0
cfg.TRAIN.exc_kps_w = 1e0
cfg.TRAIN.exc_euc_w = 1e0
cfg.TRAIN.ck_frequency = 50




cfg.TEST = CN(new_allowed=True)
cfg.TEST.save_path = cfg.PATH.root_dir
cfg.TEST.resume = [False, '']


def update_cfg(cfg_file):
    cfg.merge_from_file(cfg_file)
    return cfg