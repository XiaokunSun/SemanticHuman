from tqdm import tqdm
import numpy as np
import os, argparse

# example: python data_generation.py -r /home/sxk/16T/3D_human_representation -d DFAUST_female --train_measure /home/sxk/16T/data/DFAUST/DFAUST_female/train_measurements.npy --test_measure /home/sxk/16T/data/DFAUST/DFAUST_female/test_measurements.npy -v 0
# Generate training data

parser = argparse.ArgumentParser(description='Arguments for dataset split')
parser.add_argument('-r','--root_dir', type=str,
            help='Root data directory location, should be same as in neural3dmm.ipynb') 
parser.add_argument('-d','--dataset', type=str, 
            help='Dataset name, Default is DFAUST')
parser.add_argument('--train_measure', type=str, 
            help='Dataset name, Default is DFAUST', default = None)
parser.add_argument('--test_measure', type=str, 
            help='Dataset name, Default is DFAUST', default = None)
parser.add_argument('-v','--num_valid', type=int, default=0, 
            help='Number of meshes in validation set, default 100')

args = parser.parse_args()


nVal = args.num_valid
root_dir = args.root_dir
dataset = args.dataset
name = ''

data = os.path.join(root_dir, dataset, 'preprocessed',name)
train = np.load(data+'/train.npy')
train_measure = np.load(os.path.join(args.train_measure), allow_pickle=True)
test_measure = np.load(os.path.join(args.test_measure), allow_pickle=True)
if not os.path.exists(os.path.join(data,'points_train')):
    os.makedirs(os.path.join(data,'points_train'))

if not os.path.exists(os.path.join(data,'points_val')):
    os.makedirs(os.path.join(data,'points_val'))

if not os.path.exists(os.path.join(data,'points_test')):
    os.makedirs(os.path.join(data,'points_test'))

if args.train_measure != None:
    if not os.path.exists(os.path.join(data,'measure_train')):
        os.makedirs(os.path.join(data,'measure_train'))

    if not os.path.exists(os.path.join(data,'measure_val')):
        os.makedirs(os.path.join(data,'measure_val'))

    if not os.path.exists(os.path.join(data,'measure_test')):
        os.makedirs(os.path.join(data,'measure_test'))

for i in tqdm(range(len(train)-nVal)):
    np.save(os.path.join(data,'points_train','{0}.npy'.format(str(i).zfill(6))),train[i])
    if args.train_measure != None:
        np.save(os.path.join(data,'measure_train','{0}.npy'.format(str(i).zfill(6))),train_measure[i])
for i in tqdm(range(len(train)-nVal,len(train))):
    np.save(os.path.join(data,'points_val','{0}.npy'.format(str(i).zfill(6))),train[i])
    if args.train_measure != None:
        np.save(os.path.join(data,'measure_val','{0}.npy'.format(str(i).zfill(6))),train_measure[i])
    
test = np.load(data+'/test.npy')
for i in tqdm(range(len(test))):
    np.save(os.path.join(data,'points_test','{0}.npy'.format(str(i).zfill(6))),test[i])
    if args.test_measure != None:
        np.save(os.path.join(data,'measure_test','{0}.npy'.format(str(i).zfill(6))),test_measure[i])
    
files = []
for f in sorted(os.listdir(os.path.join(data,'points_train'))):
    if '.npy' in f:
        files.append(os.path.splitext(f)[0])
np.save(os.path.join(data,'paths_train.npy'),files)

files = []
for f in sorted(os.listdir(os.path.join(data,'points_val'))):
    if '.npy' in f:
        files.append(os.path.splitext(f)[0])
np.save(os.path.join(data,'paths_val.npy'),files)

files = []
for f in sorted(os.listdir(os.path.join(data,'points_test'))):
    if '.npy' in f:
        files.append(os.path.splitext(f)[0])
np.save(os.path.join(data,'paths_test.npy'),files)

