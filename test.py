import os
import torch
import numpy as np
np.set_printoptions(threshold=np.nan)

from src.crowd_count import CrowdCounter
from src import network
from src.data_loader import ImageDataLoader
from src import utils

os.environ["CUDA_VISIBLE_DEVICES"] = "0"   
np.warnings.filterwarnings('ignore')
# dataset, model, and pooling method    
datasets = ['shtechA', 'shtechB']     # datasets
models = ['base', 'wide', 'deep']     # backbone network architecture
pools = ['vpool','stackpool','mpool']    #  vpool is vanilla pooling; stackpool is stacked pooling; mpool is multi-kernel pooling

###
dataset_name = datasets[0]   # choose the dataset
model = models[2]          # choose the backbone network architecture
pool = pools[0]          # choose the pooling method 
method=model+'_'+pool

name = dataset_name[-1]
data_path =  './data/original/shanghaitech/part_'+name+'_final/test_data/images/'
gt_path = './data/original/shanghaitech/part_'+name+'_final/test_data/ground_truth_csv/'
model_path = './saved_models/'+method+'_shtech'+name+'_0.h5' 
print 'Testing %s' % (model_path)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
vis = False
save_output = True    

output_dir = './output/'
model_name = os.path.basename(model_path).split('.')[0]
file_results = os.path.join(output_dir,'results_' + model_name + '_.txt')
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
output_dir = os.path.join(output_dir, 'density_maps_' + model_name)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

net = CrowdCounter(model,pool)      
trained_model = os.path.join(model_path)
network.load_net(trained_model, net)
net.cuda()
net.eval()

if model in ['base','wide']:
    scaling = 4
if model=='deep':
    scaling = 8

#load test data
data_loader = ImageDataLoader(data_path, gt_path, shuffle=False, gt_downsample=True, pre_load=False, batch_size=1, scaling=scaling)

mae = 0.0
mse = 0.0
num = 0
for blob in data_loader:  
    num+=1
    im_data = blob['data']
    gt_data = blob['gt_density']
    density_map = net(im_data)
    density_map = density_map.data.cpu().numpy()
    gt_count = np.sum(gt_data)
    et_count = np.sum(density_map)
    mae += abs(gt_count-et_count)
    mse += ((gt_count-et_count)*(gt_count-et_count))    
    if vis:
        utils.display_results(im_data, gt_data, density_map)
    if save_output:
        utils.save_density_map(density_map, output_dir, 'output_' + blob['fname'].split('.')[0] + '.png')
    if num%100==0:
        print '%d/%d' % (num,data_loader.get_num_samples())

mae = mae/data_loader.get_num_samples()
mse = np.sqrt(mse/data_loader.get_num_samples())
print 'MAE: %0.2f, MSE: %0.2f' % (mae,mse)

f = open(file_results, 'w') 
f.write('MAE: %0.2f, MSE: %0.2f' % (mae,mse))
f.close()
