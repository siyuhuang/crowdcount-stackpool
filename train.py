import os
import torch
import numpy as np
import sys

from src.crowd_count import CrowdCounter
from src import network
from src.data_loader import ImageDataLoader
from src.timer import Timer
from src import utils
from src.evaluate_model import evaluate_model
import time
       
np.warnings.filterwarnings('ignore')
### assign dataset, model, and pooling method    
datasets = ['shtechA', 'shtechB']     # datasets
models = ['base', 'wide', 'deep']     # backbone network architecture
pools = ['vpool','stackpool','mpool']    #  vpool is vanilla pooling; stackpool is stacked pooling; mpool is multi-kernel pooling; 

dataset_name = datasets[0]   # choose the dataset
model = models[0]          # choose the backbone network architecture
pool = pools[0]          # choose the pooling method 
method=model+'_'+pool
print 'Training %s on %s' % (method, dataset_name)

### assign GPU 
if pool=='vpool':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"   
if pool=='stackpool':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"   
if pool=='mpool':
    os.environ["CUDA_VISIBLE_DEVICES"] = "2" 
    
### PyTorch configuration
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

### model saving folder
output_dir = './saved_models/'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

### data folder
name = dataset_name[-1]
train_path = './data/formatted_trainval/shanghaitech_part_'+name+'_patches_9/train'
train_gt_path = './data/formatted_trainval/shanghaitech_part_'+name+'_patches_9/train_den'
val_path = './data/formatted_trainval/shanghaitech_part_'+name+'_patches_9/val'
val_gt_path = './data/formatted_trainval/shanghaitech_part_'+name+'_patches_9/val_den'

### training configuration
start_step = 0
end_step = 500
batch_size=1
disp_interval = 1500
if model=='base':
    if dataset_name == 'shtechA':
        lr = 2*1e-5
    if dataset_name == 'shtechB':
        lr = 1e-5
    scaling=4   # output density map is 1/4 size of input image
if model=='wide':
    if dataset_name == 'shtechA':
        lr = 1e-5
    if dataset_name == 'shtechB':
        lr = 1e-5
    scaling=4  # output density map is 1/4 size of input image
if model=='deep':
    if dataset_name == 'shtechA':
        lr = 1e-5
    if dataset_name == 'shtechB':
        lr = 5*1e-6
    scaling=8  # output density map is 1/8 size of input image     
print 'learning rate %f' % (lr)

### random seed
rand_seed = 64678  
if rand_seed is not None: 
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed_all(rand_seed)

### initialize network
net = CrowdCounter(model=model,pool=pool)
network.weights_normal_init(net, dev=0.01)
net.cuda()
net.train()

### optimizer
params = list(net.parameters())
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)

### load data
pre_load=True   
data_loader = ImageDataLoader(train_path, train_gt_path, shuffle=True, gt_downsample=True, pre_load=pre_load, 
                              batch_size=batch_size,scaling=scaling)
data_loader_val = ImageDataLoader(val_path, val_gt_path, shuffle=False, gt_downsample=True, pre_load=pre_load, 
                              batch_size=1,scaling=scaling)

### training
train_loss = 0
t = Timer()
t.tic()
best_mae = sys.maxint

for epoch in range(start_step, end_step+1):    
    step = 0
    train_loss = 0
    for blob in data_loader:                
        step = step + 1    
        im_data = blob['data']
        gt_data = blob['gt_density']
        density_map = net(im_data, gt_data)
        loss = net.loss
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    

        if step % disp_interval == 0: 
            duration = t.toc(average=False)
            density_map = density_map.data.cpu().numpy()
            utils.save_results(im_data,gt_data,density_map, output_dir)
            print 'epoch: %4d, step %4d, Time: %.4fs, loss: %4.10f' % (epoch, step, duration, train_loss/disp_interval)
            train_loss = 0
            t.tic()

    if (epoch % 2 == 0):
        # save model checkpoint
        save_name = os.path.join(output_dir, '{}_{}_{}.h5'.format(method,dataset_name,epoch))
        network.save_net(save_name, net)     
        # calculate error on the validation dataset 
        mae,mse = evaluate_model(save_name, data_loader_val, model, pool)
        if mae < best_mae:
            best_mae = mae
            best_mse = mse
            best_model = '{}_{}_{}.h5'.format(method,dataset_name,epoch)
        print 'EPOCH: %d, MAE: %0.2f, MSE: %0.2f' % (epoch,mae,mse)
        print 'BEST MAE: %0.2f, BEST MSE: %0.2f, BEST MODEL: %s' % (best_mae,best_mse, best_model)

        
    t.tic()
        
    

