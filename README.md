# Stacked Pooling: Improving Crowd Counting by Boosting Scale Invariance

This is the implementation of paper "Stacked Pooling: Improving Crowd Counting by Boosting Scale Invariance"

<p align="center">
   <img src="https://github.com/siyuhuang/crowdcount-stackpool/blob/master/thumbnails/stackpool.jpg" width="400">
</p>

This code is implemented based on [https://github.com/svishwa/crowdcount-mcnn](https://github.com/svishwa/crowdcount-mcnn)

# Dependency
1. Python 2.7
2. PyTorch-0.4.0

# Data Setup
1. Download ShanghaiTech Dataset from   
   Dropbox:   https://www.dropbox.com/s/fipgjqxl7uj8hd5/ShanghaiTech.zip?dl=0
   
   Baidu Disk: http://pan.baidu.com/s/1nuAYslz
2. Create Directory 
   ```bash
   mkdir ./data/original/shanghaitech/  
   ```
3. Save "part_A_final" under ./data/original/shanghaitech/
4. Save "part_B_final" under ./data/original/shanghaitech/
5. `cd ./data_preparation/`

   run `create_gt_test_set_shtech.m` in matlab to create ground truth files for test data
6. `cd ./data_preparation/`

   run `create_training_set_shtech.m` in matlab to create training and validataion set along with ground truth files

# Train
1. To train Base-M Net+vanilla pooling on ShanghaiTechA, edit `train.py` 
   ```bash
   dataset_name = datasets[0]   
   model = models[0]         
   pool = pools[0] 
   ```
   
   To train Base-M Net+stacked pooling on ShanghaiTechA, edit `train.py`
   ```bash
   dataset_name = datasets[0]   
   model = models[0]         
   pool = pools[1] 
   ```
   
2. Run `python train.py` to start training

# Test
1. To test **Base-M Net**+**vanilla pooling** on **ShanghaiTechA**, edit `train.py` 
   ```bash
   dataset_name = datasets[0]   
   model = models[0]         
   pool = pools[0] 
   ```
   
   To test **Base-M Net**+**stacked pooling** on **ShanghaiTechA**, edit `train.py`
   ```bash
   dataset_name = datasets[0]   
   model = models[0]         
   pool = pools[1] 
   ```
2. Edit `model_path` in `test.py` with the best model checkpoint on validation set (output by training process)  
3. Run `python test.py` to compare them!

# Note
1. To compare the pooling methods (**vanilla pooling**, **stacked pooling**, and **multi-kernel pooling**) described in our paper:

     Edit `pool` in `train.py` and `test.py`

2. To try datasets (ShanghaiTechA, ShanghaiTechB) and backbone models (**Base-M Net**, **Wide-Net**, **Deep-Net**) described in our paper:

     Edit `dataset_name` and `model` in `train.py` and `test.py`



