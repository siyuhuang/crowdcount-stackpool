# Stacked Pooling: Improving Crowd Counting by Boosting Scale Invariance

This is the implementation of paper "Stacked Pooling: Improving Crowd Counting by Boosting Scale Invariance.
<p align="center">
   <img src="https://github.com/siyuhuang/crowdcount-stackpool/blob/master/thumbnails/stackpool.jpg" width="450">
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

   run create_training_set_shtech.m in matlab to create training and validataion set along with ground truth files

# Train
Run
```bash
python train.py
```

# Test
1. Edit `model_path` in `test.py` with the model checkpoint which has the best MAE on validation set (output in training process).   
2. Run
   ```bash   
   python test.py 
   ```

# Note
1. To compare the pooling methods (vanilla pooling, stacked pooling, and multi-kernel pooling) described in our paper, 

edit `pool` in `train.py` and `test.py`
2. To try datasets (ShanghaiTechA, ShanghaiTechB) and backbone models (Base-M Net, Wide-Net, Deep-Net) described in our paper,

edit `dataset_name` and `model` in `train.py` and `test.py`



