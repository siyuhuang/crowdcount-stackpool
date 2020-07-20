## Stacked Pooling for Boosting Scale Invariance of Crowd Counting

PyTorch implementation of  "**Stacked Pooling for Boosting Scale Invariance of Crowd Counting**" [\[ICASSP 2020\]](https://siyuhuang.github.io/papers/ICASSP-2020-STACKED%20POOLING%20FOR%20BOOSTING%20SCALE%20INVARIANCE%20OF%20CROWD%20COUNTING.pdf). 

```
@inproceedings{huang2020stacked,
  title={Stacked Pooling for Boosting Scale Invariance of Crowd Counting},
  author={Huang, Siyu and Li, Xi and Cheng, Zhi-Qi and Zhang, Zhongfei and Hauptmann, Alexander},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing},
  pages={2578--2582},
  year={2020},
}
```

This code is implemented based on [https://github.com/svishwa/crowdcount-mcnn](https://github.com/svishwa/crowdcount-mcnn)

<p align="left">
   <img src="https://github.com/siyuhuang/crowdcount-stackpool/blob/master/thumbnails/stackpool.jpg" width="400">
</p>

| | ShanghaiTech-A    |  ShanghaiTech-B  | WorldExpo'10|
| --------   | :-----:   | :----: | :----: |
| Vanilla Pooling | 97.63      |   21.17    | 14.74 |
| Stacked Pooling | **93.98**  |  **18.73** |  **12.92**|


## Dependency
1. Python 2.7
2. PyTorch 0.4.0

## Data Setup
1. Download ShanghaiTech Dataset from   
     Dropbox:   https://www.dropbox.com/s/fipgjqxl7uj8hd5/ShanghaiTech.zip?dl=0  
     Baidu Disk: http://pan.baidu.com/s/1nuAYslz
2. Create Directory `mkdir ./data/original/shanghaitech/`
3. Save "part_A_final" under ./data/original/shanghaitech/  
   Save "part_B_final" under ./data/original/shanghaitech/
4. `cd ./data_preparation/`  
   Run `create_gt_test_set_shtech.m` in matlab to create ground truth files for test data     
   Run `create_training_set_shtech.m` in matlab to create training and validataion set along with ground truth files
   
## Train
1. To train **Deep Net**+**vanilla pooling** on **ShanghaiTechA**, edit configurations in `train.py` 
   ```bash       
   pool = pools[0] 
   ```
   
   To train **Deep Net**+**stacked pooling** on **ShanghaiTechA**, edit configurations in `train.py`
   ```bash     
   pool = pools[1] 
   ```   
2. Run `python train.py` respectively to start training

## Test
1. Follow step 1 of **Train** to edit corresponding `pool` in `test.py`
2. Edit `model_path` in `test.py` using the best checkpoint on validation set (output by training process)  
3. Run `python test.py` respectively to compare them!

## Note
1. To try pooling methods (**vanilla pooling**, **stacked pooling**, and **multi-kernel pooling**) described in our paper:

     Edit `pool` in `train.py` and `test.py`

2. To evaluate on datasets (**ShanghaiTechA**, **ShanghaiTechB**) or backbone models (**Base Net**, **Wide-Net**, **Deep-Net**) described in our paper:

     Edit `dataset_name` or `model` in `train.py` and `test.py`



