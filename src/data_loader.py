import numpy as np
import cv2
import os
import random
import pandas as pd

class ImageDataLoader():
    def __init__(self, data_path, gt_path, shuffle=False, gt_downsample=False, pre_load=False, 
                 batch_size=1, scaling=4, re_scale=1.0, re_size=None):
        #pre_load: if true, all training and validation images are loaded into CPU RAM for faster processing.
        #          This avoids frequent file reads. Use this only for small datasets.
        self.data_path = data_path
        self.gt_path = gt_path
        self.gt_downsample = gt_downsample
        self.pre_load = pre_load
        self.data_files = [filename for filename in os.listdir(data_path) \
                           if os.path.isfile(os.path.join(data_path,filename))]
        self.data_files.sort()
        self.shuffle = shuffle
        self.scaling = scaling
        self.re_scale = re_scale
        self.re_size = re_size
        if shuffle:
            random.seed(2468)
        self.num_samples = len(self.data_files)
        self.blob_list = {}        
        self.id_list = range(0,self.num_samples/batch_size)
        
        batch = -1
        batch_full=False
        if self.pre_load:
            print 'Pre-loading the data. This may take a while...'
            idx = 0
            for fname in self.data_files:
                
                img = cv2.imread(os.path.join(self.data_path,fname),0)
                img = img.astype(np.float32, copy=False)
                if self.re_size is None:
                    ht = img.shape[0]
                    wd = img.shape[1]
                else:
                    ht = self.re_size[0]
                    wd = self.re_size[1]
                ht_1 = (ht/self.scaling)*self.scaling
                wd_1 = (wd/self.scaling)*self.scaling
                img = cv2.resize(img,(wd_1,ht_1))
                img = img.reshape((1,1,img.shape[0],img.shape[1]))
                img = img/self.re_scale
                den = pd.read_csv(os.path.join(self.gt_path,os.path.splitext(fname)[0] + '.csv'), sep=',',header=None).as_matrix()                        
                den  = den.astype(np.float32, copy=False)
                if self.gt_downsample:
                    wd_1 = wd_1/self.scaling
                    ht_1 = ht_1/self.scaling
                    den = cv2.resize(den,(wd_1,ht_1))                
                    den = den * ((wd*ht)/(wd_1*ht_1))
                else:
                    den = cv2.resize(den,(wd_1,ht_1))
                    den = den * ((wd*ht)/(wd_1*ht_1))
                
                den = den.reshape((1,1,den.shape[0],den.shape[1]))
                if idx==0:
                    blob = {}
                    blob['data']=img
                    blob['gt_density']=den
                    blob['fname'] = [fname]
                    idx+=1
                    batch_full=False
                    if idx==batch_size:
                        idx = 0
                        batch_full=True
                else:
                    blob['data']=np.concatenate((blob['data'],img))
                    blob['gt_density']=np.concatenate((blob['gt_density'],den))
                    blob['fname'].append(fname)
                    idx+=1
                    batch_full=False
                    if idx==batch_size:
                        idx = 0
                        batch_full=True
                
                if batch_full:
                    batch+=1
                    self.blob_list[batch] = blob
                    if batch % 200 == 0:                    
                        print 'Loaded', batch, 'batch', batch*batch_size, '/', self.num_samples, 'files'
               
            print 'Completed Loading ', batch+1, 'batches'
        
        
    def __iter__(self):
        if self.shuffle:            
            if self.pre_load:            
                random.shuffle(self.id_list)  
            else:
                random.shuffle(self.data_files)
        files = self.data_files
        id_list = self.id_list
       
        for idx in id_list:
            if self.pre_load:
                blob = self.blob_list[idx]    
                blob['idx'] = idx
            else:                    
                fname = files[idx]
                img = cv2.imread(os.path.join(self.data_path,fname),0)
                img = img.astype(np.float32, copy=False)
                if self.re_size is None:
                    ht = img.shape[0]
                    wd = img.shape[1]
                else:
                    ht = self.re_size[0]
                    wd = self.re_size[1]
                ht_1 = (ht/self.scaling)*self.scaling
                wd_1 = (wd/self.scaling)*self.scaling
                img = cv2.resize(img,(wd_1,ht_1))
                img = img.reshape((1,1,img.shape[0],img.shape[1]))
                img = img/self.re_scale
                den = pd.read_csv(os.path.join(self.gt_path,os.path.splitext(fname)[0] + '.csv'), sep=',',header=None).as_matrix()                        
                den  = den.astype(np.float32, copy=False)
                if self.gt_downsample:
                    wd_1 = wd_1/self.scaling
                    ht_1 = ht_1/self.scaling
                    den = cv2.resize(den,(wd_1,ht_1))                
                    den = den * ((wd*ht)/(wd_1*ht_1))
                else:
                    den = cv2.resize(den,(wd_1,ht_1))
                    den = den * ((wd*ht)/(wd_1*ht_1))
                    
                den = den.reshape((1,1,den.shape[0],den.shape[1]))            
                blob = {}
                blob['data']=img
                blob['gt_density']=den
                blob['fname'] = fname
                
            yield blob
            
    def get_num_samples(self):
        return self.num_samples
                
        
            
        
