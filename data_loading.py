import os
from skimage import io, transform, color,img_as_ubyte
import numpy as np
from torch.utils.data import Dataset
import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensor

#Dataset Loader
class multi_classes(Dataset):
        def __init__(self,path,data, transform=None):
            self.path = path
            self.folders = data
            self.transforms = transform
        
        def __len__(self):
            return len(self.folders)
              
        
        def __getitem__(self,idx):        
            image_path = os.path.join(self.path,'images/',self.folders[idx])
            mask_path = os.path.join(self.path,'masks/',self.folders[idx])    
            
            #img = io.imread(image_path)[:,:,:3].astype('float32')   
            img = color.gray2rgb(io.imread(image_path))[:, :, :3].astype('float32')
            mask = io.imread(mask_path)
            image_id = self.folders[idx]
            file_name, file_extension = os.path.splitext(image_id)
            
            augmented = self.transforms(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
            #mask_nl = np.squeeze(mask)
       
            unique_values = mask.view(-1).unique()
            new_mask = mask.clone()  # 创建一个新的 Tensor 来存储更新后的值

            for idx, value in enumerate(unique_values):
                new_mask[mask == value.item()] = idx       
            mask = new_mask
            mask = torch.squeeze(mask)
            mask = torch.nn.functional.one_hot(mask.to(torch.int64),4)
            mask = mask.permute(2, 0, 1)
            return (img,mask,file_name) 


class binary_class(Dataset):
        def __init__(self,path,data, transform=None):
            self.path = path
            self.folders = data
            self.transforms = transform
        
        def __len__(self):
            return len(self.folders)
              
        
        def __getitem__(self,idx):
            image_path = os.path.join(self.path,'images/',self.folders[idx])
            mask_path = os.path.join(self.path,'masks/',self.folders[idx])
            #img = io.imread(image_path)[:,:,:3].astype('float32') color.gray2rgb(image)
            img = color.gray2rgb(io.imread(image_path))[:, :, :3].astype('float32')
            mask = io.imread(mask_path)
            image_id = self.folders[idx]
            
            augmented = self.transforms(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
            return (img,mask,image_id)
        
        
