import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import cv2
import torch
from construct_pseudo_deepfake import make_pseudo_deepfake
import random

class Pseudo_Deepfake_Dataset(Dataset):
    
    def __init__(self,root_dir, transform=None,split_file=None,frames_sampled=40):
        
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []  # List to store paths of all images
        self.labels = []  # List to store corresponding folder names 
        folders=split_file   
        # Iterate through all folders and collect image paths
        for folder_name in folders:
            folder_path = os.path.join(self.root_dir, folder_name)
            images = [os.path.join(folder_name, f) for f in sorted(os.listdir(folder_path)) if os.path.isfile(os.path.join(folder_path, f))]
            if len(images) > frames_sampled:
                images = random.sample(images, frames_sampled)
            for image in images:
                bounding_box_path = os.path.join(self.root_dir, image).replace("images","bounding_boxes").replace("png", "npy")
                if os.path.exists(bounding_box_path):
                    bounding_box = np.load(bounding_box_path)
                    if bounding_box.size != 0:
                        self.image_paths.append(image)
                 
          
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        
        image_path = os.path.join(self.root_dir,self.image_paths[idx])
        bounding_box_path = os.path.join(self.root_dir,self.image_paths[idx]).replace("images","bounding_boxes").replace("png","npy")
        facial_landmarks_path = os.path.join(self.root_dir,self.image_paths[idx]).replace("images","facial_landmarks").replace("png","npy")
        #Import image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #Import bounding box and facial landmarks
        bounding_box = np.load(bounding_box_path)
        facial_landmarks = np.load(facial_landmarks_path)

        _, img_s, img_t, _, _, blended_img, pseudo_deepfake_type = make_pseudo_deepfake(image,bounding_box,facial_landmarks)
        if self.transform:
            img_t = self.transform(Image.fromarray(np.uint8(img_t)).convert('RGB'))
            blended_img = self.transform(Image.fromarray(np.uint8(blended_img)).convert('RGB'))
        
        return img_t, blended_img,pseudo_deepfake_type
    
    def collate_fn(self,batch):
         
         img_t,img_b,pseudo_deepfake_type=zip(*batch)
         img_t = torch.stack(img_t)
         img_b = torch.stack(img_b)
         data={}
         data['img'] = torch.cat((img_b, img_t), dim=0)
         data['label']=torch.tensor([0]*len(img_b)+[1]*len(img_t))
         return data
