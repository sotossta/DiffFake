import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import cv2
from utils.funcs import crop_image
class AD_dataset(Dataset):
    
    def __init__(self, root_dir,transform=None, split_file=False,frames_sampled=32):
        
        self.root_dir = root_dir
        self.transform = transform
        self.images = [] # List used to store paths to images
        if split_file is not None:
            folders = split_file
        else:
            folders = sorted(os.listdir(os.path.join(self.root_dir,"images")))      
        for folder_name in folders:
            
            images_to_remove = []
            images = os.listdir(os.path.join(self.root_dir, folder_name))
            images = list(map(lambda img: os.path.join(folder_name, img), images))
            
            for image in images:
                bounding_box_path = os.path.join(self.root_dir, image).replace("images","bounding_boxes").replace("png", "npy")
                if os.path.exists(bounding_box_path):
                    bounding_box = np.load(bounding_box_path)
                    if bounding_box.size == 0:
                        #If bounding box of face has not been detected remove image from data
                        images_to_remove.append(image)
                else:
                    images_to_remove.append(image)

            for element in images_to_remove:
                images.remove(element)
           
            
            self.images.extend(images)
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        
        
        image_path = os.path.join(self.root_dir,self.images[idx])
        bounding_box_path = os.path.join(self.root_dir,self.images[idx]).replace("images","bounding_boxes").replace("png","npy")
        facial_landmarks_path = os.path.join(self.root_dir,self.images[idx]).replace("images","facial_landmarks").replace("png","npy")
        #Import images
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #Import bounding box and facial landmarks
        bounding_box = np.load(bounding_box_path)
        facial_landmarks = np.load(facial_landmarks_path)
        if bounding_box.size != 0:
            img_cropped,landmarks_cropped = crop_image(bounding_box,image,facial_landmarks)
        else:
            img_cropped=image
        if self.transform:
            img_cropped = self.transform(Image.fromarray(np.uint8(img_cropped)).convert('RGB'))
            
        video_id = image_path.split("/")[-2]
        return img_cropped,video_id
        
 
            
            