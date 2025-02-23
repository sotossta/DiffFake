import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import cv2
from utils.funcs import crop_image
import random
class Differential_AD_dataset(Dataset):
    
    def __init__(self, root_dir, images_to_pair ,transform=None, split_file=None,frames_sampled=40):
        
        self.root_dir = root_dir
        self.transform = transform
        self.image_pairs = [] # List used to store paths to paired images
        if split_file is not None:
            folders = split_file
        else:
            folders = sorted(os.listdir(os.path.join(self.root_dir,"images")))    
        for folder_name in folders:    
            images_to_remove = []
            images = os.listdir(os.path.join(self.root_dir,folder_name))
            images = list(map(lambda img: os.path.join(folder_name, img), images))
            if frames_sampled<len(images):
                images = random.sample(images, frames_sampled)
            
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
           
            
            if len(images)>2*images_to_pair:
                base_images = random.sample(images, len(images)-images_to_pair)
            else:
                base_images = random.sample(images, len(images)//2)

            pair_images = [item for item in images if item not in base_images]
            random.shuffle(pair_images)
            #pairs = [(base, pair) for base in base_images for pair in pair_images]
            pairs = [(item, random.choice(pair_images)) for item in base_images]
            self.image_pairs.extend(pairs)
    def __len__(self):
        return len(self.image_pairs)
    
    def __getitem__(self, idx):
        
        
        image1_path = os.path.join(self.root_dir,self.image_pairs[idx][0])
        image2_path = os.path.join(self.root_dir,self.image_pairs[idx][1])
        bounding_box1_path = os.path.join(self.root_dir,self.image_pairs[idx][0]).replace("images","bounding_boxes").replace("png","npy")
        bounding_box2_path = os.path.join(self.root_dir,self.image_pairs[idx][1]).replace("images","bounding_boxes").replace("png","npy")
        facial_landmarks1_path = os.path.join(self.root_dir,self.image_pairs[idx][0]).replace("images","facial_landmarks").replace("png","npy")
        facial_landmarks2_path = os.path.join(self.root_dir,self.image_pairs[idx][1]).replace("images","facial_landmarks").replace("png","npy")
        

        #Import images
        image1 = cv2.imread(image1_path)
        image2 = cv2.imread(image2_path)
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        #Import bounding box and facial landmarks
        bounding_box1 = np.load(bounding_box1_path)
        facial_landmarks1 = np.load(facial_landmarks1_path)
        bounding_box2 = np.load(bounding_box2_path)
        facial_landmarks2 = np.load(facial_landmarks2_path)
        
        
        if bounding_box1.size != 0:
            img1_cropped,landmarks1_cropped = crop_image(bounding_box1,image1,facial_landmarks1)
        else:
            img1_cropped=image1
        if bounding_box2.size != 0:
            img2_cropped,landmarks2_cropped = crop_image(bounding_box2,image2,facial_landmarks2)
        else:
            img2_cropped=image2
        if self.transform:
            img1_cropped = self.transform(Image.fromarray(np.uint8(img1_cropped)).convert('RGB'))
            img2_cropped = self.transform(Image.fromarray(np.uint8(img2_cropped)).convert('RGB'))
            
        video_id = image1_path.split("/")[-2]
        return img1_cropped,img2_cropped,video_id
        
 
            
            