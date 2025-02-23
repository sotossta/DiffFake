import numpy as np
import cv2
from utils.funcs import  crop_image, random_get_hull, blur_mask
from utils.augmentations import source_transforms, randaffine, elastic_def
import random


def blend_images(img_s, img_t, mask):
    
    #Randomly choose blend ratio
    alpha_list=[0.25,0.5,0.75,1,1,1]
    alpha= np.random.choice(alpha_list,size=1)[0]
    # Ensure images and mask are of the same size
    img_s = cv2.resize(img_s, (img_t.shape[1], img_t.shape[0]))
    mask = cv2.resize(mask, (img_t.shape[1], img_t.shape[0]))
    mask = alpha*mask 
    # Blend images using the given formula: ISB = Is ⊙ M + It ⊙ (1 − M)
    blended_img = img_s * mask[:, :, np.newaxis] + img_t * (1 - mask[:, :, np.newaxis])
    blended_img = blended_img.astype(np.uint8)
    return blended_img

def make_pseudo_deepfake(img,bounding_box,landmarks):
    
    #Crop Original image to bounding box
    img_cropped,landmarks_cropped = crop_image(bounding_box,img,landmarks)
    
    #----------------------- Source-target generator-----------------------------
    if np.random.random()<0.5:
        
        img_s = source_transforms(img_cropped)
        img_t = img_cropped
    else:
        img_s =img_cropped
        img_t = source_transforms(img_cropped)
    mask = np.zeros(img_cropped.shape[0:2] + (1, ), dtype=np.float32)
    pseudo_deepfake_type =random.choice([0,0,0,1,2,3])
    if pseudo_deepfake_type==0:
        #Full Face Pseudo-deepfake
        #Create mask after the landmarks transformation (used in Face X-ray)
        mask = random_get_hull(landmarks_cropped,img_cropped)[:,:,0]
    elif pseudo_deepfake_type==1:
        #Eyes only Pseudo-deepfake
        eyes_1 = landmarks_cropped[17:27]
        eyes_2 = landmarks_cropped[36:48]
        eyes = np.vstack((eyes_1,eyes_2,landmarks_cropped[0:2],landmarks_cropped[14:16]))
        mask = cv2.fillConvexPoly(mask, cv2.convexHull(eyes.astype(int)), 255.)[:,:,0]/255
    elif pseudo_deepfake_type==2:
        #Mouth only Pseudo-deepfake
        mouth = landmarks_cropped[48:68]
        lower_jaw = landmarks_cropped[4:13]
        nose_apex = landmarks_cropped[31:36]
        mouth_region = np.vstack((mouth,lower_jaw,nose_apex))
        mask = cv2.fillConvexPoly(mask, cv2.convexHull(mouth_region.astype(int)), 255.)[:,:,0]/255
    elif pseudo_deepfake_type==3:
        #Lower head Pseudo-deepfake
        jaw =  landmarks_cropped[1:16]
        lower_nose_ridge = landmarks_cropped[29:31]
        lower_head_region = np.vstack((jaw,lower_nose_ridge))
        mask = cv2.fillConvexPoly(mask, cv2.convexHull(lower_head_region.astype(int)), 255.)[:,:,0]/255
        
    #Resize and Translate source image and corresponding mask
    img_s,mask = randaffine(img_s,mask)
    #Deform the mask (elastic deformation)
    mask = elastic_def(mask)
    #Smooth mask using two Gaussian filters
    mask = blur_mask(mask)
    #Blend source and target images with the generated mask
    blended_img = blend_images(img_s, img_t, mask)
    return img_cropped, img_s, img_t, landmarks_cropped, mask, blended_img,pseudo_deepfake_type 

