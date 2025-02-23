import cv2
import random
from utils.DeepFakeMask import dfl_full,facehull,components,extended
import numpy as np


def extract_bounding_box(detector,img):
    
    image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(image_gray)
    if len(faces)!=0:
        return faces
    else:
        return np.array([[]])
        

def extract_facial_landmarks(detector,img,bounding_box):
    
   
    image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if bounding_box.size!=0:
        _, landmarks = detector.fit(image_gray, bounding_box)
        return landmarks[0][0]
    else:
        return np.array([])


def crop_image(bounding_box,img,landmarks):
    
    x1, y1, width, height =  bounding_box
    # Calculate center of the detected face
    center_x = x1 + width // 2
    center_y = y1 + height // 2
    # Calculate new bounding box coordinates with conservative crop (enlarged by a factor of 1.3)
    new_width = int(width * 1.3)
    new_height = int(height * 1.3)
    new_x1 = max(0, center_x - new_width // 2)
    new_y1 = max(0, center_y - new_height // 2)
    new_x2 = min(img.shape[1], new_x1 + new_width)
    new_y2 = min(img.shape[0], new_y1 + new_height)
    # Adjust landmarks
    adjusted_landmarks = [(x - new_x1, y - new_y1) for (x, y) in landmarks]
    
    return img[new_y1:new_y2, new_x1:new_x2], np.array([adjusted_landmarks])[0]

def random_get_hull(landmark,img):
    
    hull_type =random.choice([0,1,2,3])
    
    if hull_type == 0:
        mask = dfl_full(landmarks=landmark.astype('int32'),face=img,channels=3).mask
        return mask/255
    elif hull_type == 1:
        mask = extended(landmarks=landmark.astype('int32'),face=img, channels=3).mask
        return mask/255
    elif hull_type == 2:
        mask = components(landmarks=landmark.astype('int32'),face=img, channels=3).mask
        return mask/255
    elif hull_type == 3:
        mask = facehull(landmarks=landmark.astype('int32'),face=img, channels=3).mask
        return mask/255
    
def blur_mask(mask):
    
    size_h = np.random.randint(192, 257)
    size_w = np.random.randint(192, 257)
    kernel_1 = random.randrange(5, 26, 2)
    kernel_1 = (kernel_1, kernel_1)
    kernel_2 = random.randrange(5, 26, 2)
    kernel_2 = (kernel_2, kernel_2)
    random_int = np.random.randint(5, 46)
    if len(mask)==2:
        
        mask1 = mask[0]
        mask2 = mask[1]
        H1, W1 = mask1.shape
        H2, W2 = mask2.shape
        mask1 = cv2.resize(mask1, (size_w, size_h))
        mask2 = cv2.resize(mask2, (size_w, size_h))
        mask1_blured = cv2.GaussianBlur(mask1, kernel_1, 0)
        mask1_blured = mask1_blured / (mask1_blured.max())
        mask1_blured[mask1_blured < 1] = 0
        mask1_blured = cv2.GaussianBlur(mask1_blured, kernel_2, random_int)
        mask1_blured = mask1_blured / (mask1_blured.max())
        mask1_blured = cv2.resize(mask1_blured, (W1, H1))
        mask2_blured = cv2.GaussianBlur(mask2, kernel_2, 0)
        mask2_blured = mask2_blured / (mask2_blured.max())
        mask2_blured[mask2_blured < 1] = 0
        mask2_blured = cv2.GaussianBlur(mask2_blured, kernel_2, random_int)
        mask2_blured = mask2_blured / (mask2_blured.max())
        mask2_blured = cv2.resize(mask2_blured, (W2, H2))
        
        return mask1_blured.reshape((mask1_blured.shape + (1,))), mask2_blured.reshape((mask2_blured.shape + (1,)))
        
    else:
        H, W = mask.shape
        mask = cv2.resize(mask, (size_w, size_h))
        mask_blured = cv2.GaussianBlur(mask, kernel_1, 0)
        mask_blured = mask_blured / (mask_blured.max())
        mask_blured[mask_blured < 1] = 0
        mask_blured = cv2.GaussianBlur(mask_blured, kernel_2, random_int)
        mask_blured = mask_blured / (mask_blured.max())
        mask_blured = cv2.resize(mask_blured, (W, H))
    
        return mask_blured.reshape((mask_blured.shape + (1,)))

