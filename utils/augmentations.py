#Adapted from https://github.com/mapooon/SelfBlendedImages
import albumentations as alb
import numpy as np
import cv2

class RandomDownScale(alb.core.transforms_interface.ImageOnlyTransform):
	def apply(self,img,**params):
     
		return self.randomdownscale(img)

	def randomdownscale(self,img):
		keep_ratio=True
		keep_input_shape=True
		H,W,C=img.shape
		ratio_list=[2,4]
		r=ratio_list[np.random.randint(len(ratio_list))]
       
		img_ds=cv2.resize(img,(int(W/r),int(H/r)),interpolation=cv2.INTER_NEAREST)
		if keep_input_shape:
			img_ds=cv2.resize(img_ds,(W,H),interpolation=cv2.INTER_LINEAR)
            
		return img_ds

def source_transforms(img):
    
    st = alb.Compose([
    				alb.Compose([
    				alb.RGBShift((-20,20),(-20,20),(-20,20),p=0.3),
    				alb.HueSaturationValue(hue_shift_limit=(-0.3,0.3), sat_shift_limit=(-0.3,0.3), val_shift_limit=(-0.3,0.3), p=1),
    				alb.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1,0.1), p=1),
    					],p=1),
    	
    				alb.OneOf([
    					RandomDownScale(p=1),
    					alb.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1),
    				],p=1),			
    	        ], p=1.)
    transformed=st(image=img)
    img=transformed['image']
    return img
        

def randaffine(img,mask):
        
    f = alb.Affine(translate_percent={'x':(-0.03,0.03),'y':(-0.015,0.015)},
    				scale=[0.95,1/0.95],
    				fit_output=False,
    				p=1)
    transformed=f(image=img,mask=mask)
    img=transformed['image']
    mask=transformed['mask']
    return img,mask
    
def elastic_def(mask):
    
    g=alb.ElasticTransform(
    			alpha=50,
    			sigma=7,
    			alpha_affine=0,
    			p=1,
    		)
    transformed=g(image=mask)
    mask=transformed['image']
    return mask

def jpeg_compression(img, param):
    h, w, _ = img.shape
    s_h = h // param
    s_w = w // param
    img = cv2.resize(img, (s_w, s_h))
    img = cv2.resize(img, (w, h))

    return img