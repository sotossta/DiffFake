import os 
import cv2
import argparse
from utils.funcs import extract_bounding_box, extract_facial_landmarks
import numpy as np
from utils.get_splits import get_dirs_FNet

def save_bb_and_fl(args,face_detector,landmark_detector):

    if args.dataset=="FF_original":
        root_dir = os.path.join(args.data_dir_path,"FF++","original_sequences",args.compression,"images")
        save_dir_bb = os.path.join(args.data_dir_path,"FF++","original_sequences",args.compression,"bounding_boxes")
        save_dir_fl = os.path.join(args.data_dir_path,"FF++","original_sequences",args.compression,"facial_landmarks")   
    elif args.dataset.split("-")[0] =="Celeb":
        root_dir = os.path.join(args.data_dir_path,"Celeb-DF",args.dataset,"images")
        save_dir_bb = os.path.join(args.data_dir_path,"Celeb-DF",args.dataset,"bounding_boxes")
        save_dir_fl = os.path.join(args.data_dir_path,"Celeb-DF",args.dataset,"facial_landmarks")
    elif args.dataset =="YouTube-real":
        root_dir = os.path.join(args.data_dir_path,"Celeb-DF",args.dataset,"images")
        save_dir_bb = os.path.join(args.data_dir_path,"Celeb-DF",args.dataset,"bounding_boxes")
        save_dir_fl = os.path.join(args.data_dir_path,"Celeb-DF",args.dataset,"facial_landmarks")
    elif args.dataset=="DeeperForensics":
        root_dir = os.path.join(args.data_dir_path,args.dataset,"manipulated_sequences","images")
        save_dir_bb = os.path.join(args.data_dir_path,args.dataset,"manipulated_sequences","bounding_boxes")
        save_dir_fl = os.path.join(args.data_dir_path,args.dataset,"manipulated_sequences","facial_landmarks")  
    elif args.dataset=="ForgeryNet":
        root_dir = os.path.join(args.data_dir_path,args.dataset,"images")
        save_dir_bb = os.path.join(args.data_dir_path,args.dataset,"bounding_boxes")
        save_dir_fl = os.path.join(args.data_dir_path,args.dataset,"facial_landmarks")
         
    else:
        root_dir = os.path.join(args.data_dir_path,"FF++","manipulated_sequences",args.dataset,args.compression,"images")
        save_dir_bb = os.path.join(args.data_dir_path,"FF++","manipulated_sequences",args.dataset,args.compression,"bounding_boxes")
        save_dir_fl = os.path.join(args.data_dir_path,"FF++","manipulated_sequences",args.dataset,args.compression, "facial_landmarks")
    
    if args.dataset.split("_")[0]=="ForgeryNet":
        image_folders_real_dir = get_dirs_FNet(os.path.join(args.data_dir_path,"ForgeryNet/splits/test_real.txt"))
        image_folders_fake_dir = get_dirs_FNet(os.path.join(args.data_dir_path,"ForgeryNet/splits/test_fake.txt"))
        image_folders_dir = image_folders_real_dir + image_folders_fake_dir
        image_folders_dir = [element.replace("images/", "") for element in image_folders_dir]
    else:
        image_folders_dir = sorted(os.listdir(root_dir))

    all_folders = len(image_folders_dir)
    folder_iter=0
    for folder_dir in image_folders_dir:

        
        
        print("Progress: {}/{} Current folder {}".format(folder_iter,all_folders,folder_dir),flush=True)
        folder_iter+=1
        images_dir = os.listdir(os.path.join(root_dir,folder_dir))   
        for image_dir in images_dir:
            
            img = cv2.imread(os.path.join(root_dir,folder_dir,image_dir))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            bounding_box = extract_bounding_box(face_detector, img)
            if len(bounding_box)>1:
                #If multiple faces are detected choose the biggest bounding box
                sums = bounding_box[:,-2] + bounding_box[:,-1]
                max_sum_index = np.argmax(sums)
                bounding_box_biggest = np.array([bounding_box[max_sum_index]])
            else:
                bounding_box_biggest = bounding_box
            landmarks = extract_facial_landmarks(landmark_detector,img,bounding_box_biggest)
            os.makedirs(os.path.join(save_dir_bb,folder_dir), exist_ok=True)
            os.makedirs(os.path.join(save_dir_fl,folder_dir), exist_ok=True)
            np.save(os.path.join(save_dir_bb,folder_dir,image_dir.split(".")[0] + ".npy"),arr=bounding_box_biggest[0])
            np.save(os.path.join(save_dir_fl,folder_dir,image_dir.split(".")[0] + ".npy"),arr=landmarks)
    

if __name__ == '__main__':

    p = argparse.ArgumentParser(description="Extracting bounding boxes and facial landmarks.")
    p.add_argument("--data_dir_path",type=str,default="/Data")
    p.add_argument("--dataset",choices=["FF_original","Deepfakes","Face2Face", "FaceSwap", "FaceShifter", "NeuralTextures",
                                        "Celeb-real","Celeb-synthesis","YouTube-real","DeeperForensics",
                                        "ForgeryNet"],
                            type=str,default="FF_original")
    p.add_argument("--compression",choices=["c0","c23","c40"],type=str,default="c0")    
    args = p.parse_args()
    #Import face detector
    haarcascade = "face_detectors/haarcascade_frontalface_alt2.xml"
    face_detector = cv2.CascadeClassifier(haarcascade)
    #Import face landmark detector
    LBFmodel = "face_detectors/lbfmodel.yaml"
    landmark_detector  = cv2.face.createFacemarkLBF()
    landmark_detector.loadModel(LBFmodel)
    save_bb_and_fl(args, face_detector, landmark_detector)





        
        
        