# ------------------ Import libraries --------------------------------
import torch
import random
import argparse
import numpy as np
from prepare_data import get_dataloaders_differential_AD, get_dataloaders_AD
import os
from models.model import Detector
from models.resnet50 import ResNet50
from models.xception import Xception
from tqdm import tqdm
from sklearn.mixture import GaussianMixture as GMM
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import normalize
import pickle

def normalize_tensor(tensor):
    tensor_shape = tensor.size()
    flattened = tensor.view(tensor_shape[0], -1).detach().cpu().numpy()  # Flatten to 2D for sklearn
    normalized = normalize(flattened, norm='l2')  # Apply L2 normalization (or 'l1', 'max' as needed)
    return torch.tensor(normalized, dtype=tensor.dtype, device=tensor.device).view(tensor_shape)  # Reshape to original


def get_combined_embeds(args,loader,model,device):
    
    model.eval()
    combined_embeds = []
    with torch.no_grad():
        for batch_idx, (img1,img2,_) in enumerate(tqdm(loader)):
        
            x1 = img1.to(device)
            x2 = img2.to(device)
            y_1 = model(x1)
            y_2 = model(x2)
            if args.feature_combination==0:
                comb = torch.abs(y_1 - y_2) 
            elif args.feature_combination==1:
                comb = y_1-y_2
            elif args.feature_combination==2:
                comb = (y_1 - y_2) ** 2 
            elif args.feature_combination==3:
                comb = (y_1 - y_2) ** 3   
            combined_embeds.append(comb.detach().cpu().numpy().reshape(comb.shape[0],-1))
 
        combined_embeds = np.vstack(combined_embeds)
        return normalize(combined_embeds)

def get_embeds(args,loader,model,device):
    
    model.eval()
    embeds = []
    with torch.no_grad():
        for batch_idx, (img,_) in enumerate(tqdm(loader)):
        
            x = img.to(device)
            y = model(x)  
            embeds.append(y.detach().cpu().numpy().reshape(y.shape[0],-1))
        embeds = np.vstack(embeds)
        return normalize(embeds)

def train_AD_module(args,train_embeds,save_path = None, num_components=3):

    os.makedirs(save_path, exist_ok=True)
    if args.ADM =="GMM":
        """Fits a Gaussian Mixture Model"""
        gmm = GMM(
            n_components=num_components,
            verbose=1,
            verbose_interval=5,
            init_params="kmeans",
            max_iter=250,
        ).fit(train_embeds)
        print("Finished GMM fitting")

        if save_path:
            if args.differential_AD==0:
                filename = os.path.join(save_path,"gmm_{}.pkl".format(num_components))
            else:
                filename = os.path.join(save_path,"gmm_{}_fcomb{}.pkl".format(num_components,args.feature_combination))
            pickle.dump(gmm, open(filename, "wb"))
        return gmm
    elif args.ADM=="OC-SVM":
        """Fits a OC-SVM Model"""

        svm = OneClassSVM(gamma="auto").fit(train_embeds)
        print("Finished SVM fitting")
        if save_path:
            if args.differential_AD==0:
                filename = os.path.join(save_path,"svm_{}.pkl".format(num_components))
            else:
                filename = os.path.join(save_path,"svm_{}_fcomb{}.pkl".format(num_components,args.feature_combination))
            pickle.dump(svm, open(filename, "wb"))
        return svm


if __name__ == '__main__':
    
    p = argparse.ArgumentParser(description="Train differential AD model")
    p.add_argument("--data_dir_path",type=str,default="/Data")
    p.add_argument("--seed",type=int,default=1)
    p.add_argument("--model_dir",type=str)
    p.add_argument("--batch_size",type=int,default=2)
    p.add_argument("--compression",type=str,choices=["c0", "c23","c40"],default='c0')
    p.add_argument("--feature_combination",type=int,choices=[0,1,2,3],default=2)
    p.add_argument("--differential_AD",type=int,choices=[0,1],default=1)
    p.add_argument("--frames_sampled",type=int,default=40)
    p.add_argument("--ADM",type=str,choices=["GMM","OC-SVM"],default="GMM")
    p.add_argument("--backbone",type=str,choices = ["efficientnet", "resnet", "xception"],default="efficientnet")
    args = p.parse_args()
    print(args)
    save_dir = os.path.join("saved_AD_modules",args.backbone)
    # Set the random seed 
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    #-------------------- Get Backbone ------------------------------
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    model_dir  = os.path.join(".", "saved_models",args.backbone, args.model_dir)
    if args.backbone =="efficientnet":
        Backbone = Detector(extract_embeddings=True,multi_class=2).to(device)
    elif args.backbone =="resnet":
        Backbone = ResNet50(extract_embeddings=True,multi_class=2).to(device)
    elif args.backbone =="xception":
        Backbone = Xception(extract_embeddings=True,multi_class=2).to(device)
    Backbone.load_state_dict(torch.load(model_dir)["model"])
    #-------------------- Train ADM ---------------------------------

    if args.differential_AD==0:
        train_loader= get_dataloaders_AD(args)
        train_embeds = get_embeds(args,train_loader, Backbone, device)           
        train_AD_module(args,train_embeds,save_path = save_dir, num_components=3)
    else:
        train_loader= get_dataloaders_differential_AD(args)
        train_embeds = get_combined_embeds(args,train_loader, Backbone, device)
        train_AD_module(args,train_embeds,save_path = save_dir, num_components=3)
   
