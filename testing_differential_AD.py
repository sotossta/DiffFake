# ------------------ Import libraries --------------------------------
import torch
import random
import argparse
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import os
from models.model import Detector
from models.resnet50 import ResNet50
from models.xception import Xception
from sklearn.mixture import GaussianMixture
from prepare_data import get_dataloaders_differential_AD_testing, get_dataloaders_AD_testing
from sklearn.preprocessing import normalize
import pickle

def normalize_tensor(tensor):
    tensor_shape = tensor.size()
    flattened = tensor.view(tensor_shape[0], -1).detach().cpu().numpy()  # Flatten to 2D for sklearn
    normalized = normalize(flattened, norm='l2')  # Apply L2 normalization (or 'l1', 'max' as needed)
    return torch.tensor(normalized, dtype=tensor.dtype, device=tensor.device).view(tensor_shape)  # Reshape to original

def get_predictions(args,feature_extractor,OCC,device,loader,save_file):

    feature_extractor.eval()
    video_ids = []
    predictions = []
    combined_embeds = []
    feature_extractor.eval()
    video_ids = []
    predictions = []
    combined_embeds = []
    with torch.no_grad():
            if args.differential_AD==0:
                for batch_idx, (img,video_id) in enumerate(tqdm(loader)):     
                    x = img.to(device)
                    f = feature_extractor(x)
                    combined_embeds.append(f.detach().cpu().numpy().reshape(f.shape[0],-1))
                    video_ids.extend(video_id)  
            else:
                for batch_idx, (img1,img2,video_id) in enumerate(tqdm(loader)):     
                    x1 = img1.to(device)
                    x2 = img2.to(device)
                    f_1 = feature_extractor(x1)
                    f_2 = feature_extractor(x2)
                    if args.feature_combination==0:
                        comb = torch.abs(f_1 - f_2) 
                    elif args.feature_combination==1:
                        comb = f_1-f_2       
                    elif args.feature_combination==2:
                        comb = (f_1 - f_2) ** 2     
                    elif args.feature_combination==3:
                        comb = (f_1 - f_2) ** 3    
                    combined_embeds.append(comb.detach().cpu().numpy().reshape(comb.shape[0],-1))
                    video_ids.extend(video_id)

    combined_embeds = np.vstack(combined_embeds)
    combined_embeds = normalize(combined_embeds)
    predictions = OCC.score_samples(combined_embeds)
    video_ids = np.array(video_ids)
    # Create a dictionary to hold predictions for each video
    video_predictions = {}
    for vid, pred in zip(video_ids, predictions):
        if vid not in video_predictions:
            video_predictions[vid] = []
        video_predictions[vid].append(pred)
       
    video_predictions = {vid: np.mean(preds) for vid, preds in video_predictions.items()}
    video_preds = np.array(list(video_predictions.values()))
    return video_preds

def roc_auc(predictions_real, predictions_fake):

    true_labels = np.append(np.ones(len(predictions_real)),np.zeros(len(predictions_fake)))
    predictions = np.append(predictions_real,predictions_fake)
    fpr, tpr, thresholds = roc_curve( true_labels, predictions, pos_label=1)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    predictions = [1 if score >= optimal_threshold else 0 for score in predictions]
    auc_roc = roc_auc_score(y_true = true_labels, y_score=predictions)
    return auc_roc, optimal_threshold


def testing_FF(args, feature_extractor, OCC, device, real_loader, df_loader, f2f_loader, fsw_loader, nt_loader):
    
    #Deepfakes
    predictions_real = get_predictions(args, feature_extractor, OCC, device, real_loader,"Real")
    predictions_df = get_predictions(args, feature_extractor, OCC, device, df_loader, "Deepfakes")
    auc_roc_df,threshold_df = roc_auc(predictions_real, predictions_df)
    print("AUC-ROC Deepfakes : {}".format(auc_roc_df))
    #Face2Face
    predictions_f2f = get_predictions(args, feature_extractor, OCC, device, f2f_loader,"Face2Face")
    auc_roc_f2f,threshold_f2f = roc_auc(predictions_real, predictions_f2f)
    print("AUC-ROC Face2Face : {}".format(auc_roc_f2f))
    #FaceSwap
    predictions_fsw = get_predictions(args, feature_extractor, OCC, device, fsw_loader,"FaceSwap")
    auc_roc_fsw,threshold_fsw = roc_auc(predictions_real, predictions_fsw)
    print("AUC-ROC FaceSwap : {}".format(auc_roc_fsw))
    #NeuralTextures  
    predictions_nt = get_predictions(args, feature_extractor, OCC, device, nt_loader,"NeuralTextures")   
    auc_roc_nt,threshold_nt = roc_auc(predictions_real, predictions_nt)
    print("AUC-ROC NeuralTextures : {}".format(auc_roc_nt))
    
    

def testing(args, feature_extractor, OCC, device, real_loader, fake_loader,deepfake_name):
     
    predictions_real = get_predictions(args, feature_extractor, OCC, device, real_loader,deepfake_name+"_real")   
    predictions_fake = get_predictions(args, feature_extractor, OCC, device, fake_loader,deepfake_name+"_fake")
    auc_roc,threshold = roc_auc(predictions_real, predictions_fake)
    print("AUC-ROC {} : {}".format(deepfake_name, auc_roc))
    

if __name__ =='__main__':

    p = argparse.ArgumentParser(description="Results of Differential-AD.")
    p.add_argument("--data_dir_path",type=str,default="/Data")
    p.add_argument("--seed",type=int,default=1)
    p.add_argument("--model_dir",type=str,)
    p.add_argument("--batch_size",type=int,default=64)
    p.add_argument("--dataset",type=str,choices=["FF", "CDF", "DF1","FNet","FSh"],default="FF")
    p.add_argument("--compression",type=str,choices=["c0", "c23", "c40"],default="c23")
    p.add_argument("--feature_combination",type=int,choices=[0,1,2,3],default=2)
    p.add_argument("--differential_AD",type=int,choices=[0,1],default=1)
    p.add_argument("--frames_sampled",type=int,default=40)
    p.add_argument("--images_to_pair",type=int,default=8)
    p.add_argument("--ADM",type=str,choices=["GMM","OC-SVM"],default="GMM")
    p.add_argument("--backbone",type=str,choices = ["efficientnet", "resnet", "xception"],default="efficientnet")
    args = p.parse_args()
    # Set the random seed 
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print(args)
    #-------------------- Get Feature Extractor ------------------------------
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_dir  = os.path.join(".", "saved_models",args.backbone,args.model_dir)

    if args.backbone =="efficientnet":
        Backbone = Detector(extract_embeddings=True,multi_class=2).to(device)
    elif args.backbone =="resnet":
        Backbone = ResNet50(extract_embeddings=True,multi_class=2).to(device)
    elif args.backbone =="xception":
        Backbone = Xception(extract_embeddings=True,multi_class=2).to(device)
    Backbone.load_state_dict(torch.load(model_dir)["model"])
    #------------------- Get AD model --------------------------------------
    if args.ADM == "GMM":
        if args.differential_AD==0:
            with open(os.path.join('saved_AD_modules',args.backbone,'gmm_3.pkl'), 'rb') as f:
                gmm = pickle.load(f)
        else:
            with open(os.path.join('saved_AD_modules',args.backbone,'gmm_3_fcomb{}.pkl'.format(args.feature_combination)), 'rb') as f:
                gmm = pickle.load(f)
    elif args.ADM == "OC-SVM":

        with open('saved_AD_modules',args.backbone,'svm_3_fcomb{}.pkl'.format(args.feature_combination), 'rb') as f:
            gmm = pickle.load(f)        
    #-------------------- Define dataloaders -------------------
    if args.dataset== "FF":
        if args.differential_AD==0:
            test_real_loader,df_loader,f2f_loader, fsw_loader, nt_loader = get_dataloaders_AD_testing(args)
        else:
            test_real_loader,df_loader,f2f_loader, fsw_loader, nt_loader = get_dataloaders_differential_AD_testing(args)
        testing_FF(args, Backbone, gmm, device, test_real_loader, df_loader, f2f_loader, fsw_loader, nt_loader) 
    elif args.dataset== "CDF":

        if args.differential_AD==0:
            cdf_real_loader, cdf_fake_loader = get_dataloaders_AD_testing(args)
        else:
            cdf_real_loader, cdf_fake_loader = get_dataloaders_differential_AD_testing(args)
        testing(args, Backbone, gmm, device, cdf_real_loader, cdf_fake_loader,"Celeb-DF")

    elif args.dataset== "DF1":
        if args.differential_AD==0:
            df1_real_loader, df1_fake_loader = get_dataloaders_AD_testing(args)
        else:
            df1_real_loader, df1_fake_loader = get_dataloaders_differential_AD_testing(args)
        testing(args, Backbone, gmm, device, df1_real_loader, df1_fake_loader,"DeeperForensics-1.0")

    elif args.dataset== "FNet":
        if args.differential_AD==0:
            fnet_real_loader, fnet_fake_loader = get_dataloaders_AD_testing(args)
        else:
            fnet_real_loader, fnet_fake_loader = get_dataloaders_differential_AD_testing(args)
        testing(args, Backbone, gmm, device, fnet_real_loader, fnet_fake_loader,"ForgeryNet")

    elif args.dataset== "FSh":
        if args.differential_AD==0:
            fsh_real_loader, fsh_fake_loader = get_dataloaders_AD_testing(args)
        else:
            fsh_real_loader, fsh_fake_loader = get_dataloaders_differential_AD_testing(args)
        testing(args, Backbone, gmm, device, fsh_real_loader, fsh_fake_loader,"FaceShifter")
    
    
    