import argparse
import os
import torch
from models.model import Detector
from models.resnet50 import ResNet50
from models.xception import Xception
from prepare_data import get_dataloaders
import random
import numpy as np
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from utils.sam import SAM
from utils.scheduler import LinearDecayLR
import time
from datetime import timedelta


if __name__ == '__main__':   

    p = argparse.ArgumentParser(description="Training of BackBone.")
    p.add_argument("--data_dir_path",type=str,default="/Data")
    p.add_argument("--lr",type=float,default=1e-3)
    p.add_argument("--batch_size",type=int,default=32)
    p.add_argument("--epochs",type=int,default=100)
    p.add_argument("--seed",type=int,default=1)
    p.add_argument("--log_interval",type=int,default=40)
    p.add_argument("--compression",type=str,choices=["c0", "c23","c40"],default='c0')
    p.add_argument("--frames_sampled",type=int,default=40)
    p.add_argument("--backbone",type=str,choices = ["efficientnet", "resnet", "xception"],default="efficientnet")
    args = p.parse_args()
    print(args,flush=True)
    save_dir = os.path.join("saved_models",args.backbone)
    # Set the random seed 
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    #-------------------- Define dataloaders -------------------
    train_loader, val_loader= get_dataloaders(args)
    
    #------------------ Define model ---------------------
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if args.backbone =="efficientnet":
        model = Detector(multi_class=2).to(device)
    elif args.backbone =="resnet":
        model = ResNet50(multi_class=2).to(device)
    elif args.backbone =="xception":
        model = Xception(multi_class=2).to(device)

    #------- Define optimizer ---------
    optimizer = SAM(model.parameters(),torch.optim.SGD,lr=0.001, momentum=0.9)
    lr_scheduler=LinearDecayLR(optimizer, args.epochs, int(args.epochs/4*3))
    criterion = nn.CrossEntropyLoss()
    #---------------------------Training Loop----------------------------------------------
    train_losses = []
    n_weight = 5  # Set the desired number of top weights to keep
    weight_dict = {}
    start_time = time.monotonic()
    for epoch in range(1, args.epochs+ 1):
        
        train_loss = 0
        model.train()
        for batch_idx, data in enumerate(train_loader):
            
            x = data['img'].to(device)
            labels = data['label'].to(device)
            for i in range(2):
                y_pred = model(x)
                if i==0:
                    pred_first = y_pred
                loss = criterion(y_pred,labels)
                optimizer.zero_grad()
                loss.backward()
                if i==0:
                    optimizer.first_step(zero_grad=True)
                else:
                    optimizer.second_step(zero_grad=True)
            train_loss += criterion(pred_first,labels).item()
            if batch_idx % args.log_interval == 0:
                 print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(x), len(train_loader.dataset)*2,
                100.* args.batch_size/2 * batch_idx / float(len(train_loader.dataset)),
                loss.item() / len(x)),flush=True)   
        
        lr_scheduler.step()         
        #------------------ -------Model Validation -------------------------------------
        model.eval()
        predictions = np.array([])
        real_labels = np.array([])
        with torch.no_grad():
            
            for batch_idx, data in enumerate(val_loader):
                
                x = data['img'].to(device)
                labels = data['label']
                y = model(x)
                y_probs = F.softmax(y,dim=1)
                predictions = np.append(predictions,y_probs[:,1].cpu().detach().numpy())
                real_labels = np.append(real_labels,labels.numpy())
 
        val_auc = roc_auc_score(real_labels, predictions)

    #----------------------------------- Saving Models ---------------------------------------------
        os.makedirs(save_dir, exist_ok=True)
        if len(weight_dict)< n_weight:
            save_model_path = os.path.join(save_dir, 'model_epoch{}_val{:.4f}_{}.tar'.format(epoch,val_auc,args.compression))
            weight_dict[save_model_path]=val_auc
            torch.save({
                'model': model.state_dict(),
                'epoch': epoch
            }, save_model_path)
            last_val_auc = min(weight_dict.values())
            print('Saving model from epoch {}'.format(epoch),flush=True)

        elif val_auc >= last_val_auc:
            save_model_path = os.path.join(save_dir, 'model_epoch{}_val{:.4f}_{}.tar'.format(epoch, val_auc,args.compression))
            for k in list(weight_dict.keys()):
                if weight_dict[k] == last_val_auc:
                    del weight_dict[k]
                    os.remove(k)
                    break
            weight_dict[save_model_path] = val_auc
            torch.save({
                'model': model.state_dict(),
                'epoch': epoch
            }, save_model_path)
            last_val_auc = min(weight_dict.values())
            print('Saving model from epoch {}'.format(epoch),flush=True)  
    #--------------------------------- Logging ---------------------------------------------------------------
        print('====> Epoch: {} Average loss: {:.5f}, AUC-ROC : {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset), val_auc),flush=True)
        train_losses.append(train_loss / len(train_loader.dataset))
        end_time = time.monotonic()
        elapsed_time = timedelta(seconds=end_time - start_time)
        print("====> Time Elapsed : {}".format(timedelta(seconds=end_time - start_time)))
        print("----------------------------------------------------------------")

            
          
           
                
            
    
    
