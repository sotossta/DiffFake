import json
import numpy as np

def get_dirs_real_FF(dir):
    
    with open(dir, 'r') as f:
        data_split = json.load(f) 
    data_split_real = [
        x
        for xs in data_split
        for x in xs
        ]
    data_split_real = ['images/'+element for element in data_split_real]
    return data_split_real

def get_dirs_fake_FF(dir):

    with open(dir, 'r') as f:
        data_split = json.load(f) 
    data_split_fake = []
    for pair in data_split:
        num1 = pair[0]
        num2 = pair[1]
        # Generate combinations in both formats
        data_split_fake.append(num1 + "_" + num2)
        data_split_fake.append(num2 + "_" + num1)
    data_split_fake = ['images/'+element for element in data_split_fake]
    return data_split_fake

def get_dirs_CDF(dir,real):
    
    data_split = np.loadtxt(dir,dtype=str)
    if real==True:
        real_idx = np.where(data_split[:,0]=="1")
        data_split_real = data_split[real_idx][:,1].tolist()
        data_split_real = [element.replace(".mp4", "").replace('/', '/images/') for element in data_split_real] 
        return data_split_real
        
    else:
        fake_idx = np.where(data_split[:,0]=="0")
        data_split_fake = data_split[fake_idx][:,1].tolist()
        data_split_fake = [element.replace(".mp4", "").replace('/', '/images/') for element in data_split_fake] 
        return data_split_fake

def get_dirs_DF1(dir,real):
    
    data_split = np.loadtxt(dir,dtype=str)
    if real ==True:
        data_split_real = data_split.tolist()
        data_split_real = [element.split("_")[0] for element in data_split_real]
        data_split_real = ['images/'+element.replace(".mp4", "") for element in data_split_real]
        return data_split_real
    else:
        data_split_fake = data_split.tolist()
        data_split_fake = ['images/'+element.replace(".mp4", "") for element in data_split_fake]
        return data_split_fake

def get_dirs_FNet(dir):

    data_split = np.loadtxt(dir,dtype=str)
    data_split = data_split.tolist()
    data_split = ['images/'+element for element in data_split]
    return data_split


