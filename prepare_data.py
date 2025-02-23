from dataset_class.pseudo_deepfake_dataset import Pseudo_Deepfake_Dataset
from dataset_class.custom_dataset import Custom_Dataset
from dataset_class.differential_AD_dataset import Differential_AD_dataset
from dataset_class.AD_dataset import AD_dataset
from torchvision import transforms
import os
import torch
from torch.utils.data.distributed import DistributedSampler
from utils.get_splits import get_dirs_real_FF, get_dirs_fake_FF, get_dirs_CDF, get_dirs_DF1, get_dirs_FNet

def get_dataloaders(args):
    
    #------------------------Define transformations -------------------
    TRANSFORM_IMG = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    #----------------------- Import real datasets ------------------------------
    train_dataset = Pseudo_Deepfake_Dataset(root_dir = os.path.join(args.data_dir_path,"FF++/original_sequences",args.compression),
                               transform=TRANSFORM_IMG,
                               split_file = get_dirs_real_FF(os.path.join(args.data_dir_path,"FF++/splits/train.json")),
                               frames_sampled=args.frames_sampled
                               )
    val_dataset = Pseudo_Deepfake_Dataset(root_dir = os.path.join(args.data_dir_path,"FF++/original_sequences",args.compression),
                             transform=TRANSFORM_IMG,
                             split_file = get_dirs_real_FF(os.path.join(args.data_dir_path,"FF++/splits/val.json")),
                             frames_sampled=args.frames_sampled
                             )
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               collate_fn=train_dataset.collate_fn,
                                               shuffle=True,
                                               batch_size=args.batch_size//2
                                               )
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             collate_fn=val_dataset.collate_fn,
                                             batch_size=args.batch_size//2
                                             )
    return train_loader,val_loader

def get_dataloaders_testing(args):
    
    #------------------------Define transformations -------------------
    TRANSFORM_IMG = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    if args.dataset =="FF":

        real_dataset = Custom_Dataset(root_dir = os.path.join(args.data_dir_path,"FF++/original_sequences",args.compression),
                              transform=TRANSFORM_IMG,
                              split_file = get_dirs_real_FF(os.path.join(args.data_dir_path,"FF++/splits/test.json")),
                              frames_sampled=args.frames_sampled
                              )
        df_dataset = Custom_Dataset(root_dir = os.path.join(args.data_dir_path,"FF++/manipulated_sequences/Deepfakes",args.compression),
                                            transform=TRANSFORM_IMG,
                                            split_file = get_dirs_fake_FF(os.path.join(args.data_dir_path,"FF++/splits/test.json")),
                                            frames_sampled=args.frames_sampled
                                            )
        
        f2f_dataset = Custom_Dataset(root_dir = os.path.join(args.data_dir_path,"FF++/manipulated_sequences/Face2Face",args.compression),
                                            transform=TRANSFORM_IMG,
                                            split_file = get_dirs_fake_FF(os.path.join(args.data_dir_path,"FF++/splits/test.json")),
                                            frames_sampled=args.frames_sampled
                                            )
        fsw_dataset = Custom_Dataset(root_dir = os.path.join(args.data_dir_path,"FF++/manipulated_sequences/FaceSwap",args.compression),
                                            transform=TRANSFORM_IMG,
                                            split_file = get_dirs_fake_FF(os.path.join(args.data_dir_path,"FF++/splits/test.json")),
                                            frames_sampled=args.frames_sampled
                                            )
        nt_dataset = Custom_Dataset(root_dir = os.path.join(args.data_dir_path,"FF++/manipulated_sequences/NeuralTextures",args.compression),
                                            transform=TRANSFORM_IMG,
                                            split_file = get_dirs_fake_FF(os.path.join(args.data_dir_path,"FF++/splits/test.json")),
                                            frames_sampled=args.frames_sampled
                                            )

        real_loader = torch.utils.data.DataLoader(real_dataset, batch_size=args.batch_size)
        df_loader = torch.utils.data.DataLoader(df_dataset, batch_size=args.batch_size)
        f2f_loader = torch.utils.data.DataLoader(f2f_dataset, batch_size=args.batch_size)
        fsw_loader = torch.utils.data.DataLoader(fsw_dataset, batch_size=args.batch_size)
        nt_loader = torch.utils.data.DataLoader(nt_dataset, batch_size=args.batch_size)
        return real_loader, df_loader, f2f_loader, fsw_loader, nt_loader
    elif args.dataset =="CDF":
        cdf_real_dataset = Custom_Dataset(root_dir = os.path.join(args.data_dir_path,"Celeb-DF/"),
                                          transform=TRANSFORM_IMG,
                                          split_file = get_dirs_CDF(os.path.join(args.data_dir_path,"Celeb-DF/splits/test.txt"),real=True),
                                          frames_sampled=args.frames_sampled
                                          )
        cdf_fake_dataset = Custom_Dataset(root_dir = os.path.join(args.data_dir_path,"Celeb-DF/"),
                                          transform=TRANSFORM_IMG,
                                          split_file = get_dirs_CDF(os.path.join(args.data_dir_path,"Celeb-DF/splits/test.txt"),real=False),
                                          frames_sampled=args.frames_sampled
                                          )
        cdf_real_loader = torch.utils.data.DataLoader(cdf_real_dataset, batch_size=args.batch_size)
        cdf_fake_loader = torch.utils.data.DataLoader(cdf_fake_dataset, batch_size=args.batch_size)
        return cdf_real_loader, cdf_fake_loader
    
    elif args.dataset =="DF1":
        df1_real_dataset = Custom_Dataset(root_dir = os.path.join(args.data_dir_path,"FF++/original_sequences/c0"),
                                          transform=TRANSFORM_IMG,
                                          split_file = get_dirs_DF1(os.path.join(args.data_dir_path,"DeeperForensics/splits/test.txt"),real=True),
                                          frames_sampled=args.frames_sampled
                                          )
        df1_fake_dataset = Custom_Dataset(root_dir = os.path.join(args.data_dir_path,"DeeperForensics/manipulated_sequences"),
                                          transform=TRANSFORM_IMG,
                                          split_file = get_dirs_DF1(os.path.join(args.data_dir_path,"DeeperForensics/splits/test.txt"),real=False),
                                          frames_sampled=args.frames_sampled
                                          )
        df1_real_loader = torch.utils.data.DataLoader(df1_real_dataset, batch_size=args.batch_size)
        df1_fake_loader = torch.utils.data.DataLoader(df1_fake_dataset, batch_size=args.batch_size)
        return df1_real_loader, df1_fake_loader
    elif args.dataset =="FNet":
        fnet_real_dataset = Custom_Dataset(root_dir = os.path.join(args.data_dir_path,"ForgeryNet"),
                                      transform=TRANSFORM_IMG,
                                      split_file = get_dirs_FNet(os.path.join(args.data_dir_path,"ForgeryNet/splits/test_real.txt")),
                                      frames_sampled=args.frames_sampled
                                      )
        fnet_fake_dataset = Custom_Dataset(root_dir = os.path.join(args.data_dir_path,"ForgeryNet"),
                                      transform=TRANSFORM_IMG,
                                      split_file = get_dirs_FNet(os.path.join(args.data_dir_path,"ForgeryNet/splits/test_fake.txt")),
                                      frames_sampled=args.frames_sampled
                                      )
        fnet_real_loader = torch.utils.data.DataLoader(fnet_real_dataset, batch_size=args.batch_size//2)
        fnet_fake_loader = torch.utils.data.DataLoader(fnet_fake_dataset, batch_size=args.batch_size//2)
        return fnet_real_loader, fnet_fake_loader
    elif args.dataset =="FSh":

        real_dataset = Custom_Dataset(root_dir = os.path.join(args.data_dir_path,"FF++/original_sequences",args.compression),
                              transform=TRANSFORM_IMG,
                              split_file = get_dirs_real_FF(os.path.join(args.data_dir_path,"FF++/splits/test.json")),
                              frames_sampled=args.frames_sampled
                              )
        fsh_dataset = Custom_Dataset(root_dir = os.path.join(args.data_dir_path,"FF++/manipulated_sequences/FaceShifter",args.compression),
                                            transform=TRANSFORM_IMG,
                                            split_file = get_dirs_fake_FF(os.path.join(args.data_dir_path,"FF++/splits/test.json")),
                                            frames_sampled=args.frames_sampled
                                            )
        real_loader = torch.utils.data.DataLoader(real_dataset, batch_size=args.batch_size)
        fsh_loader = torch.utils.data.DataLoader(fsh_dataset, batch_size=args.batch_size)
        return real_loader,fsh_loader
    



def get_dataloaders_differential_AD(args):
    
    #------------------------Define transformations -------------------
    TRANSFORM_IMG = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    #----------------------- Import real datasets ---------------------
    
    
    train_dataset = Differential_AD_dataset(root_dir = os.path.join(args.data_dir_path,"FF++/original_sequences",args.compression),
                               images_to_pair=8,
                               transform=TRANSFORM_IMG,
                               split_file = get_dirs_real_FF(os.path.join(args.data_dir_path,"FF++/splits/train.json")),
                               frames_sampled=args.frames_sampled
                               )
    
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               #collate_fn=train_dataset.collate_fn,
                                               shuffle=True,
                                               batch_size=args.batch_size//2
                                               )
    return train_loader

def get_dataloaders_differential_AD_testing(args):
    
    #------------------------Define transformations -------------------
    TRANSFORM_IMG = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    if args.dataset =="FF":

        real_dataset = Differential_AD_dataset(root_dir = os.path.join(args.data_dir_path,"FF++/original_sequences",args.compression),
                                   images_to_pair=args.images_to_pair,
                                   transform=TRANSFORM_IMG,
                                   split_file = get_dirs_real_FF(os.path.join(args.data_dir_path,"FF++/splits/test.json")),
                                   frames_sampled=args.frames_sampled
                                   )
        df_dataset = Differential_AD_dataset(root_dir = os.path.join(args.data_dir_path,"FF++/manipulated_sequences/Deepfakes",args.compression),
                                   images_to_pair=args.images_to_pair,
                                   transform=TRANSFORM_IMG,
                                   split_file = get_dirs_fake_FF(os.path.join(args.data_dir_path,"FF++/splits/test.json")),
                                   frames_sampled=args.frames_sampled
                                   )       
        f2f_dataset = Differential_AD_dataset(root_dir = os.path.join(args.data_dir_path,"FF++/manipulated_sequences/Face2Face",args.compression),
                                   images_to_pair=args.images_to_pair,
                                   transform=TRANSFORM_IMG,
                                   split_file = get_dirs_fake_FF(os.path.join(args.data_dir_path,"FF++/splits/test.json")),
                                   frames_sampled=args.frames_sampled
                                   )       
        fsw_dataset = Differential_AD_dataset(root_dir = os.path.join(args.data_dir_path,"FF++/manipulated_sequences/FaceSwap",args.compression),
                                   images_to_pair=args.images_to_pair,
                                   transform=TRANSFORM_IMG,
                                   split_file = get_dirs_fake_FF(os.path.join(args.data_dir_path,"FF++/splits/test.json")),
                                   frames_sampled=args.frames_sampled
                                   )
        nt_dataset = Differential_AD_dataset(root_dir = os.path.join(args.data_dir_path,"FF++/manipulated_sequences/NeuralTextures",args.compression),
                                   images_to_pair=args.images_to_pair,
                                   transform=TRANSFORM_IMG,
                                   split_file = get_dirs_fake_FF(os.path.join(args.data_dir_path,"FF++/splits/test.json")),
                                   frames_sampled=args.frames_sampled
                                   )
        
        real_loader = torch.utils.data.DataLoader(real_dataset, batch_size=args.batch_size//2)
        df_loader = torch.utils.data.DataLoader(df_dataset, batch_size=args.batch_size//2)
        f2f_loader = torch.utils.data.DataLoader(f2f_dataset, batch_size=args.batch_size//2)
        fsw_loader = torch.utils.data.DataLoader(fsw_dataset, batch_size=args.batch_size//2)
        nt_loader = torch.utils.data.DataLoader(nt_dataset, batch_size=args.batch_size//2)
        return real_loader, df_loader, f2f_loader, fsw_loader, nt_loader
    elif args.dataset =="CDF":
        cdf_real_dataset = Differential_AD_dataset(root_dir = os.path.join(args.data_dir_path,"Celeb-DF/"),
                                      images_to_pair=args.images_to_pair,
                                      transform=TRANSFORM_IMG,
                                      split_file = get_dirs_CDF(os.path.join(args.data_dir_path,"Celeb-DF/splits/test.txt"),real=True),
                                      frames_sampled=args.frames_sampled
                                      )
        cdf_fake_dataset = Differential_AD_dataset(root_dir = os.path.join(args.data_dir_path,"Celeb-DF/"),
                                      images_to_pair=args.images_to_pair,
                                      transform=TRANSFORM_IMG,
                                      split_file = get_dirs_CDF(os.path.join(args.data_dir_path,"Celeb-DF/splits/test.txt"),real=False),
                                      frames_sampled=args.frames_sampled
                                      )
        cdf_real_loader = torch.utils.data.DataLoader(cdf_real_dataset, batch_size=args.batch_size//2)
        cdf_fake_loader = torch.utils.data.DataLoader(cdf_fake_dataset, batch_size=args.batch_size//2)
        return cdf_real_loader, cdf_fake_loader
    elif args.dataset =="DF1":
        df1_real_dataset = Differential_AD_dataset(root_dir = os.path.join(args.data_dir_path,"FF++/original_sequences/c0"),
                                      images_to_pair=args.images_to_pair,
                                      transform=TRANSFORM_IMG,
                                      split_file = get_dirs_DF1(os.path.join(args.data_dir_path,"DeeperForensics/splits/test.txt"),real=True),
                                      frames_sampled=args.frames_sampled
                                      )
        df1_fake_dataset = Differential_AD_dataset(root_dir = os.path.join(args.data_dir_path,"DeeperForensics/manipulated_sequences"),
                                      images_to_pair=args.images_to_pair,
                                      transform=TRANSFORM_IMG,
                                      split_file = get_dirs_DF1(os.path.join(args.data_dir_path,"DeeperForensics/splits/test.txt"),real=False),
                                      frames_sampled=args.frames_sampled
                                      )
        df1_real_loader = torch.utils.data.DataLoader(df1_real_dataset, batch_size=args.batch_size//2)
        df1_fake_loader = torch.utils.data.DataLoader(df1_fake_dataset, batch_size=args.batch_size//2)
        return df1_real_loader, df1_fake_loader
    elif args.dataset =="FNet":
        fnet_real_dataset = Differential_AD_dataset(root_dir = os.path.join(args.data_dir_path,"ForgeryNet"),
                                      images_to_pair=args.images_to_pair,
                                      transform=TRANSFORM_IMG,
                                      split_file = get_dirs_FNet(os.path.join(args.data_dir_path,"ForgeryNet/splits/test_real.txt")),
                                      frames_sampled=args.frames_sampled
                                      )
        fnet_fake_dataset = Differential_AD_dataset(root_dir = os.path.join(args.data_dir_path,"ForgeryNet"),
                                      images_to_pair=args.images_to_pair,
                                      transform=TRANSFORM_IMG,
                                      split_file = get_dirs_FNet(os.path.join(args.data_dir_path,"ForgeryNet/splits/test_fake.txt")),
                                      frames_sampled=args.frames_sampled
                                      )
        fnet_real_loader = torch.utils.data.DataLoader(fnet_real_dataset, batch_size=args.batch_size//2)
        fnet_fake_loader = torch.utils.data.DataLoader(fnet_fake_dataset, batch_size=args.batch_size//2)
        return fnet_real_loader, fnet_fake_loader
    elif args.dataset =="FSh":

        real_dataset = Differential_AD_dataset(root_dir = os.path.join(args.data_dir_path,"FF++/original_sequences",args.compression),
                                   images_to_pair=args.images_to_pair,
                                   transform=TRANSFORM_IMG,
                                   split_file = get_dirs_real_FF(os.path.join(args.data_dir_path,"FF++/splits/test.json")),
                                   frames_sampled=args.frames_sampled
                                   )
        fsh_dataset = Differential_AD_dataset(root_dir = os.path.join(args.data_dir_path,"FF++/manipulated_sequences/FaceShifter",args.compression),
                                   images_to_pair=args.images_to_pair,
                                   transform=TRANSFORM_IMG,
                                   split_file = get_dirs_fake_FF(os.path.join(args.data_dir_path,"FF++/splits/test.json")),
                                   frames_sampled=args.frames_sampled
                                   )       
        real_loader = torch.utils.data.DataLoader(real_dataset, batch_size=args.batch_size)
        fsh_loader = torch.utils.data.DataLoader(fsh_dataset, batch_size=args.batch_size)
        return real_loader,fsh_loader


def get_dataloaders_AD(args):
    
    #------------------------Define transformations -------------------
    TRANSFORM_IMG = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    #----------------------- Import real datasets ------------------------------
    
    train_dataset = AD_dataset(root_dir = os.path.join(args.data_dir_path,"FF++/original_sequences",args.compression),
                               transform=TRANSFORM_IMG,
                               split_file = get_dirs_real_FF(os.path.join(args.data_dir_path,"FF++/splits/train.json")),
                               frames_sampled=args.frames_sampled
                               )
    
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               #collate_fn=train_dataset.collate_fn,
                                               shuffle=True,
                                               batch_size=args.batch_size//2
                                               )
    return train_loader
        
def get_dataloaders_AD_testing(args):
    
    #------------------------Define transformations -------------------
    TRANSFORM_IMG = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    if args.dataset =="FF":

        real_dataset = AD_dataset(root_dir = os.path.join(args.data_dir_path,"FF++/original_sequences",args.compression),
                                   transform=TRANSFORM_IMG,
                                   split_file = get_dirs_real_FF(os.path.join(args.data_dir_path,"FF++/splits/test.json")),
                                   frames_sampled=args.frames_sampled
                                   )
        df_dataset = AD_dataset(root_dir = os.path.join(args.data_dir_path,"FF++/manipulated_sequences/Deepfakes",args.compression),
                                   transform=TRANSFORM_IMG,
                                   split_file = get_dirs_fake_FF(os.path.join(args.data_dir_path,"FF++/splits/test.json")),
                                   frames_sampled=args.frames_sampled
                                   )
        
        f2f_dataset = AD_dataset(root_dir = os.path.join(args.data_dir_path,"FF++/manipulated_sequences/Face2Face",args.compression),
                                   transform=TRANSFORM_IMG,
                                   split_file = get_dirs_fake_FF(os.path.join(args.data_dir_path,"FF++/splits/test.json")),
                                   frames_sampled=args.frames_sampled
                                   )
        
        fsw_dataset = AD_dataset(root_dir = os.path.join(args.data_dir_path,"FF++/manipulated_sequences/FaceSwap",args.compression),
                                   transform=TRANSFORM_IMG,
                                   split_file = get_dirs_fake_FF(os.path.join(args.data_dir_path,"FF++/splits/test.json")),
                                   frames_sampled=args.frames_sampled
                                   )
        nt_dataset = AD_dataset(root_dir = os.path.join(args.data_dir_path,"FF++/manipulated_sequences/NeuralTextures",args.compression),
                                   transform=TRANSFORM_IMG,
                                   split_file = get_dirs_fake_FF(os.path.join(args.data_dir_path,"FF++/splits/test.json")),
                                   frames_sampled=args.frames_sampled)
        
        real_loader = torch.utils.data.DataLoader(real_dataset, batch_size=args.batch_size)
        df_loader = torch.utils.data.DataLoader(df_dataset, batch_size=args.batch_size)
        f2f_loader = torch.utils.data.DataLoader(f2f_dataset, batch_size=args.batch_size)
        fsw_loader = torch.utils.data.DataLoader(fsw_dataset, batch_size=args.batch_size)
        nt_loader = torch.utils.data.DataLoader(nt_dataset, batch_size=args.batch_size)
        return real_loader, df_loader, f2f_loader, fsw_loader, nt_loader
    elif args.dataset =="CDF":
        cdf_real_dataset = AD_dataset(root_dir = os.path.join(args.data_dir_path,"Celeb-DF/"),
                                      transform=TRANSFORM_IMG,
                                      split_file = get_dirs_CDF(os.path.join(args.data_dir_path,"Celeb-DF/splits/test.txt"),real=True),
                                      frames_sampled=args.frames_sampled
                                      )
        cdf_fake_dataset = AD_dataset(root_dir = os.path.join(args.data_dir_path,"Celeb-DF/"),
                                      transform=TRANSFORM_IMG,
                                      split_file = get_dirs_CDF(os.path.join(args.data_dir_path,"Celeb-DF/splits/test.txt"),real=False),
                                      frames_sampled=args.frames_sampled
                                      )
        cdf_real_loader = torch.utils.data.DataLoader(cdf_real_dataset, batch_size=args.batch_size//2)
        cdf_fake_loader = torch.utils.data.DataLoader(cdf_fake_dataset, batch_size=args.batch_size//2)
        return cdf_real_loader, cdf_fake_loader
    elif args.dataset =="DF1":
        df1_real_dataset = AD_dataset(root_dir = os.path.join(args.data_dir_path,"FF++/original_sequences/c0"),
                                      transform=TRANSFORM_IMG,
                                      split_file = get_dirs_DF1(os.path.join(args.data_dir_path,"DeeperForensics/splits/test.txt"),real=True),
                                      frames_sampled=args.frames_sampled
                                      )
        df1_fake_dataset = AD_dataset(root_dir = os.path.join(args.data_dir_path,"DeeperForensics/manipulated_sequences"),
                                      transform=TRANSFORM_IMG,
                                      split_file = get_dirs_DF1(os.path.join(args.data_dir_path,"DeeperForensics/splits/test.txt"),real=False),
                                      frames_sampled=args.frames_sampled
                                      )
        df1_real_loader = torch.utils.data.DataLoader(df1_real_dataset, batch_size=args.batch_size//2)
        df1_fake_loader = torch.utils.data.DataLoader(df1_fake_dataset, batch_size=args.batch_size//2)
        return df1_real_loader, df1_fake_loader
    elif args.dataset =="FNet":
        fnet_real_dataset = AD_dataset(root_dir = os.path.join(args.data_dir_path,"ForgeryNet"),
                                      transform=TRANSFORM_IMG,
                                      split_file = get_dirs_FNet(os.path.join(args.data_dir_path,"ForgeryNet/splits/test_real.txt")),
                                      frames_sampled=args.frames_sampled
                                      )
        fnet_fake_dataset = AD_dataset(root_dir = os.path.join(args.data_dir_path,"ForgeryNet"),
                                      transform=TRANSFORM_IMG,
                                      split_file = get_dirs_FNet(os.path.join(args.data_dir_path,"ForgeryNet/splits/test_fake.txt")),
                                      frames_sampled=args.frames_sampled
                                      )
        fnet_real_loader = torch.utils.data.DataLoader(fnet_real_dataset, batch_size=args.batch_size//2)
        fnet_fake_loader = torch.utils.data.DataLoader(fnet_fake_dataset, batch_size=args.batch_size//2)
        return fnet_real_loader, fnet_fake_loader
    elif args.dataset =="FSh":

        real_dataset = AD_dataset(root_dir = os.path.join(args.data_dir_path,"FF++/original_sequences",args.compression),
                                   transform=TRANSFORM_IMG,
                                   split_file = get_dirs_real_FF(os.path.join(args.data_dir_path,"FF++/splits/test.json")),
                                   frames_sampled=args.frames_sampled
                                   )
        fsh_dataset = AD_dataset(root_dir = os.path.join(args.data_dir_path,"FF++/manipulated_sequences/FaceShifter",args.compression),
                                   transform=TRANSFORM_IMG,
                                   split_file = get_dirs_fake_FF(os.path.join(args.data_dir_path,"FF++/splits/test.json")),
                                   frames_sampled=args.frames_sampled
                                   )
        real_loader = torch.utils.data.DataLoader(real_dataset, batch_size=args.batch_size)
        fsh_loader = torch.utils.data.DataLoader(fsh_dataset, batch_size=args.batch_size)
        return real_loader,fsh_loader



        
        