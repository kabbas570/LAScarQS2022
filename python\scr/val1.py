   #### Specify all the paths here #####
   
test_imgs='/data/scratch/acw676/new_data/task2/valid/img/'
test_masks='/data/scratch/acw676/new_data/task2/valid/seg_gt/'

path_to_checkpoints="/data/home/acw676/test/weights_LR/task2_munet4.pth.tar"

#path_to_checkpoints="/data/home/acw676/two_stage/m_unet4_Final.pth.tar"

Save_Visual_results= False   ## set to False, if only the quanitative resutls are required

        #### Set Hyperparameters ####
        #######################################

batch_size=20

#### Import All libraies used for training  #####

from tqdm import tqdm
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
import numpy as np
import torch.nn as nn
import torch.optim as optim
from Early_Stopping import EarlyStopping
import matplotlib.pyplot as plt


  ### Data_Generators ########
  
from read_data1 import Data_Loader
test_loader=Data_Loader(test_imgs,test_masks,batch_size)

print(len(test_loader))

   ### LOAD MODELS #####
#######################################
from models import m_unet4,m_unet6
model=m_unet4()

#model=m_unet6()

def Evaluate_model(loader, model, device=DEVICE):
    dice_score1=0
    loop = tqdm(loader)
    model.eval()
    with torch.no_grad():
        for batch_idx, (img1,img2,gt1,label,org_dim) in enumerate(loop):
            img1 = img1.to(device=DEVICE,dtype=torch.float)
            img2 = img2.to(device=DEVICE,dtype=torch.float)
            gt1 = gt1.to(device=DEVICE,dtype=torch.float)
            
            p1 = model(img1,img2)  
            p1 = (p1 > 0.5) * 1
            dice_score1 += (2 * (p1 * gt1).sum()) / (
                (p1 + gt1).sum() + 1e-8)
               
                
    print(f"Dice score for LA segmentation : {dice_score1/len(loader)}")
    return dice_score1/len(loader)
    
def eval_():
    model.to(device=DEVICE,dtype=torch.float)
    optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.999),lr=0)
    checkpoint = torch.load(path_to_checkpoints,map_location=DEVICE)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    Evaluate_model(test_loader,model, device=DEVICE)

if __name__ == "__main__":
    eval_()
    
    