
     #### Set all paths #####
   #######################################

train_imgs='/data/scratch/acw676/new_data/patches/train/img/'
train_masks='/data/scratch/acw676/new_data/patches/train/gt/'   ## segmentation ground-truths

val_imgs='/data/scratch/acw676/new_data/patches/valid/img/'
val_masks='/data/scratch/acw676/new_data/patches/valid/gt/'

path_to_save_check_points='/data/home/acw676/test/patches/Weights_LR/'+'/model5'
path_to_save_Learning_Curve='/data/home/acw676/test/patches/Weights_LR/'+'/model5'


        #### Set Hyperparameters ####
        #######################################

batch_size=120
Max_Epochs=50
LEARNING_RATE=0.0001

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
train_loader=Data_Loader(train_imgs,train_masks,batch_size)
val_loader=Data_Loader(val_imgs,val_masks,batch_size)

print(len(train_loader))
print(len(val_loader))


   ### LOAD MODELS #####
#######################################
from models import model1


avg_train_losses = []   # losses of all training epochs
avg_valid_losses = []  #losses of all training epochs
avg_valid_DS = []  # all training epochs


### Main training fucntion to update the weights during training / validation #######

def train_fn(loader_train,loader_valid, model, optimizer,loss_fn1,scaler): 
     
    train_losses = [] # loss of each batch
    valid_losses = []  # loss of each batch
    
    model.train()
    loop = tqdm(loader_train)
    for batch_idx, (img1,gt1,label) in enumerate(loop):
        img1 = img1.to(device=DEVICE,dtype=torch.float)
        gt1 = gt1.to(device=DEVICE,dtype=torch.float)
    
        # forward
        with torch.cuda.amp.autocast():
            out1 = model(img1)   
            loss1 = loss_fn1(out1, gt1)
           
            
        # backward
        loss=loss1 
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # update tqdm loop
        loop.set_postfix(loss=loss.item())
        train_losses.append(loss.item())
    
    loop_v = tqdm(loader_valid)
    model.eval()
    for batch_idx, (img1,gt1,label) in enumerate(loop_v):
        img1 = img1.to(device=DEVICE,dtype=torch.float)
        gt1 = gt1.to(device=DEVICE,dtype=torch.float)
        
        # forward
        with torch.no_grad():
            out1 = model(img1)   
            loss1 = loss_fn1(out1, gt1)
        
        loss=loss1
        loop_v.set_postfix(loss=loss.item())
        valid_losses.append(loss.item())
        
    train_loss_per_epoch = np.average(train_losses)
    valid_loss_per_epoch = np.average(valid_losses)
    ## all epochs
    avg_train_losses.append(train_loss_per_epoch)
    avg_valid_losses.append(valid_loss_per_epoch)
    
    return train_loss_per_epoch,valid_loss_per_epoch
    


def save_checkpoint(state, filename=path_to_save_check_points+".pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)
    
def check_accuracy(loader, model, device=DEVICE):
    dice_score1=0
    loop = tqdm(loader)
    model.eval()
    with torch.no_grad():
        for batch_idx, (img1,gt1,label) in enumerate(loop):
            img1 = img1.to(device=DEVICE,dtype=torch.float)
            gt1 = gt1.to(device=DEVICE,dtype=torch.float)
            
            p1 = model(img1)  
            p1 = (p1 >= 0.5) * 1
            dice_score1 += (2 * (p1 * gt1).sum()) / (
                (p1 + gt1).sum() + 1e-8)
               
                
    print(f"Dice score for LA segmentation : {dice_score1/len(loader)}")
    return dice_score1/len(loader)
           
ALPHA = 0.5
BETA = 0.5
GAMMA = .75

class FocalTverskyLoss(nn.Module):
    def _init_(self, weight=None, size_average=True):
        super(FocalTverskyLoss, self)._init_()

    def forward(self, inputs, targets, smooth=.0001, alpha=ALPHA, beta=BETA, gamma=GAMMA):
             
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
        
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        FocalTversky = (1 - Tversky)**gamma
                       
        return FocalTversky
        
epoch_len = len(str(Max_Epochs))
early_stopping = EarlyStopping(patience=5, verbose=True) 
                
def main():
    model = model1().to(device=DEVICE,dtype=torch.float)
    loss_fn1 =FocalTverskyLoss()
    
   ########################

    optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.999),lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(Max_Epochs):
        train_loss,valid_loss=train_fn(train_loader,val_loader, model, optimizer, loss_fn1,scaler)
        print_msg = (f'[{epoch:>{epoch_len}}/{Max_Epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}')
        
        print(print_msg)
        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)
        dice_score= check_accuracy(val_loader, model, device=DEVICE)
        avg_valid_DS.append(dice_score.detach().cpu().numpy())
        
        early_stopping(valid_loss, dice_score)
        if early_stopping.early_stop:
            print("Early stopping Reached at  :",epoch)
            break

if __name__ == "__main__":
    main()


avg_train_losses=avg_train_losses
avg_train_losses=avg_train_losses
avg_valid_DS=avg_valid_DS
# visualize the loss as the network trained
fig = plt.figure(figsize=(10,8))
plt.plot(range(1,len(avg_train_losses)+1),avg_train_losses, label='Training Loss')
plt.plot(range(1,len(avg_valid_losses)+1),avg_valid_losses,label='Validation Loss')
plt.plot(range(1,len(avg_valid_DS)+1),avg_valid_DS,label='Validation DS')

# find position of lowest validation loss
minposs = avg_valid_losses.index(min(avg_valid_losses))+1 
plt.axvline(minposs,linestyle='--', color='r',label='Early Stopping Checkpoint')

font1 = {'size':20}

plt.title("Learning Curve Graph",fontdict = font1)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.ylim(0, 1) 
plt.xlim(0, len(avg_train_losses)+1)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
fig.savefig(path_to_save_Learning_Curve+'.png', bbox_inches='tight')   
