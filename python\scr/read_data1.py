import os
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader
import albumentations as A


NUM_WORKERS=0
PIN_MEMORY=True


transform2 = A.Compose([
    A.Resize(width=320, height=320)
])
 
class Dataset_(Dataset):
    def __init__(self, image_dir, mask_dir,transform2=transform2):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(image_dir)

        self.transform2 = transform2

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index][:-4]+'_gt.npy')

        image = np.load(img_path,allow_pickle=True, fix_imports=True)
    
        org_dim=image.shape[0]
        
        mean=np.mean(image)
        std=np.std(image)
        image=(image-mean)/std
        
        mask = np.load(mask_path,allow_pickle=True, fix_imports=True)
        mask[np.where(mask>0)]=1
        
        if org_dim==576:
          temp=np.zeros([640,640])
          temp1=np.zeros([640,640])
          temp[32:608, 32:608] = image
          image=temp
          temp1[32:608, 32:608] = mask
          mask=temp1
          
        if org_dim==480:
         temp=np.zeros([640,640])
         temp[80:560, 80:560] = image
         image=temp
         temp1=np.zeros([640,640])
         temp1[80:560, 80:560] = mask
         mask=temp1
          
        if org_dim==864:
         temp=np.zeros([640,640])
         temp[:,:] = image[112:752,112:752]
         image=temp
         
         temp1=np.zeros([640,640])
         temp1[:,:] = mask[112:752,112:752]
         mask=temp1
         
        if org_dim==784:
         temp=np.zeros([640,640])
         temp[:,:] = image[72:712,72:712]
         image=temp
         
         temp1=np.zeros([640,640])
         temp1[:,:] = mask[72:712,72:712]
         mask=temp1
         
        if org_dim==768:
         temp=np.zeros([640,640])
         temp[:,:] = image[64:704,64:704]
         image=temp
         
         temp1=np.zeros([640,640])
         temp1[:,:] = mask[64:704,64:704]
         mask=temp1

        if self.transform2 is not None:
            augmentations2 = self.transform2(image=image)
            image2 = augmentations2["image"]

            image=np.expand_dims(image, axis=0)
            image2=np.expand_dims(image2, axis=0)
            mask=np.expand_dims(mask, axis=0)
          
        return image,image2,mask,self.images[index][:-4],org_dim
    
def Data_Loader( test_dir,test_maskdir,batch_size,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY):
    
    test_ids = Dataset_( image_dir=test_dir, mask_dir=test_maskdir)

    data_loader = DataLoader(test_ids,batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory,shuffle=True)
    
    return data_loader


# batch_size=5
# val_imgs=r'C:\My data\sateg0\task_2_both_data\tas2_2D\valid\img'
# val_masks=r'C:\My data\sateg0\task_2_both_data\tas2_2D\valid\gt'
# val_loader=Data_Loader(val_imgs,val_masks,batch_size)
# a=iter(val_loader)
# a1=next(a)
# o=a1[4].numpy()
