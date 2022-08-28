import os
import numpy as np
import glob

names=[]
img_files = []
for infile in sorted(glob.glob(r'/data/scratch/acw676/new_data/task1/valid/img/*.npy')):
    img_files.append(infile)
    names.append(infile[46:-4])
    
    
sc_files=[]
for n in names:
    sc_files.append(r'/data/scratch/acw676/new_data/task1/valid/sc_gt/'+n+'sc_gt.npy')
    
    
path_img='/data/scratch/acw676/new_data/patches/valid/img/'
path_gt='/data/scratch/acw676/new_data/patches/valid/gt/'


def find_(img,sc_gt,ID):
    count=0
    sc_count=0
    if np.sum(sc_gt)!=0:
        sc_count=sc_count+1
        for x in range(sc_gt.shape[0]):
           for y in range(sc_gt.shape[1]):
               if sc_gt[y,x]==1:
                   crop_gt=sc_gt[y-32:y+32,x-32:x+32]
                   if np.sum(crop_gt)>=20:
                       count=count+1
                       if (count % 20 ==0):
                           filename_img = os.path.join(path_img,ID+'_'+str(count+1))
                           crop_img=img[y-24:y+24,x-24:x+24]
                           filename_gt = os.path.join(path_gt,ID+'_'+str(count+1))
                           np.save(filename_gt, crop_gt)
                           np.save(filename_img, crop_img)                  
for i in range(880):
    print(i)
    img=np.load(img_files[i])
    sc_gt=np.load(sc_files[i])
    sc_gt[np.where(sc_gt>0)]=1
    n_=names[i]
    if np.sum(sc_gt)!=0:
         _=find_(img,sc_gt,n_)
