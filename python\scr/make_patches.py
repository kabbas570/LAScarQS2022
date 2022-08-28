import os
import numpy as np
import glob

names=[]
img_files = []
for infile in sorted(glob.glob(r'C:\My_Data\sateg0\task_1_both_data\task1_2d\valid\img/*.npy')):
    img_files.append(infile)
    names.append(infile[46:-4])
    
    
sc_files=[]
for n in names:
    sc_files.append(r'C:\My_Data\sateg0\task_1_both_data\task1_2d\valid\sc_gt/'+n+'sc_gt.npy')
    
    
path=r'C:\My data\temp_data\patches\train\img'
path1=r'C:\My data\temp_data\patches\train\gt'


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
                           filename1 = os.path.join(path1,ID+'_'+str(count+1))
                           crop_img=img[y-24:y+24,x-24:x+24]
                           filename = os.path.join(path,ID+'_'+str(count+1))
                           np.save(filename1, crop_gt)
                           np.save(filename, crop_img)                  
for i in range(880):
    print(i)
    img=np.load(img_files[i])
    sc_gt=np.load(sc_files[i])
    sc_gt[np.where(sc_gt>0)]=1
    n_=names[i]
    if np.sum(sc_gt)!=0:
         _=find_(img,sc_gt,n_)
