from PIL import Image
import glob
import os
import numpy as np
import time
import tqdm



i=0
print("\tData Preprocessing\t")
dataset='dataset/raw'
os.mkdir(os.path.join(dataset+'/MRI'))
os.mkdir(os.path.join(dataset+'/CT'))
start=time.time()
for x in tqdm.tqdm(glob.glob(os.path.join('dataset/raw')+'/*.*'),desc='Data Processing'):
    im=Image.open(x)
    im=np.array(im)
    m,n=im.shape
    n=int(n/2)
    
    mri=im[:,:n]
    ct=im[:,n:]
    mri_da=Image.fromarray(mri)
    ct_da=Image.fromarray(ct)
    mri_da.save('{0}/MRI/mri-{1}.png'.format(dataset,i))
    ct_da.save('{0}/CT/ct-{1}.png'.format(dataset,i))
    i=i+1
end=time.time()
comp=end-start
print("Execution Time:\t",comp)
