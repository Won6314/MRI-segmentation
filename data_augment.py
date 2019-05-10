import os
import numpy as np
import nibabel as nib
import scipy.ndimage
import glob
import matplotlib.pyplot as plt
import math

#in_data에 [512][512][70]이 들어감
segment_list = glob.glob('D:\seg_input\SEG_complete\*.nii')
qsm_list = glob.glob('D:\seg_input\QSM_complete\*.nii')

i=0
interval = 45
rotate = 15
#range(start_file,end_file)
#now, 0~84/3
for k in range(0,28):
    file = os.path.join(segment_list[k])
    img = nib.load(file)
    seg_data = img.get_data()
    file = os.path.join(qsm_list[k])
    img = nib.load(file)
    qsm_data = img.get_data()
    
    for slice_num in range(0,70):
        seg_slice = seg_data[:, :, slice_num]
        for seg_num in range(1,8):
            seg_slice[seg_slice==(seg_num*2-1)]=seg_num
            seg_slice[seg_slice==(seg_num*2)]  =seg_num
        seg_slice[seg_slice==15]=0
        seg_slice[seg_slice==16]=0
        seg_slice = np.array(seg_slice,dtype = np.int32)
        qsm_slice = qsm_data[:, :, slice_num]
        print('saving for file %d, slice %d' %(k,slice_num) )
        if np.mean(seg_slice)>0.05:
            fig = plt.figure()
            fig.add_subplot(1, 2, 1)
            plt.imshow(qsm_slice, cmap='gray')
            fig.add_subplot(1, 2, 2)
            plt.imshow(seg_slice)
        for rot in range(0, 90, rotate):
            seg_slice = scipy.ndimage.rotate(seg_slice,rot, reshape = False, mode = 'constant', cval=0)
            seg_slice = np.clip(seg_slice, 0, 7)
            qsm_slice = scipy.ndimage.rotate(qsm_slice,rot, reshape = False, mode = 'constant', cval=0)
            if np.mean(seg_slice)>0.05:
                fig = plt.figure()
            for ys in range(0,512,interval):
                for xs in range(0,512,interval):
                    x=xs*math.cos(math.radians(rot))-ys*math.sin(math.radians(rot))
                    y=xs*math.sin(math.radians(rot))+ys*math.cos(math.radians(rot))
                    x=int(x)
                    y=int(y)
                    seg_image = seg_slice[y - 64:y + 64, x - 64:x + 64]
                    is_save=np.mean(seg_image == 0)
                    if(is_save<0.95):
                        np.save('D:\_unet\seg_data\seg1_%d' % i, seg_image)
                        np.save('D:\_unet\seg_data\seg1_fliplr_%d' % i, np.fliplr(seg_image))
                        np.save('D:\_unet\seg_data\seg1_flipud_%d' % i, np.flipud(seg_image))
                        np.save('D:\_unet\seg_data\seg1_fade_%d' % i, seg_image)
                        qsm_image = qsm_slice[y - 64:y + 64, x - 64:x + 64]
                        np.save('D:\_unet\qsm_data\qsm1_%d' % i, qsm_image)
                        np.save('D:\_unet\qsm_data\qsm1_fliplr_%d' % i, np.fliplr(qsm_image))
                        np.save('D:\_unet\qsm_data\qsm1_flipud_%d' % i, np.flipud(qsm_image))
                        np.save('D:\_unet\qsm_data\qsm1_fade_%d' % i, qsm_image*0.95)
                        print('saved %d files'%i)
                        i+=1