import numpy as np
import cv2
from skimage import measure
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import mean_squared_error as compare_mse


import os

from math import sqrt

cntx =0
psnrs = 0
ssims = 0
nrmse = 0
mse = 0

for val_path in os.listdir('./dataset/test/test_C'):

    if '.png' in val_path or '.jpg' or '.JPG' or '.bmp' in val_path:
        # print(val_path)
        imtarget = cv2.imread('./test_result/detected_shadow/'+ val_path[:-4]+'.jpg')
        # imoutput = cv2.resize(imoutput,(256,256))
        # imtarget = cv2.imread('./dataset/test/test_C/' + val_path)
        # imoutput = cv2.imread('./dataset/ShadowRemoval/' + val_path)
        imoutput = cv2.imread('./test_result/grid/'+ val_path[:-4]+'.jpg')
        # imtarget = cv2.resize(imtarget,(256,256))
        # immask = cv2.imread('F:/Dataset/ISTD_Dataset/test/test_B/' + val_path)
        # immask = cv2.resize(immask,(256,256))
        # immask = immask[:, :, 0:1]

        #bin_mask = np.where(immask > 128, np.ones_like(immask), np.zeros_like(immask))

        s = compare_ssim(imoutput, imtarget, channel_axis=2)
        p = compare_psnr(imoutput, imtarget)

        m = compare_mse(imoutput,imtarget)
        psnrs = psnrs+p
        ssims = ssims+s

        mse = mse+m
        cntx+=1
        print("TEST: PSNR: %.4f,SSIM:%.4f, RMSE:%.4f" % (p, s, m))
        print(val_path[:-4])
psnr = psnrs/cntx#PSNR的值越大，表示融合图像的质量越好
ssim = ssims/cntx#越接近1，代表相似度越高

mse = mse/cntx
rmse = sqrt(mse)

print(cntx)
print("TEST: PSNR: %.4f,SSIM:%.4f, RMSE:%.4f" % (psnr,ssim,rmse))