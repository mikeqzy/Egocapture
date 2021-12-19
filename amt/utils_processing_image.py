'''
from the prox data, this script is to flip every 10 frame horizentally and save them for annotation
'''

import numpy as np
import cv2
import glob, os, sys
import pdb

seq_names = ['BasementSittingBooth_00142_01', 'MPH112_00034_01','MPH11_00150_01','MPH16_00157_01',
            'MPH1Library_00034_01','MPH8_00168_01', 'N0SittingBooth_00169_01', ' N0Sofa_00034_01',
            'N3Library_00157_02', 'N3Office_00139_01', 'N3OpenArea_00157_02', 'Werkraum_03403_01']

prox_src = '/vlg-data/PROX/qualitative/recordings/'
prox_dst = os.path.join(os.getcwd(), 'data')



for seq in seq_names:
    print('-- processing: {}'.format(seq))
    seq_fullpath = os.path.join(prox_src, seq, 'Color')
    imgs = sorted(glob.glob(seq_fullpath + '/*.jpg'))
    outfolder = os.path.join(prox_dst, seq)
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)

    for imgfile in imgs[::30]:
        img = cv2.imread(imgfile)
        flipHorizontal = cv2.flip(img, 1)
        imgname = os.path.basename(imgfile)
        imgoutname = os.path.join(outfolder, imgname).replace('.jpg', '.png')
        cv2.imwrite(imgoutname, flipHorizontal)




