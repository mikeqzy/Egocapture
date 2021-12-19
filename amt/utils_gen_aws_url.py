import numpy as np
import cv2
import glob, os, sys
import pdb
import csv

seq_names = ['BasementSittingBooth_00142_01', 'MPH112_00034_01','MPH11_00150_01','MPH16_00157_01',
            'MPH1Library_00034_01','MPH8_00168_01', 'N0SittingBooth_00169_01', 'N0Sofa_00034_01',
            'N3Library_00157_02', 'N3Office_00139_01', 'N3OpenArea_00157_02', 'Werkraum_03403_01']

prox_src = os.path.join(os.getcwd(), 'data')
url_prefix = 'https://prox2djointsannotation.s3-us-west-2.amazonaws.com'
csv_file = 'img_urls.csv'

with open(csv_file, "w") as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    writer.writerow(['image_url'])
    for seq in seq_names:
        print('-- processing: {}'.format(seq))
        seq_fullpath = os.path.join(prox_src, seq)
        imgs = sorted(glob.glob(seq_fullpath + '/*.png'))

        for imgfile in imgs:
            imgurl = imgfile.replace(os.getcwd(), url_prefix)
            writer.writerow([imgurl])


# https://prox2djointsannotation.s3-us-west-2.amazonaws.com/data/MPH11_00150_01/s001_frame_00031__00.00.01.001.png
