import numpy as np
from glob import glob
import os
from os.path import join
from shutil import copy2, rmtree
from tqdm import tqdm

from IPython import embed

data_root = '/local/home/zhqian/sp/data/egocapture'
output_root = '/local/home/zhqian/sp/data/amt_annotation_data'


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_recs_from_split(train=True):
    if train:
        recs = glob(join(data_root, 'PROX_temp', 'train', '*'))
    else:
        recs = glob(join(data_root, 'PROX_temp', 'test', '*'))
    return [x.split('/')[-1] for x in recs]

def copy_amt_data():
    recs = get_recs_from_split()
    recs.extend(get_recs_from_split(False))
    imgs = []
    for rec in recs:
        gt_dir = glob(join(data_root, 'PROX_temp', '*', rec, 'body_idx_*', 'results'))[0]
        gt_frames = [int(x[-5:]) for x in glob(join(gt_dir, '*'))]

        rec_dir = join(data_root, 'hololens_data', f'record_{rec[10:18]}', rec)
        rec_dir = glob(join(rec_dir, '2021*'))[0]
        valid_file = np.load(join(rec_dir, 'valid_frame.npz'))
        for imgname, valid in zip(valid_file['imgname'], valid_file['valid']):
            if not valid:
                continue
            frame = int(imgname[-9:-4])
            if frame in gt_frames:
                imgs.append({
                    'imgname': imgname,
                    'rec': rec,
                })
    downsample = 50
    amt_imgs = imgs[::downsample]
    # embed()
    for amt_img in amt_imgs:
        imgname = amt_img['imgname']
        rec = amt_img['rec']
        output_dir = join(output_root, rec, 'images')
        mkdir(output_dir)
        copy2(join(data_root, imgname), output_dir)

if __name__ == '__main__':
    copy_amt_data()