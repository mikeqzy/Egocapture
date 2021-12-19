import numpy as np
import glob, os, sys
from os.path import join, relpath
import pdb
import csv

import pandas as pd
import json
import pickle

from IPython import embed


# kps_labels=['Nose',
#             'Neck',
#             'Right Shoulder',
#             'Right Elbow',
#             'Right Wrist',
#             'Left Shoulder',
#             'Left Elbow',
#             'Left Wrist',
#             'Middle Hip',
#             'Right Hip',
#             'Right Knee',
#             'Right Ankle',
#             'Left Hip',
#             'Left Knee',
#             'Left Ankle',
#             'Right Eye',
#             'Left Eye',
#             'Right Ear',
#             'Left Ear',
#             'Left BigToe',
#             'Left SmallToe',
#             'Left Heel',
#             'Right BigToe',
#             'Right SmallToe',
#             'Right Heel'
#             ]

kps_labels = ['Nose',
              'Neck Bottom',
              'Right Shoulder',
              'Right Elbow',
              'Right Wrist',
              'Left Shoulder',
              'Left Elbow',
              'Left Wrist',
              'Pelvis',
              'Right Hip',
              'Right Knee',
              'Right Ankle',
              'Right Toebase',
              'Left Hip',
              'Left Knee',
              'Left Ankle',
              'Left Toebase']

n_annotators = 5
max_std = 1.5

def swap_label(label):
    if label[:4] == 'Left':
        return True, 'Right' + label[4:]
    if label[:5] == 'Right':
        return True, 'Left' + label[5:]
    return False, label

def calc_dist(loc):
    dist = loc - np.mean(loc, axis=0)[None, :]
    dist = np.sqrt(np.sum(dist ** 2, axis=1))
    return dist

def filter_None(l):
    return list(filter(lambda x: x is not None, l))

def filter_annotation(locations):
    ann_idx = np.where([x is not None for x in locations])[0]
    ann_locs = filter_None(locations)
    ann_locs = np.stack(ann_locs)
    while True:
        if ann_idx.size < 3:
            break
        change = False
        dist = calc_dist(ann_locs)
        scale = (dist - dist.mean()) / (dist.std() + 1e-5)
        mask = scale < max_std
        if (~mask).sum() > 0:
            change = True
        ann_idx = ann_idx[mask]
        ann_locs = ann_locs[mask]
        if not change:
            break
    return ann_idx


def init_resentry():
    res_entry = {}
    res_entry['annotation'] = {}
    res_entry['annotation']['workerid'] = []
    for kps in kps_labels:
        res_entry['annotation'][kps] = {}
        res_entry['annotation'][kps]['location'] = [None] * n_annotators
        res_entry['annotation'][kps]['mean'] = None
    return res_entry


user_study_file = 'results/amt_batch_results.csv'

df = pd.read_csv(user_study_file)

imagenames = list(df['Input.image_url'].unique())

allresults = []


for img in imagenames:
    sframe = df.loc[df['Input.image_url']==img]['Answer.annotatedResult.keypoints']
    workers = df.loc[df['Input.image_url']==img]['WorkerId']
    outres = init_resentry()
    outres['frame_name'] = '/'.join(img.split('/')[-3:])
    # outres['seq_name'] = img.split('/')[-2]
    for idx, (worker, allkps) in enumerate(zip(workers, sframe)):
        outres['annotation']['workerid'].append(worker)
        allkps = json.loads(allkps)
        for anno in allkps:
            kpnamefull = anno['label']
            kploc = np.array([anno['x'], anno['y']], dtype=np.float32)
            kpidx = [x in kpnamefull for x in kps_labels].index(True)
            kplabel = kps_labels[kpidx]
            outres['annotation'][kplabel]['location'][idx] = kploc

    valid_idx_dict = {}
    for kps in kps_labels:
        if len(filter_None(outres['annotation'][kps]['location'])) < 3:
            valid_idx_dict[kps] = np.array([])
            continue
        ann_idx = filter_annotation(outres['annotation'][kps]['location'])
        valid_idx_dict[kps] = ann_idx

    for kps in kps_labels:
        valid_idx = valid_idx_dict[kps]
        if valid_idx.size < 3:
            continue
        can_swap, swap_kps = swap_label(kps)
        if not can_swap:
            valid_loc = []
            for idx in valid_idx:
                valid_loc.append(outres['annotation'][kps]['location'][idx])
            outres['annotation'][kps]['mean'] = np.mean(np.stack(valid_loc), axis=0)
        else:
            # check swap
            swap_valid_idx = valid_idx_dict[swap_kps]
            swap_invalid_idx = [x for x in range(n_annotators) if x not in swap_valid_idx]
            locations = outres['annotation'][kps]['location'].copy()
            for idx in swap_invalid_idx:
                if idx not in valid_idx:
                    locations[idx] = outres['annotation'][swap_kps]['location'][idx]
            if len(filter_None(locations)) < 3:
                continue
            _valid_idx = filter_annotation(locations)
            if _valid_idx.size < 3:
                continue
            valid_loc = []
            for idx in _valid_idx:
                valid_loc.append(locations[idx])
            outres['annotation'][kps]['mean'] = np.mean(np.stack(valid_loc), axis=0)




        ## The joint location is the mean
        # valid_location = list(filter(lambda x: x is not None, outres['annotation'][kps]['location']))
        # if len(valid_location) < 3:
        #     continue
        #
        # valid_location = np.stack(valid_location)
        # # mean_location = np.mean(valid_location, axis=0)
        # # dist = np.sqrt(np.sum((valid_location - mean_location[None, :]) ** 2, axis=1))
        # # mean_dist = np.mean(dist)
        # # pass_idx = dist < 2 * mean_dist
        # # valid_location = valid_location[pass_idx]
        #
        # outres['annotation'][kps]['mean'] = np.mean(valid_location, axis=0)


    allresults.append(outres)
    
print('-- total frames to annotate:{}'.format(len(imagenames)))

outname = os.path.join('data', 'kps_annotation.pkl')
with open(outname, 'wb') as f:
    pickle.dump(allresults, f)

def transform_list_to_dict(l):
    d = {}
    for k in l[0].keys():
        d[k] = np.stack([np.squeeze(x[k]) for x in l])
    return d

npzname = os.path.join('data', 'amt_annotation')
data_root = '/local/home/zhqian/sp/data/egocapture'
output_list = []
for result in allresults:
    imgname = result['frame_name']
    rec = imgname.split('/')[0]
    frame_name = imgname.split('/')[2]
    rec_dir = join(data_root, 'hololens_data', f'record_{rec[10:18]}', rec)
    rec_dir = glob.glob(join(rec_dir, '2021*'))[0]
    image_dir = join(rec_dir, 'PV')
    imgname = join(image_dir, frame_name)
    imgname = relpath(imgname, data_root)

    anno = result['annotation']
    keypoints = np.zeros((17, 2))
    for i in range(17):
        label = kps_labels[i]
        loc = anno[label]['mean']
        if loc is not None:
            keypoints[i] = loc
    output_list.append({
        'imgname': imgname,
        'keypoints': keypoints,
    })
output_dict = transform_list_to_dict(output_list)
embed()
np.savez(npzname, **output_dict)










