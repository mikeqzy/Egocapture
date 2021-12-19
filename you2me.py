import argparse

import numpy as np
import pandas as pd
from glob import glob
import os
from os.path import join, relpath
from tqdm import tqdm, trange
import json
import cv2
from PIL import Image, ImageDraw
import pickle

from IPython import embed

data_root = '/local/home/zhqian/sp/data/you2me'
confidence_thresh = 0.2
valid_joints_thresh = 6
rescale = 1.2

def get_dirs(rec, capture='kinect'):
    if capture == 'kinect':
        img_dir = join(rec, 'synchronized', 'frames')
        gt_dir = join(rec, 'synchronized', 'gt-interactee')
        openpose_dir = join(rec, 'features', 'openpose', 'output_json')
    elif capture == 'cmu':
        img_dir = join(rec, 'synchronized', 'frames')
        gt_dir = join(rec, 'synchronized', 'gt-skeletons')
        openpose_dir = join(rec, 'features', 'openpose', 'output_json')
    else:
        raise ValueError
    return img_dir, gt_dir, openpose_dir

def draw_keypoints(keypoints, img, frame=0, waitTime=0, save_name=None):
    draw = ImageDraw.Draw(img)

    valid = keypoints[:, -1] > confidence_thresh
    valid_keypoints = keypoints[valid][:, :-1]

    for k in range(valid_keypoints.shape[0]):
        draw.ellipse((valid_keypoints[k][0] - 2, valid_keypoints[k][1] - 2,
                      valid_keypoints[k][0] + 2, valid_keypoints[k][1] + 2),
                     fill=(255, 0, 0, 0))

    img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)

    if save_name is None:
        win = f'Keypoints {frame:05d}'
        # cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        # cv2.setWindowProperty(win, cv2.WND_PROP_TOPMOST, 1)
        cv2.imshow(win, img)
        cv2.waitKey(waitTime)
        cv2.destroyWindow(win)
    else:
        cv2.imwrite(save_name, img)

def merge(valid_people):
    if len(valid_people) == 1:
        return valid_people[0]
    keypoints = np.zeros((25, 3))
    for k in range(25):
        idx = np.argmax([x[k,2] for x in valid_people])
        keypoints[k, :] = valid_people[idx][k, :]
    return keypoints

def get_center_scale(keypoints):
    valid = keypoints[:, 2] > confidence_thresh
    valid_keypoints = keypoints[valid][:, :-1]

    center = valid_keypoints.mean(axis=0)
    # center = (valid_keypoints.max(axis=0) + valid_keypoints.min(axis=0)) / 2.
    bbox_size = (valid_keypoints.max(axis=0) - valid_keypoints.min(axis=0)).max()
    scale = bbox_size / 200.0
    scale *= rescale
    return center, scale

def is_valid_frame(valid_people):

    if len(valid_people) == 0:
        return False, None
    # merge keypoints
    keypoints = merge(valid_people)

    face_joints = [0, 15, 16, 17, 18]
    lfoot_joints = [11, 22, 23, 24]
    rfoot_joints = [14, 19, 20, 21]
    body_joints = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13]

    n_valid_joints = 0
    for k in face_joints:
        if keypoints[k, 2] > confidence_thresh:
            n_valid_joints += 1
            break
    for k in lfoot_joints:
        if keypoints[k, 2] > confidence_thresh:
            n_valid_joints += 1
            break
    for k in rfoot_joints:
         if keypoints[k, 2] > confidence_thresh:
            n_valid_joints += 1
            break
    for k in body_joints:
        if keypoints[k, 2] > confidence_thresh:
            n_valid_joints += 1

    if n_valid_joints < valid_joints_thresh:
        return False, None

    return True, keypoints

def transform_list_to_dict(l):
    d = {}
    for k in l[0].keys():
        d[k] = np.stack([np.squeeze(x[k]) for x in l])
    return d

def load_gt(gt_dir, frame, capture='kinect', id=0):
    if capture == 'kinect':
        gt_file = join(gt_dir, f'pose2_{frame}.txt')
        with open(gt_file, 'r') as f:
            gt = f.readline()
        gt = [float(x) for x in gt[:-1].split(' ')]
        gt = np.reshape(np.array(gt), (25, 3))
        kinect2openpose18 = [
            20,
            8,
            9,
            10,
            4,
            5,
            6,
            16,
            17,
            18,
            12,
            13,
            14,
        ]
        out_18 = np.zeros((18, 3))
        out_18[1:14] = gt[kinect2openpose18]
        return out_18
    elif capture == 'cmu':
        gt_file = join(gt_dir, f'body3DScene_{frame}.json')
        with open(gt_file, 'r') as f:
            gt = json.load(f)
        gt = np.reshape(np.array(gt['bodies'][id]['joints19']), (19, 4))
        panoptic2openpose25 = [
            1,
            0,
            9,
            10,
            11,
            3,
            4,
            5,
            2,
            12,
            13,
            14,
            6,
            7,
            8,
            17,
            15,
            18,
            16,
        ]
        out_25 = np.zeros((25, 4))
        out_25[:19] = gt[panoptic2openpose25]
        return out_25
    else:
        raise ValueError

def you2me(capture='kinect'):
    root = join(data_root, capture)
    recs = glob(join(root, '*'))
    recs = ['1-catch1', '2-catch2', '4-convo1', '5-convo2', '6-convo3', '7-convo4',
            '10-hand1', '13-sports1', '14-sports2']
    recs = [join(root, rec) for rec in recs]
    ids = [0, 1, 0, 0, 1, 0, 1, 0, 1]
    output_list = []
    for rec, id in tqdm(zip(recs, ids)):
        print(rec, id)
        img_dir, gt_dir, openpose_dir = get_dirs(rec, capture)
        if not os.path.exists(gt_dir):
            continue
        n_frames = len(glob(join(img_dir, '*')))
        for frame in trange(1, n_frames + 1):
            img_file = join(img_dir, f'imxx{frame}.jpg')
            img_relpath = relpath(img_file, data_root)
            openpose_file = join(openpose_dir, f'imxx{frame}_keypoints.json')

            with open(openpose_file, 'r') as f:
                people = json.load(f)['people']
            valid_detections = [np.reshape(x['pose_keypoints_2d'], (-1, 3))
                                for x in people]
            valid, keypoints = is_valid_frame(valid_detections)

            # img = Image.open(img_file)
            # if valid:
            #     draw_keypoints(keypoints, img.copy())

            if not valid:
                continue
            center, scale = get_center_scale(keypoints)
            try:
                gt = load_gt(gt_dir, frame, capture, id=id)
            except FileNotFoundError:
                continue
            output_list.append({
                'center': center,
                'scale': scale,
                'keypoints': keypoints,
                'imgname': img_relpath,
                '3d_joints': gt,
            })
    output_dict = transform_list_to_dict(output_list)
    np.savez(join(data_root, f'{capture}.npz'), **output_dict)

if __name__ == '__main__':
    # you2me('kinect')
    you2me('cmu')


