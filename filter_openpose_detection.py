import ast

import numpy as np
from glob import glob
import os
from os.path import join
import json
import argparse

import pandas as pd
from tqdm import tqdm
import smplx

from IPython import embed

data_root = '/local/home/zhqian/sp/data/egocapture'
smpl_root = '/local/home/zhqian/sp/data/smpl'

parser = argparse.ArgumentParser()
parser.add_argument('--recordings', nargs='+', default=[], help='Recordings to be processed')

def get_all_recording_name():
    recs = glob(join(data_root, 'hololens_data', '*', '*'))
    return [x.split('/')[-1] for x in recs]


def filter_openpose_detection(recordings):
    for rec in tqdm(recordings):
        rec_dir = join(data_root, 'hololens_data', f'record_{rec[10:18]}', rec)
        pv_dir = glob(join(rec_dir, '2021*'))[0]
        cal_dir = join(rec_dir, 'cal_trans')
        gt_dir = join(data_root, 'gt', rec) # todo: change path

        df = pd.read_csv(join(data_root, 'gt_info.csv'))
        info = df.loc[df['recording_name'] == rec]
        gender = info['body_idx_fpv'].iloc[0][2]
        gender = 'male' if gender == 'm' else 'female'

        holo2kinect_file = join(cal_dir, 'holo_to_kinect12.json')
        with open(holo2kinect_file, 'r') as f:
            trans_holo2kinect = np.array(json.load(f)['trans'])
        trans_kinect2holo = np.linalg.inv(trans_holo2kinect)

        pv_info_file = glob(join(pv_dir, '*_pv.txt'))[0]
        with open(pv_info_file) as f:
            lines = f.readlines()
        cx, cy, w, h = ast.literal_eval(lines[0])

        pv_fx_dict = {}
        pv_fy_dict = {}
        pv2world_transform_dict = {}
        for i, frame in enumerate(lines[1:]):
            frame = frame.split((','))
            cur_timestamp = int(frame[0])
            cur_fx = float(frame[1])
            cur_fy = float(frame[2])
            cur_pv2world_transform = np.array(frame[3:20]).astype(float).reshape((4, 4))

            pv_fx_dict[cur_timestamp] = cur_fx
            pv_fy_dict[cur_timestamp] = cur_fy
            pv2world_transform_dict[cur_timestamp] = cur_pv2world_transform

        body_model = smplx.create(smpl_root, model_type='smplx',
                                  gender=gender, ext='npz',
                                  num_pca_comps=args.num_pca_comps,
                                  create_global_orient=True,
                                  create_body_pose=True,
                                  create_betas=True,
                                  create_left_hand_pose=True,
                                  create_right_hand_pose=True,
                                  create_expression=True,
                                  create_jaw_pose=True,
                                  create_leye_pose=True,
                                  create_reye_pose=True,
                                  create_transl=True
                                  )




if __name__ == '__main__':
    args = parser.parse_args()
    recordings = args.recordings if len(args.recordings) > 0 else get_all_recording_name()
    filter_openpose_detection(recordings)