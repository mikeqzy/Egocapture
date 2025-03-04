import argparse

import numpy as np
from glob import glob
import os
from os.path import join
from tqdm import tqdm, trange
from tqdm.contrib import tenumerate
import pickle
import smplx
import torch
import gc

from IPython import embed

data_root = '/local/home/zhqian/sp/data/egocapture'
model_dir = '/local/home/zhqian/sp/data/smpl'

def get_recs_from_dates(dates):
    recs = []
    for date in dates:
        date_recs = glob(join(data_root, 'hololens_data', f'record_2021{date}', '*'))
        recs.extend([x.split('/')[-1] for x in date_recs])
    return recs

def get_all_recs():
    recs = glob(join(data_root, 'hololens_data', '*', '*'))
    return [x.split('/')[-1] for x in recs]

def get_recs_from_split(train=True):
    if train:
        recs = glob(join(data_root, 'PROX_temp', 'train', '*'))
    else:
        recs = glob(join(data_root, 'PROX_temp', 'test', '*'))
    return [x.split('/')[-1] for x in recs]

def transform_list_to_dict(l):
    d = {}
    for k in l[0].keys():
        d[k] = np.stack([np.squeeze(x[k]) for x in l])
    return d

def generate_kp_npz_file(rec):
    male_model = smplx.create(model_dir, model_type='smplx',
                              gender='male', ext='npz',
                              num_pca_comps=12,
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
                              ).to('cuda')
    female_model = smplx.create(model_dir, model_type='smplx',
                                gender='female', ext='npz',
                                num_pca_comps=12,
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
                                ).to('cuda')

    output_list = []
    rec_dir = join(data_root, 'hololens_data', f'record_{rec[10:18]}', rec)
    rec_dir = glob(join(rec_dir, '2021*'))[0]
    gt_dir = glob(join(data_root, 'PROX_temp', '*', rec, 'body_idx_*', 'results'))[0]
    kp_file = np.load(join(rec_dir, 'keypoints.npz'))
    for idx, imgname in tenumerate(kp_file['imgname']):
        gender = kp_file['gender'][idx]
        frame_dict = {
            'center': kp_file['center'][idx],
            'scale': kp_file['scale'][idx],
            'valid_keypoints': kp_file['keypoints'][idx],
            'imgname': kp_file['imgname'][idx],
            'gender': gender,
        }
        gt_file = join(gt_dir, f'frame_{imgname[-9:-4]}', '000.pkl')
        try:
            with open(gt_file, 'rb') as f:
                gt = pickle.load(f, encoding='latin1')
        except:
            continue
        gt.pop('gender')
        frame_dict.update(gt)
        frame_dict['shape'] = gt['betas']
        frame_dict['pose'] = gt['body_pose']

        torch_param = {}
        skip = False
        for key in gt.keys():
            if key in ['pose_embedding', 'camera_rotation', 'camera_translation']:
                continue
            else:
                try:
                    assert not np.any(np.isnan(gt[key]))
                except AssertionError:
                    embed()
                    skip = True
                    print(f"{kp_file['imgname'][idx]}, nan in {key}")
                torch_param[key] = torch.tensor(gt[key]).to('cuda')
        if skip:
            continue

        if gender == 'm':
            output = male_model(**torch_param)
        else:
            output = female_model(**torch_param)
        joints = output.joints.detach().cpu().numpy().squeeze()

        frame_dict['3d_joints'] = joints
        output_list.append(frame_dict)

        # output_dict = transform_list_to_dict(output_list)
        # np.savez(join(data_root, 'tmp', f'{rec}_{save_name}'), **output_dict)

    output_dict = transform_list_to_dict(output_list)
    np.savez(join(data_root, 'test', f'{rec}.npz'), **output_dict)

def merge_npz_files():
    train_files = glob(join(data_root, 'valid_train', '*'))
    train_files = [np.load(x) for x in train_files]
    train_dict = {k: np.concatenate([x[k] for x in train_files], axis=0) for k in train_files[0].files}
    test_files = glob(join(data_root, 'valid_test', '*'))
    test_files = [np.load(x) for x in test_files]
    test_dict = {k: np.concatenate([x[k] for x in test_files], axis=0) for k in test_files[0].files}
    np.savez(join(data_root, 'egocapture_valid_train.npz'), **train_dict)
    np.savez(join(data_root, 'egocapture_valid_test.npz'), **test_dict)

def generate_valid_npz_files(train=True):
    recs = get_recs_from_split(train)
    for rec in recs:
        output_dict = {}
        output_dict['imgname'] = []
        output_dict['valid'] = []
        gt_dir = glob(join(data_root, 'PROX_temp', '*', rec, 'body_idx_*', 'results'))[0]
        gt_frames = [int(x[-5:]) for x in glob(join(gt_dir, '*'))]

        rec_dir = join(data_root, 'hololens_data', f'record_{rec[10:18]}', rec)
        rec_dir = glob(join(rec_dir, '2021*'))[0]
        valid_file = np.load(join(rec_dir, 'valid_frame.npz'))
        for imgname, valid in zip(valid_file['imgname'], valid_file['valid']):
            frame = int(imgname[-9:-4])
            if frame in gt_frames:
                output_dict['imgname'].append(imgname)
                output_dict['valid'].append(valid)
        np.savez(join(data_root, 'valid_test', f'{rec}.npz'), **output_dict)

def merge_keypoints_file(recs, save_name='egocapture.npz'):
    male_model = smplx.create(model_dir, model_type='smplx',
                              gender='male', ext='npz',
                              num_pca_comps=12,
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
                              ).to('cuda')
    female_model = smplx.create(model_dir, model_type='smplx',
                              gender='female', ext='npz',
                              num_pca_comps=12,
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
                              ).to('cuda')

    output_list = []
    for rec in tqdm(recs):
        rec_dir = join(data_root, 'hololens_data', f'record_{rec[10:18]}', rec)
        rec_dir = glob(join(rec_dir, '2021*'))[0]
        gt_dir = glob(join(data_root, 'PROX_temp', '*', rec, 'body_idx_*', 'results'))[0]
        kp_file = np.load(join(rec_dir, 'keypoints.npz'))
        for idx, imgname in tenumerate(kp_file['imgname']):
            gender = kp_file['gender'][idx]
            frame_dict = {
                'center': kp_file['center'][idx],
                'scale': kp_file['scale'][idx],
                'valid_keypoints': kp_file['keypoints'][idx],
                'imgname': kp_file['imgname'][idx],
                'gender': gender,
            }
            gt_file = join(gt_dir, f'frame_{imgname[-9:-4]}', '000.pkl')
            try:
                with open(gt_file, 'rb') as f:
                    gt = pickle.load(f, encoding='latin1')
            except:
                continue
            gt.pop('gender')
            frame_dict.update(gt)
            frame_dict['shape'] = gt['betas']
            frame_dict['pose'] = gt['body_pose']

            torch_param = {}
            skip = False
            for key in gt.keys():
                if key in ['pose_embedding', 'camera_rotation', 'camera_translation']:
                    continue
                else:
                    try:
                        assert not np.any(np.isnan(gt[key]))
                    except AssertionError:
                        embed()
                        skip = True
                        print(f"{kp_file['imgname'][idx]}, nan in {key}")
                    torch_param[key] = torch.tensor(gt[key]).to('cuda')
            if skip:
                continue

            if gender == 'm':
                output = male_model(**torch_param)
            else:
                output = female_model(**torch_param)
            joints = output.joints.detach().cpu().numpy().squeeze()

            frame_dict['3d_joints'] = joints
            output_list.append(frame_dict)

        # output_dict = transform_list_to_dict(output_list)
        # np.savez(join(data_root, 'tmp', f'{rec}_{save_name}'), **output_dict)
        # output_list = []

    output_dict = transform_list_to_dict(output_list)
    np.savez(join(data_root, save_name), **output_dict)

def get_num_frames(train=True):
    recs = get_recs_from_split(train)
    n_frames = 0
    n_valid_frames = 0
    for rec in recs:
        gt_dir = glob(join(data_root, 'PROX_temp', '*', rec, 'body_idx_*', 'results'))[0]
        gt_frames = [int(x[-5:]) for x in glob(join(gt_dir, '*'))]

        rec_dir = join(data_root, 'hololens_data', f'record_{rec[10:18]}', rec)
        rec_dir = glob(join(rec_dir, '2021*'))[0]
        valid_file = np.load(join(rec_dir, 'valid_frame.npz'))
        for imgname, valid in zip(valid_file['imgname'], valid_file['valid']):
            frame = int(imgname[-9:-4])
            if frame in gt_frames:
                n_frames += 1
                if valid:
                    n_valid_frames += 1
    return n_frames, n_valid_frames

def show_statistics():
    train_frames, train_valid_frames = get_num_frames()
    test_frames, test_valid_frames = get_num_frames(train=False)
    print(f'''
    Frames:
    Train: {train_frames},
    Test: {test_frames},
    Total: {train_frames + test_frames},
    Valid frames:
    Train: {train_valid_frames},
    Test: {test_valid_frames},
    Total: {train_valid_frames + test_valid_frames},
    ''')

def check_align():
    valid_train = np.load(join(data_root, 'egocapture_valid_train.npz'))
    valid_test = np.load(join(data_root, 'egocapture_valid_test.npz'))
    valid_imgname_train = []
    for idx in trange(len(valid_train['imgname'])):
        if valid_train['valid'][idx]:
            valid_imgname_train.append(valid_train['imgname'][idx])

    valid_imgname_test = [valid_test['imgname'][idx]
                          for idx in trange(len(valid_test['imgname']))
                          if valid_test['valid'][idx]]

    kp_train = np.load(join(data_root, 'egocapture_train.npz'))
    kp_test = np.load(join(data_root, 'egocapture_test.npz'))
    kp_imgname_train = kp_train['imgname']
    kp_imgname_test = kp_test['imgname']

    assert valid_imgname_train == kp_imgname_train
    assert valid_imgname_test == kp_imgname_test

if __name__ == '__main__':
    # train_recs = get_recs_from_split()
    # test_recs = get_recs_from_split(train=False)
    # merge_keypoints_file(train_recs, save_name='egocapture_train.npz')
    # merge_keypoints_file(test_recs, save_name='egocapture_test.npz')
    # get_num_frames()
    # rec = 'recording_20211004_s2_05_zixin_ric'
    # generate_kp_npz_file(rec)
    # merge_npz_files()
    show_statistics()
    # generate_valid_npz_files(False)
    # check_align()