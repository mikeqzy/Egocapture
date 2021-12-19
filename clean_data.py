'''
Usage: python clean_data.py --recordings recording_20210925_s1_05_siwei_lam
'''

import argparse

import numpy as np
import pandas as pd
from glob import glob
import os
from os.path import join
from tqdm import tqdm, trange
import json
import cv2
from PIL import Image, ImageDraw
import pickle

from IPython import embed

# define path and constants
data_root = '/local/home/zhqian/sp/data/egocapture'
confidence_thresh = 0.2
rescale = 1.2
valid_joints_thresh = 6
resize_scale = 2

# Define command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--recordings', nargs='+', default=[], help='Recordings to be processed')
parser.add_argument('--visualize', dest='visualize', action='store_true')
parser.add_argument('--resume', dest='resume', action='store_true')

def merge(valid_people):
    if len(valid_people) == 1:
        return valid_people[0]
    keypoints = np.zeros((25, 3))
    for k in range(25):
        idx = np.argmax([x[k,2] for x in valid_people])
        keypoints[k, :] = valid_people[idx][k, :]
    return keypoints

def draw_keypoints(keypoints, img, frame=0, waitTime=0, save_name=None):
    draw = ImageDraw.Draw(img)

    valid = keypoints[:, -1] > confidence_thresh
    valid_keypoints = keypoints[valid][:, :-1]

    for k in range(valid_keypoints.shape[0]):
        draw.ellipse((valid_keypoints[k][0] / resize_scale - 4, valid_keypoints[k][1] / resize_scale - 4,
                      valid_keypoints[k][0] / resize_scale + 4, valid_keypoints[k][1] / resize_scale + 4),
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

def get_center_scale(keypoints):
    valid = keypoints[:, 2] > confidence_thresh
    valid_keypoints = keypoints[valid][:, :-1]

    center = valid_keypoints.mean(axis=0)
    # center = (valid_keypoints.max(axis=0) + valid_keypoints.min(axis=0)) / 2.
    bbox_size = (valid_keypoints.max(axis=0) - valid_keypoints.min(axis=0)).max()
    scale = bbox_size / 200.0
    scale *= rescale
    return center, scale

def transform_list_to_dict(l):
    d = {}
    for k in l[0].keys():
        d[k] = np.stack([np.squeeze(x[k]) for x in l])
    return d

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

def get_root_path(rec):
    rec_dir = join(data_root, 'hololens_data', f'record_{rec[10:18]}', rec)
    rec_dir = glob(join(rec_dir, '2021*'))[0]
    image_dir = join(rec_dir, 'PV')
    openpose_dir = join(rec_dir, 'keypoints')
    return image_dir, openpose_dir, rec_dir

def load_data(rec_dir, output_list, valid_list, start, end):
    keypoints = np.load(join(rec_dir, 'keypoints.npz'))
    valid = np.load(join(rec_dir, 'valid_frame.npz'))
    keypoints_frame_id = keypoints['imgname']
    valid_frame_id = valid['imgname']
    keypoints_frame_id = [int(x.split('/')[-1][-9:-4]) for x in keypoints_frame_id]
    valid_frame_id = [int(x.split('/')[-1][-9:-4]) for x in valid_frame_id]
    for frame in range(start, end):
        try:
            valid_idx = valid_frame_id.index(frame)
        except ValueError:
            continue
        valid_list[frame - start] = {
            'imgname': valid['imgname'][valid_idx],
            'valid': valid['valid'][valid_idx],
        }
        if valid['valid'][valid_idx]:
            keypoints_idx = keypoints_frame_id.index(frame)
            output_list[frame - start] = {
                'center': keypoints['center'][keypoints_idx],
                'scale': keypoints['scale'][keypoints_idx],
                'keypoints': keypoints['keypoints'][keypoints_idx],
                'imgname': keypoints['imgname'][keypoints_idx],
                'gender': keypoints['gender'][keypoints_idx],
            }


def clean_data(recordings=None, resume=False):
    for rec in recordings:
        image_dir, openpose_dir, rec_dir = get_root_path(rec)

        # get frame range and gender for the sequence
        df = pd.read_csv(join(data_root, 'gt_info.csv'))
        info = df.loc[df['recording_name'] == rec]
        gender = info['body_idx_fpv'].iloc[0][2]
        starts = np.array(info['start'])
        ends = np.array(info['end'])

        output_lists = []
        valid_lists = []

        for start, end in zip(starts, ends):

            print(f'Start processing, Start: {start}, End: {end}')

            output_list = [None] * (end - start)
            valid_list = [None] * (end - start)
            if resume:
                load_data(rec_dir, output_list, valid_list, start, end)
                embed()

            auto_end_frame = -1
            frame = start
            while frame < end:
                try:
                    image_file = glob(join(image_dir, f'*frame_{frame:05d}.jpg'))[0]
                except IndexError:
                    frame += 1
                    if frame >= auto_end_frame:
                        auto_end_frame = -1
                    continue
                openpose_file = join(openpose_dir, image_file.split('/')[-1][:-4] + '_keypoints.json')

                valid_detections = []
                auto = frame < auto_end_frame

                rel_image_path = os.path.relpath(image_file, data_root)
                img = Image.open(image_file)
                img = img.resize((img.size[0] // resize_scale, img.size[1] // resize_scale))
                if auto:
                    # Auto mode
                    with open(openpose_file, 'r') as f:
                        people = json.load(f)['people']
                    print(f'Automatically processing frame {frame:05d}, {len(people)} people detected')
                    valid_detections = [np.reshape(x['pose_keypoints_2d'], (-1, 3))
                                    for x in people]
                else:
                    # Manual mode

                    # interactively check if the openpose detected person is the target
                    with open(openpose_file, 'r') as f:
                        people = json.load(f)['people']
                    print(f"{len(people)} people detected in frame {frame:05d}")

                    frame_change = -1

                    for idx, person in enumerate(people):
                        keypoints = person['pose_keypoints_2d']
                        keypoints = np.reshape(np.array(keypoints), (-1, 3))

                        draw_keypoints(keypoints, img.copy(), frame)

                        action = input(f'Frame {frame}, Person {idx + 1}/{len(people)}, Action: ')

                        '''
                        Possible actions:
                        'y' or '':  Confirm that the shown person is the target
                        'n': Reject the shown person
                        'b' or 'b {n_frames}': Go back one frame or given number of frames
                        'g {frame_id}': Goto given frame
                        'auto {end_frame_id}': Auto mode, only use it when you are sure that only the target is present 
                        in the frame, automatically accepts every detected person in frames (current_frame_id:end_frame_id)
                        !!! You can't really be sure because openpose could detect some keypoints from nowhere... So use
                        it wisely and look carefully at the images
                        '''

                        try:
                            if action == '' or action == 'y':
                                valid_detections.append(keypoints)
                            elif action == 'n' or action == 'a':
                                continue
                            elif action == 'b':
                                frame_change = frame - 1
                                break
                            elif action[0] == 'b':
                                frame_change = frame - int(action[2:])
                                break
                            elif action[0] =='g':
                                frame_change = int(action[2:])
                                break
                            elif action[:4] == 'auto':
                                auto_end_frame = min(int(action[5:]) + 1, end)
                                frame_change = frame
                                break
                            else:
                                print("Wrong action")
                                frame_change = frame
                                break
                        except:
                            print("Wrong action")
                            frame_change = frame
                            break

                    if frame_change >= 0:
                        frame = frame_change
                        continue

                valid, keypoints = is_valid_frame(valid_detections)

                if valid:
                    center, scale = get_center_scale(keypoints)
                    output_list[frame - start] = {
                        'center': center,
                        'scale': scale,
                        'keypoints': keypoints,
                        'imgname': rel_image_path,
                        'gender': gender,
                    }
                    if auto:
                        draw_keypoints(keypoints, img.copy(), frame, 400)
                else:
                    output_list[frame - start] = None

                valid_list[frame - start] = {
                    'imgname': rel_image_path,
                    'valid': valid
                }
                # gt_file = join(gt_dir, f'frame_{frame:05d}.pkl')
                # with open(gt_file, 'rb') as f:
                #     gt = pickle.load(f, encoding='latin1')
                # frame_dict.update(gt)


                frame += 1
                if frame >= auto_end_frame:
                    auto_end_frame = -1

            output_lists.extend(output_list)
            valid_lists.extend(valid_list)

        output_lists = list(filter(lambda x: False if x is None else True, output_lists))
        valid_lists = list(filter(lambda x: False if x is None else True, valid_lists))

        output_dict = transform_list_to_dict(output_lists)
        valid_dict = transform_list_to_dict(valid_lists)
        np.savez(join(rec_dir, f'keypoints.npz'), **output_dict)
        np.savez(join(rec_dir, f'valid_frame.npz'), **valid_dict)
        embed()

def filter_data(recs):
    for rec in recs:
        image_dir, openpose_dir, rec_dir = get_root_path(rec)

        kp_file = np.load(join(rec_dir, 'keypoints.npz'))
        n_frames = len(kp_file['imgname'])
        use_for_fitting = [True] * n_frames
        auto_end_idx = -1

        start = 0
        end = n_frames
        idx = start
        while idx < end:
            imgname = kp_file['imgname'][idx]
            frame = int(imgname[-9:-4])
            if idx >= auto_end_idx:
                auto_end_idx = -1
            image_file = join(data_root, imgname)
            keypoints = kp_file['keypoints'][idx]

            img = Image.open(image_file)
            img = img.resize((img.size[0] // resize_scale, img.size[1] // resize_scale))

            auto = idx < auto_end_idx
            if auto:
                print(f'Index {idx}/{n_frames - 1}')
                draw_keypoints(keypoints, img.copy(), idx, 200)
            else:
                draw_keypoints(keypoints, img.copy(), idx)
                idx_change = -1
                while True:
                    action = input(f'Index {idx + 1}/{n_frames}, Action:')
                    try:
                        if action == '':
                            break
                        elif action == 'b':
                            idx_change = idx - 1
                            break
                        elif action[0] == 'b':
                            idx_change = idx - int(action[2:])
                            break
                        elif action[0] == 'g':
                            idx_change = int(action[2:])
                            break
                        elif action[:4] == 'auto':
                            auto_end_idx = min(int(action[5:]) + 1, end)
                            idx_change = idx
                            break
                        elif action[0] == 'e':
                            embed()
                        else:
                            print('Wrong action')
                    except:
                        print('Wrong action')

                if idx_change >= 0:
                    idx = idx_change
                    continue

            idx += 1

def get_recs_from_dates(dates):
    recs = []
    for date in dates:
        date_recs = glob(join(data_root, 'hololens_data', f'record_2021{date}', '*'))
        recs.extend([x.split('/')[-1] for x in date_recs])
    return recs

def save_keypoint_images():
    dates = ['0911', '0918', '0910', '0921', '0923']
    recs = get_recs_from_dates(dates)
    for rec in tqdm(recs):
        image_dir, openpose_dir, rec_dir = get_root_path(rec)

        kp_file = np.load(join(rec_dir, 'keypoints.npz'))
        n_frames = len(kp_file['imgname'])
        for idx in trange(n_frames):
            imgname = kp_file['imgname'][idx]
            image_file = join(data_root, imgname)
            keypoints = kp_file['keypoints'][idx]

            img = Image.open(image_file)
            img = img.resize((img.size[0] // resize_scale, img.size[1] // resize_scale))

            save_dir = join(data_root, 'kp_vis', rec)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_name = join(save_dir, f'{idx:05d}.jpg')

            draw_keypoints(keypoints, img.copy(), save_name=save_name)


if __name__ == '__main__':
    args = parser.parse_args()
    # clean_data(recordings=args.recordings, resume=args.resume)
    # filter_data(args.recordings)
    save_keypoint_images()