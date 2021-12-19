import copy
import json
import numpy as np
import os
import os.path as osp
import PIL.Image as pil_img
from PIL import ImageDraw
from tqdm import tqdm
import cv2
import argparse

parser = argparse.ArgumentParser(description='Vis openpose 2D joints.')
parser.add_argument("--recording_root", default='/mnt/hdd/egocaptures/record_20210923')  # 'C:/Users/siwei/Desktop/record_20210907'
parser.add_argument("--recording_name", default='recording_20210923_s2_03_giulia_siwei')

# set start/end frame (start/end frame = 10/1000: from frame_00010.jpg to frame_01000.jpg), only need for keypoints_folder_name='keypoints'
parser.add_argument("--start_frame", default=1101, type=int)
parser.add_argument("--end_frame", default=3200, type=int)

parser.add_argument("--cur_view", default='sub_1', choices=['master', 'sub_1', 'sub_2'])
parser.add_argument("--keypoints_folder_name", default='keypoints_reorder_clean', choices=['keypoints', 'keypoints_reorder', 'keypoints_reorder_clean'])


parser.add_argument("--scale", default=2, type=float)  # scale img size by 2 (smaller)


args = parser.parse_args()

if __name__ == '__main__':
    xxx = osp.join(args.recording_root, args.recording_name, args.cur_view, args.keypoints_folder_name)
    keypoint_names = [name for name in os.listdir(osp.join(args.recording_root, args.recording_name, args.cur_view, args.keypoints_folder_name)) if name.endswith('_keypoints.json')]
    keypoint_names = sorted(keypoint_names)
    first_frame_id = int(keypoint_names[0][6:11])
    if args.keypoints_folder_name == 'keypoints':
        keypoint_names = keypoint_names[args.start_frame - first_frame_id: args.end_frame - first_frame_id + 1]
        print('start_frame: ', keypoint_names[0])
        print('end_frame: ', keypoint_names[-1])
        # print('{} to {}.'.format(keypoint_names[args.start_frame-1], keypoint_names[args.end_frame-1]))

    # visualize 2d joints
    hand_joint_idx = [2, 4, 5, 8, 9, 12, 13, 16, 17, 20]  # vis 2 joints for each finger (end / tip)

    # for cur_view in ['master', 'sub_1', 'sub_2']:
    for cur_view in [args.cur_view]:
        print('process view {}...'.format(cur_view))
        output_img_path = osp.join(args.recording_root, args.recording_name, cur_view, '{}_img'.format(args.keypoints_folder_name))
        if not osp.exists(output_img_path):
            os.mkdir(output_img_path)

        # process for each frame
        # for keypoint_name in tqdm(keypoint_names[args.start_frame-1: args.end_frame]):
        for keypoint_name in tqdm(keypoint_names):
            keypoint_fn = osp.join(args.recording_root, args.recording_name, cur_view, args.keypoints_folder_name, keypoint_name)
            with open(keypoint_fn) as keypoint_file:
                data = json.load(keypoint_file)   # data: dict, key: version/people
                data_reorder = copy.deepcopy(data)

            # read color img
            img_fn = osp.join(args.recording_root, args.recording_name, cur_view, 'color_img', keypoint_name[0:11]+'.jpg')
            cur_img = cv2.imread(img_fn)
            cur_img = cur_img[:, :, ::-1]
            cur_img = cv2.resize(src=cur_img, dsize=(int(1920/args.scale), int(1080/args.scale)), interpolation=cv2.INTER_AREA)  # resolution/4

            num_people = len(data['people'])
            if num_people != 2:
                print('{} people detected for {}!'.format(len(data['people']), keypoint_name[0:11]))

            #################### visualize original detection from openpose
            # output_img = np.ones([480, 270, 3])
            # output_img = (output_img * 255).astype(np.uint8)
            output_img = pil_img.fromarray(cur_img)
            draw = ImageDraw.Draw(output_img)

            cur_frame_body_keypoints = []
            cur_frame_hand_keypoints = []
            for idx, cur_data in enumerate(data['people']):
                body_keypoint = np.array(cur_data['pose_keypoints_2d'], dtype=np.float32).reshape([-1, 3])  # [25, 3]
                lhand_keypoint = np.array(cur_data['hand_left_keypoints_2d'], dtype=np.float32).reshape([-1, 3])  # [25, 3]
                rhand_keypoint = np.array(cur_data['hand_right_keypoints_2d'], dtype=np.float32).reshape([-1, 3])  # [25, 3]
                hand_keypoint = np.concatenate([lhand_keypoint[hand_joint_idx, :], rhand_keypoint[hand_joint_idx, :]], axis=0)  # [20, 3]
                cur_frame_body_keypoints.append(body_keypoint)
                cur_frame_hand_keypoints.append(hand_keypoint)

            if num_people >= 1:
                for k in range(len(cur_frame_body_keypoints[0])):
                    draw.ellipse((cur_frame_body_keypoints[0][k][0] / args.scale - 2, cur_frame_body_keypoints[0][k][1] / args.scale - 2,
                                  cur_frame_body_keypoints[0][k][0] / args.scale + 2, cur_frame_body_keypoints[0][k][1] / args.scale + 2),
                                 fill=(255, 0, 0, 0))  # red: idx 0 in openpose detection
                for k in range(len(cur_frame_hand_keypoints[0])):
                    draw.ellipse((cur_frame_hand_keypoints[0][k][0] / args.scale - 1, cur_frame_hand_keypoints[0][k][1] / args.scale - 1,
                                  cur_frame_hand_keypoints[0][k][0] / args.scale + 1, cur_frame_hand_keypoints[0][k][1] / args.scale + 1),
                                 fill=(255, 0, 0, 0))  # red: idx 0 in openpose detection
            if num_people >= 2:
                for k in range(len(cur_frame_body_keypoints[0])):
                    draw.ellipse((cur_frame_body_keypoints[1][k][0] / args.scale - 2, cur_frame_body_keypoints[1][k][1] / args.scale - 2,
                                      cur_frame_body_keypoints[1][k][0] / args.scale + 2, cur_frame_body_keypoints[1][k][1] / args.scale + 2),
                                     fill=(0, 0, 255, 0))  # blue: idx 1
                for k in range(len(cur_frame_hand_keypoints[0])):
                    draw.ellipse((cur_frame_hand_keypoints[1][k][0] / args.scale - 1, cur_frame_hand_keypoints[1][k][1] / args.scale - 1,
                                      cur_frame_hand_keypoints[1][k][0] / args.scale + 1, cur_frame_hand_keypoints[1][k][1] / args.scale + 1),
                                     fill=(0, 0, 255, 0))  # blue: idx 1
            if num_people >= 3:
                for k in range(len(cur_frame_body_keypoints[0])):
                    draw.ellipse((cur_frame_body_keypoints[2][k][0] / args.scale - 2, cur_frame_body_keypoints[2][k][1] / args.scale - 2,
                                      cur_frame_body_keypoints[2][k][0] / args.scale + 2, cur_frame_body_keypoints[2][k][1] / args.scale + 2),
                                     fill=(0, 255, 0, 0))  # green: idx 2
                for k in range(len(cur_frame_hand_keypoints[0])):
                    draw.ellipse((cur_frame_hand_keypoints[2][k][0] / args.scale - 1, cur_frame_hand_keypoints[2][k][1] / args.scale - 1,
                                      cur_frame_hand_keypoints[2][k][0] / args.scale + 1, cur_frame_hand_keypoints[2][k][1] / args.scale + 1),
                                     fill=(0, 255, 0, 0))  # green: idx 2
            if num_people == 4:
                for k in range(len(cur_frame_body_keypoints[0])):
                    draw.ellipse((cur_frame_body_keypoints[3][k][0] / args.scale - 2, cur_frame_body_keypoints[3][k][1] / args.scale - 2,
                                      cur_frame_body_keypoints[3][k][0] / args.scale + 2, cur_frame_body_keypoints[3][k][1] / args.scale + 2),
                                     fill=(255, 0, 255, 0))  # purple: idx 3
                for k in range(len(cur_frame_hand_keypoints[0])):
                    draw.ellipse((cur_frame_hand_keypoints[3][k][0] / args.scale - 1, cur_frame_hand_keypoints[3][k][1] / args.scale - 1,
                                      cur_frame_hand_keypoints[3][k][0] / args.scale + 1, cur_frame_hand_keypoints[3][k][1] / args.scale + 1),
                                     fill=(255, 0, 255, 0))  # purple: idx 3

            save_path = osp.join(output_img_path, keypoint_name[0:11] + '.jpg')
            output_img.save(save_path)

