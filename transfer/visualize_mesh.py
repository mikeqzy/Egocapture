import numpy as np
import pandas as pd
import json
import pickle
import ast
from glob import glob
import os
from os.path import join, basename, exists
from tqdm import tqdm
import cv2
import pyrender, trimesh
from PIL import Image, ImageDraw
import smplx

from calibration.camera import create_camera
from calibration.utils import *

from IPython import embed

data_root = '/local/home/zhqian/sp/data/egocapture'
mesh_root = '/local/home/zhqian/sp/data/meshes'
smplx_mesh_root = join(mesh_root, 'smplx')
smpl_mesh_root = join(mesh_root, 'smpl')
model_path = '/local/home/zhqian/sp/data/smpl'
resize_scale = 2

device = 'cuda'
body_color = 'white'

def visualize_meshes(rec=None, step=1):
    df = pd.read_csv(join(data_root, 'gt_info.csv'))
    info = df.loc[df['recording_name'] == rec]
    body_idx = int(info['body_idx_fpv'].iloc[0][0])
    gender = info['body_idx_fpv'].iloc[0][2:]

    # gt_dir = join(data_root, 'fit_results', 'PROXD_init_t', rec)
    # openpose_dir = glob(join(data_root, '2021*'))[0]

    fitting_root = glob(join(data_root, 'PROX_temp', '*', rec))[0]
    fitting_dir = join(fitting_root, 'body_idx_{}'.format(body_idx), 'results')

    smplx_dir = join(smplx_mesh_root, rec)
    smpl_dir = join(smpl_mesh_root, rec)


    data_folder = join(data_root, 'hololens_data', f'record_{rec[10:18]}', rec,)
    video_dir = join(data_root, 'video')
    os.makedirs(video_dir, exist_ok=True)
    video_name = join(video_dir, f'{rec}.avi')

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video = cv2.VideoWriter(video_name, fourcc, 30., (960, 1080))


    holo2kinect_dir = join(data_folder, 'cal_trans/holo_to_kinect12.json')
    with open(holo2kinect_dir, 'r') as f:
        trans_holo2kinect = np.array(json.load(f)['trans'])
    trans_kinect2holo = np.linalg.inv(trans_holo2kinect)

    fpv_recording_dir = glob(join(data_folder, '2021*'))[0]
    fpv_color_dir = join(fpv_recording_dir, 'PV')
    # pv_info_path = glob(join(fpv_recording_dir, '*_pv.txt'))[0]
    # with open(pv_info_path) as f:
    #     lines = f.readlines()
    # cx, cy, w, h = ast.literal_eval(lines[0])  # hololens pv camera infomation

    ######### create dir:
    # self.holo_frame_id_dict: key: frame_id, value: pv frame img path
    holo_pv_path_list = glob(join(fpv_color_dir, '*_frame_*.jpg'))
    holo_pv_path_list = sorted(holo_pv_path_list)
    holo_frame_id_list = [basename(x).split('.')[0].split('_', 1)[1] for x in holo_pv_path_list]
    holo_frame_id_dict = dict(zip(holo_frame_id_list, holo_pv_path_list))
    # self.holo_timestamp_dict: key: timestamp, value: frame id
    holo_timestamp_list = [basename(x).split('_')[0] for x in holo_pv_path_list]
    holo_timestamp_dict = dict(zip(holo_timestamp_list, holo_frame_id_list))

    ######### read from processed info
    valid_frame_npz = join(fpv_recording_dir, 'valid_frame.npz')
    kp_npz = join(fpv_recording_dir, 'keypoints.npz')
    valid_frames = np.load(valid_frame_npz)
    holo_annotations = np.load(kp_npz)
    assert len(valid_frames['valid']) == len(valid_frames['imgname'])
    # read info in valid_frame.npz
    holo_frame_id_all = [basename(x).split('.')[0].split('_', 1)[1] for x in valid_frames['imgname']]
    holo_valid_dict = dict(zip(holo_frame_id_all, valid_frames['valid']))  # 'frame_01888': True
    # read info in keypoints.npz
    holo_frame_id_valid = [basename(x).split('.')[0].split('_', 1)[1] for x in
                           holo_annotations['imgname']]  # list of all valid frame names (e.x., 'frame_01888')
    holo_keypoint_dict = dict(zip(holo_frame_id_valid, holo_annotations['keypoints']))

    ######## read hololens camera info
    pv_info_path = glob(join(fpv_recording_dir, '*_pv.txt'))[0]
    with open(pv_info_path) as f:
        lines = f.readlines()
    holo_cx, holo_cy, holo_w, holo_h = ast.literal_eval(lines[0])  # hololens pv camera infomation

    holo_fx_dict = {}
    holo_fy_dict = {}
    holo_pv2world_trans_dict = {}
    for i, frame in enumerate(lines[1:]):
        frame = frame.split((','))
        cur_timestamp = frame[0]  # string
        cur_fx = float(frame[1])
        cur_fy = float(frame[2])
        cur_pv2world_transform = np.array(frame[3:20]).astype(float).reshape((4, 4))

        if cur_timestamp in holo_timestamp_dict.keys():
            cur_frame_id = holo_timestamp_dict[cur_timestamp]
            holo_fx_dict[cur_frame_id] = cur_fx
            holo_fy_dict[cur_frame_id] = cur_fy
            holo_pv2world_trans_dict[cur_frame_id] = cur_pv2world_transform

    # common
    H, W = holo_h, holo_w
    camera_center = np.array([holo_cx, holo_cy])
    camera_pose = np.eye(4)
    # camera_pose = np.array([1.0, -1.0, -1.0, 1.0]).reshape(-1, 1) * camera_pose
    camera_pose = np.array([1.0, 1.0, 1.0, 1.0]).reshape(-1, 1) * camera_pose
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=4.0)

    if body_color == 'pink':
        base_color = (1.0, 193 / 255, 193 / 255, 1.0)
    elif body_color == 'white':
        base_color = (0.7, 0.7, 0.7, 1.0)
        # base_color = (1.0, 1.0, 0.9, 1.0)

    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        alphaMode='OPAQUE',
        # baseColorFactor=(1.0, 193/255, 193/255, 1.0)
        baseColorFactor=base_color,
    )

    for frame_id in tqdm(sorted(os.listdir(fitting_dir))[::step]):
        # print('viz frame {}'.format(frame_id))

        holo_frame_id = frame_id

        smplx_mesh = join(smplx_dir, f'{frame_id}.obj')
        smpl_mesh = join(smpl_dir, f'{frame_id}.obj')
        if not exists(smplx_mesh) or not exists(smpl_mesh):
            continue


        smplx_body = trimesh.load_mesh(smplx_mesh, process=False)
        smpl_body = trimesh.load_mesh(smpl_mesh, process=False)

        if holo_frame_id not in holo_frame_id_dict.keys():
            pass
            # print('{} does not exist')
        else:
            fpv_img_path = holo_frame_id_dict[holo_frame_id]
            # fpv_img_path = join(fpv_recording_dir, 'PV', holo_frame_id_dict[holo_frame_id])
            # pv_timestamp = int(fpv_img_path.split('/')[-1][0:-16])
            cur_fx = holo_fx_dict[holo_frame_id]
            cur_fy = holo_fy_dict[holo_frame_id]
            cur_pv2world_transform = holo_pv2world_trans_dict[holo_frame_id]
            cur_world2pv_transform = np.linalg.inv(cur_pv2world_transform)

            # cur_world2pv_transform[1][3] = cur_world2pv_transform[1][3] - 0.04  # todo

            camera = pyrender.camera.IntrinsicsCamera(
                fx=cur_fx, fy=cur_fy,
                cx=camera_center[0], cy=camera_center[1])

            # project 3d joints to pv img
            holo_is_valid = False
            if holo_frame_id in holo_valid_dict.keys():
                holo_is_valid = holo_valid_dict[holo_frame_id]
            if not holo_is_valid:
                continue

            img = cv2.imread(fpv_img_path)[:, :, ::-1] / 255.0
            def get_projection_img(body):
                body.apply_transform(trans_kinect2holo)  # in hololens world coordinate
                body.apply_transform(cur_world2pv_transform)  # in hololens pv coordinate
                body_mesh = pyrender.Mesh.from_trimesh(body, material=material)

                scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                                       ambient_light=(0.3, 0.3, 0.3))
                scene.add(camera, pose=camera_pose)
                scene.add(light, pose=camera_pose)
                scene.add(body_mesh, 'mesh')

                r = pyrender.OffscreenRenderer(viewport_width=W,
                                               viewport_height=H,
                                               point_size=1.0)

                color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
                color = color.astype(np.float32) / 255.0
                alpha = 0.6  # todo: set transparency
                color[:, :, -1] = color[:, :, -1] * alpha  # alpha=0.5
                color = Image.fromarray((color * 255).astype(np.uint8))
                # color.show()
                input_img = Image.fromarray((img * 255).astype(np.uint8))
                input_img.paste(color, (0, 0), color)

                input_img.convert('RGB')
                input_img = input_img.resize((int(W / 2), int(H / 2)))

                cv2_img = cv2.cvtColor(np.array(input_img), cv2.COLOR_RGB2BGR)
                return cv2_img

            smplx_img = get_projection_img(smplx_body)
            smpl_img = get_projection_img(smpl_body)
            concat_img = cv2.vconcat([smplx_img, smpl_img])
            video.write(concat_img)


    cv2.destroyAllWindows()
    video.release()

if __name__ == '__main__':
    # recs = [
    #     'recording_20210910_s2_01_ines_moh',
    #         'recording_20210910_s2_05_moh_ines',
    #         'recording_20210929_s2_01_ines_sara',
    #         'recording_20210929_s2_02_sara_ines',
    #         'recording_20210929_s2_05_ines_sara']
    recs = ['recording_20210921_s2_02_zixin_ines']
    for rec in recs:
        visualize_meshes(rec, step=1)