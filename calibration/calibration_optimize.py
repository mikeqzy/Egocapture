import torch
import torchgeometry as tgm
import pytorch3d.transforms as ptf
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from torch.optim import Adam, SGD, LBFGS
import torch.nn.functional as F

import numpy as np
import pandas as pd
import os
from os.path import join, basename
from glob import glob
from tqdm import tqdm
import json
import pickle
import smplx
import ast

from misc_utils import JointMapper, smpl_to_openpose
from camera import create_camera
from utils import *

from IPython import embed

# aligned keypoints between smplx and openpose, used to get paired 3d/2d joints
align_keypoints = [0, 2, 3, 4, 5, 6, 7, 10, 11, 13, 14, 15, 16, 17, 18]

data_root = '/local/home/zhqian/sp/data/calibration'
smpl_root = '/local/home/zhqian/sp/data/smpl'
confidence_thresh = 0.2

# dataset per sequence (only optimize the extrinsics between kinect and hololens)
def build_dataset(rec):
    df = pd.read_csv(join(data_root, 'gt_info.csv'))
    info = df.loc[df['recording_name'] == rec]
    body_idx = int(info['body_idx_fpv'].iloc[0][0])
    gender = info['body_idx_fpv'].iloc[0][2:]

    fitting_root = join(data_root, 'fit_results', 'PROXD_init_t', rec)
    fitting_dir = join(fitting_root, 'body_idx_{}'.format(body_idx), 'results')
    data_folder = join(data_root, rec)
    rendering_dir = join(data_folder, 'fpv_render_imgs_new')

    holo2kinect_dir = join(data_folder, 'cal_trans/holo_to_kinect12.json')
    with open(holo2kinect_dir, 'r') as f:
        trans_holo2kinect = np.array(json.load(f)['trans'])
    trans_kinect2holo = np.linalg.inv(trans_holo2kinect)

    fpv_recording_dir = glob(join(data_folder, '2021*'))[0]
    fpv_color_dir = join(fpv_recording_dir, 'PV')

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

    joint_mapper = JointMapper(smpl_to_openpose('smplx', use_hands=False, use_face=False))
    body_model = smplx.create(smpl_root, model_type='smplx',
                              joint_mapper=joint_mapper,
                              gender=gender, ext='npz',
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
                              )

    T1 = [] # kinect to holo global
    T2 = [] # holo global to per frame
    K = [] # intrinsics
    X = [] # 3d joint
    x = [] # 2d joint
    frames = np.load(join(data_folder, 'frames.npy'))
    for frame in tqdm(frames):
        frame_id = f'frame_{frame:05d}'

        with open(join(fitting_dir, frame_id, '000.pkl'), 'rb') as f:
            param = pickle.load(f)
        torch_param = {}
        for key in param.keys():
            if key in ['pose_embedding', 'camera_rotation', 'camera_translation', 'gender']:
                continue
            else:
                torch_param[key] = torch.tensor(param[key])

        output = body_model(**torch_param)
        joints = output.joints.detach().cpu().numpy().squeeze()

        cur_fx = holo_fx_dict[frame_id]
        cur_fy = holo_fy_dict[frame_id]
        cur_pv2world_transform = holo_pv2world_trans_dict[frame_id]
        cur_world2pv_transform = np.linalg.inv(cur_pv2world_transform)

        keypoints_holo = holo_keypoint_dict[frame_id]

        joints = joints[align_keypoints]
        keypoints_holo = keypoints_holo[align_keypoints]

        valid = keypoints_holo[:, -1] > confidence_thresh
        valid_keypoints = keypoints_holo[valid][:, :-1]
        valid_joints = joints[valid]
        n_valid = valid.sum()

        T1.append(np.tile(trans_kinect2holo, (n_valid, 1, 1,)))
        T2.append(np.tile(cur_world2pv_transform, (n_valid, 1, 1)))
        K.append(np.tile(np.array([cur_fx, cur_fy, holo_cx, holo_cy]), (n_valid, 1)))
        X.append(valid_joints)
        x.append(valid_keypoints)
        # embed()

    T1 = torch.from_numpy(np.concatenate(T1, axis=0))
    T2 = torch.from_numpy(np.concatenate(T2, axis=0))
    K = torch.from_numpy(np.concatenate(K, axis=0))
    X = torch.from_numpy(np.concatenate(X, axis=0))
    x = torch.from_numpy(np.concatenate(x, axis=0))

    assert T1.size(0) == T2.size(0) == K.size(0) == X.size(0) == x.size(0)
    np.savez(join(data_folder, 'calibration.npz'),
             T1=T1, T2=T2, K=K, X=X, x=x)


def load_dataset(rec):
    data_folder = join(data_root, rec)
    data = np.load(join(data_folder, 'calibration.npz'))
    T1, T2, K, X, x = data['T1'], data['T2'], data['K'], data['X'], data['x']
    T1 = torch.from_numpy(T1).float()
    T2 = torch.from_numpy(T2).float()
    K = torch.from_numpy(K).float()
    X = torch.from_numpy(X).float()
    x = torch.from_numpy(x).float()
    dataset = TensorDataset(T1, T2, K, X, x)
    train_len = int(0.8 * len(dataset))
    valid_len = len(dataset) - train_len
    return random_split(dataset, [train_len, valid_len], torch.Generator().manual_seed(42))

class KinectProjectionHololens(nn.Module):
    def __init__(self, T1, device, rep='aa'):
        super(KinectProjectionHololens, self).__init__()
        self.rep = rep
        self.T1 = T1

        # aa = tgm.rotation_matrix_to_angle_axis(T1[None, :3]).contiguous()
        # self.aa = nn.Parameter(aa, requires_grad=True)

        if self.rep == 'aa':
            aa = tgm.rotation_matrix_to_angle_axis(T1[None, :3]).contiguous()
            self.aa = nn.Parameter(aa, requires_grad=True)
        elif self.rep == 'quaternion':
            quaternion = ptf.matrix_to_quaternion(T1[None, :3, :3]).contiguous()
            self.quaternion = nn.Parameter(quaternion, requires_grad=True)
        elif self.rep == 'rot6d':
            rot6d = T1[:3,:2].reshape((-1, 6))
            self.rot6d = nn.Parameter(rot6d, requires_grad=True)
        else:
            raise ValueError

        trans = T1[:3, 3]
        self.trans = nn.Parameter(trans, requires_grad=True)

        self.device = device

    def compute_transformation(self):
        if self.rep == 'aa':
            T1p = tgm.angle_axis_to_rotation_matrix(self.aa)  # [1, 4, 4]
        elif self.rep == 'quaternion':
            qw = self.quaternion[...,0]
            qx = self.quaternion[...,1]
            qy = self.quaternion[...,2]
            qz = self.quaternion[...,3]

            x2 = qx + qx
            y2 = qy + qy
            z2 = qz + qz
            xx = qx * x2
            yy = qy * y2
            wx = qw * x2
            xy = qx * y2
            yz = qy * z2
            wy = qw * y2
            xz = qx * z2
            zz = qz * z2
            wz = qw * z2

            m = torch.zeros((1, 4, 4)).to(self.device)
            m[..., 0, 0] = 1.0 - (yy + zz)
            m[..., 0, 1] = xy - wz
            m[..., 0, 2] = xz + wy
            m[..., 1, 0] = xy + wz
            m[..., 1, 1] = 1.0 - (xx + zz)
            m[..., 1, 2] = yz - wx
            m[..., 2, 0] = xz - wy
            m[..., 2, 1] = yz + wx
            m[..., 2, 2] = 1.0 - (xx + yy)
            m[..., 3, 3] = 1.0

            T1p = m
        elif self.rep == 'rot6d':
            reshaped_input = self.rot6d.view(-1, 3, 2)
            b1 = F.normalize(reshaped_input[:,:,0], dim=1)
            dot_prod = torch.sum(b1 * reshaped_input[:,:,1], dim=1, keepdim=True)
            b2 = F.normalize(reshaped_input[:,:,1] - dot_prod * b1, dim=-1)
            b3 = torch.cross(b1, b2, dim=1)
            T1p = torch.zeros(1, 4, 4).to(self.device)
            T1p[:, :3, :3] = torch.stack([b1, b2, b3], dim=-1)
            T1p[:, 3, 3] = 1.0
        else:
            raise ValueError

        # T1p = tgm.angle_axis_to_rotation_matrix(self.aa)  # [1, 4, 4]

        T1p[:, :3, 3] = self.trans
        # embed()
        return T1p

    def forward(self, data):
        _, T2, K, X, x = data
        B = T2.shape[0]
        T1p = self.compute_transformation()
        T1p = T1p.repeat((B, 1, 1))
        X = torch.cat((X, torch.ones(B, 1).to(self.device)), dim=1).unsqueeze(-1)
        X = torch.bmm(T1p, X)
        X = torch.bmm(T2, X)
        T3 = torch.tensor([[[1.0, 0, 0, 0],
                            [0, -1, 0, 0],
                            [0, 0, -1, 0],
                            [0, 0, 0, 1]]]).repeat((B, 1, 1)).to(self.device)
        X = torch.bmm(T3, X)

        camera_holo = create_camera(camera_type='persp_holo',
                                    focal_length_x=K[:, 0],
                                    focal_length_y=K[:, 1],
                                    center=K[:, 2:],
                                    batch_size=B).to(device=self.device)
        X = X[:,:3,0].unsqueeze(1) # [B, 1, 3]
        xp = camera_holo(X).squeeze() # [B, 2]
        return xp, T1p

    def compute_loss(self, data, valid=False):
        data = [x.to(self.device) for x in data]
        w, h = 1920, 1080
        if valid:
            lbd = 0.
        else:
            lbd = 0.01

        x, T1 = data[-1], data[0]
        xp, T1p = self(data)
        proj_error = xp - x
        proj_error[:,0] /= w
        proj_error[:,1] /= h
        proj_loss = (proj_error ** 2).mean()
        reg_loss = ((T1p - T1) ** 2).mean()
        if valid and (torch.isnan(proj_loss) or proj_loss > 100.):
            print(f'Bad loss {proj_loss}')
            embed()
        return proj_loss + lbd * reg_loss


def optimize_extrinsics(rec, rep='aa', optim='adam', save=True):
    device = 'cuda'
    batch_size = 32
    n_epoch = 80

    log_file = join(data_root, 'train_log.pkl')
    with open(log_file, 'rb') as f:
        log = pickle.load(f)

    train_dataset, valid_dataset = load_dataset(rec)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

    T1 = train_dataset[0][0]
    model = KinectProjectionHololens(T1, device, rep).to(device)
    if optim == 'adam':
        optimizer = Adam(model.parameters(), lr=1e-4)
    elif optim == 'lbfgs':
        optimizer = LBFGS(model.parameters(), lr=1e-2, line_search_fn='strong_wolfe',
                          # tolerance_grad=1e-9, tolerance_change=1e-12
                          )
    else:
        raise ValueError

    initial_error = 0.
    with torch.no_grad():
        model.eval()
        errors = []
        for data in valid_loader:
            errors.append(model.compute_loss(data, valid=True).item())
        initial_error = np.mean(errors) * 1e4
        print(f"Original error: {initial_error}")

    best_error = 100
    best_epoch = -1
    best_T1 = None
    error_list = []

    for epoch in range(n_epoch):
        # pbar = tqdm(train_loader)
        pbar = train_loader
        model.train()
        for data in pbar:
            def closure():
                optimizer.zero_grad()
                loss = model.compute_loss(data)
                loss.backward()
                return loss
            if optim == 'adam':
                closure()
                optimizer.step()
            elif optim == 'lbfgs':
                optimizer.step(closure)
            else:
                raise ValueError

            # pbar.set_postfix({'loss': loss.item()})

        with torch.no_grad():
            model.eval()
            errors = []
            for data in valid_loader:
                errors.append(model.compute_loss(data, valid=True).item())
            error = np.mean(errors) * 1e4
            print(f"Error after epoch {epoch + 1}: {error}")
            error_list.append(error)
            if error < best_error:
                best_error = error
                best_epoch = epoch + 1
                best_T1 = model.compute_transformation()[0].cpu().detach().numpy()

    print(f"Initial error: {initial_error}\n"
          f"Best error: {best_error}, in Epoch {best_epoch}")

    # save result
    if save:
        save_name = join(data_root, rec, 'cal_trans', 'holo_to_kinect12_optimize.json')
        T1 = best_T1
        T1inv = np.linalg.inv(T1)
        save_dict = {"trans": T1inv.tolist()}
        with open(save_name, 'w') as f:
            json.dump(save_dict, f)

    exp_name = f'{rep}_{optim}'
    log_dict = log.get(exp_name, dict())
    log_dict[rec] = error_list
    log[exp_name] = log_dict
    with open(log_file, 'wb') as f:
        pickle.dump(log, f)

if __name__ == '__main__':
    recs = ['recording_20210910_s2_01_ines_moh',
            'recording_20210910_s2_05_moh_ines',
            'recording_20210929_s2_01_ines_sara',
            'recording_20210929_s2_02_sara_ines',
            'recording_20210929_s2_05_ines_sara']

    # recs = ['recording_20210929_s2_05_ines_sara']
    # for rec in recs:
    #     build_dataset(rec)

    # for rec in recs:
    #     optimize_extrinsics(rec, rep='rot6d', save=False, optim='lbfgs')

    # optims = ['adam', 'lbfgs']
    optims = ['lbfgs']
    reps = ['aa', 'quaternion', 'rot6d']
    for rec in recs:
        for rep in reps:
            for optim in optims:
                optimize_extrinsics(rec, rep=rep, save=False, optim=optim)
