import smplx
import torch
import numpy as np
import pickle
import os
from my_prox.misc_utils import *
from tqdm import tqdm
import json
from loader.eval_loader_amass import AMASS_Loader
from utils.utils import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    smplx_model_path = '/mnt/hdd/PROX/body_models/smplx_model'
    amass_dir = '/local/home/szhang/AMASS/amass'

    # smplx_model_path = '/cluster/scratch/szhang/PROX/body_models/smplx_model'
    # amass_dir = '/cluster/scratch/szhang/AMASS/amass'

    smplx_model = smplx.create(smplx_model_path, model_type='smplx',
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
                               create_transl=True,
                               batch_size=1).to(device)
    print('[INFO] smplx model loaded.')

    with open('loader/SSM2.json') as f:  # todo: my_SSM2?
        marker_ids = list(json.load(f)['markersets'][0]['indices'].values())


    ################################ load prox estimated data ##################################
    # est_params_root = '/mnt/hdd/PROX'
    # est_params_root = '/local/home/szhang/temp_prox/fit_results_dct_adam/PROXD'
    # est_params_root = '/local/home/szhang/temp_prox/fit_results_15217_S3_adam_infill_5_contact_0_2/PROXD'
    est_params_root = '/local/home/szhang/temp_prox/fit_results_15217_S3_adam_infill_2_contact_0_1_fric_1_1/PROXD'
    prox_params_root = '/mnt/hdd/PROX/PROXD'

    cam2world_dir = '/mnt/hdd/PROX/cam2world'

    # est_params_root = '/cluster/scratch/szhang/fit_results_velL2_1e5/PROXD'

    prox_seq_path = os.listdir(est_params_root)
    prox_seq_path.sort()

    marker_list = []
    joint_list = []
    # for seq_name in tqdm(prox_seq_path[0:1]):
    for seq_name in tqdm(prox_seq_path):
        frame_list = os.listdir(os.path.join(est_params_root, seq_name, 'results'))
        frame_list.sort()
        frame_total = len(frame_list)

        # todo: read cam2world
        scene_name = seq_name.split('_')[0]
        with open(os.path.join(cam2world_dir, scene_name + '.json'), 'r') as f:
            cam2world = np.array(json.load(f))
            cam2world = torch.from_numpy(cam2world).float().to(device)

        # for each sequence in prox:
        for cur_frame_name in frame_list:
            cur_prox_params_dir = os.path.join(est_params_root, seq_name, 'results', cur_frame_name, '000.pkl')
            # cur_prox_params_dir = os.path.join(prox_params_root, seq_name, 'results', cur_frame_name, '000.pkl')  # todo: prox dataset
            body_params_dict = read_prox_pkl(cur_prox_params_dir)
            for param_name in body_params_dict:
                body_params_dict[param_name] = np.expand_dims(body_params_dict[param_name], axis=0)
                body_params_dict[param_name] = torch.from_numpy(body_params_dict[param_name]).to(device)

            smplx_output = smplx_model(return_verts=True, **body_params_dict)  # generated human body mesh
            joints = smplx_output.joints[:, 0:25, :]  # [1, 25, 3]
            body_verts = smplx_output.vertices
            markers = body_verts[:, marker_ids, :]  # [1, 67, 3]

            # to world coordinate
            cam_R = cam2world[:3, :3].reshape([3, 3])
            cam_t = cam2world[:3, 3].reshape([1, 3])
            markers = torch.matmul(cam_R, markers.permute(0, 2, 1)).permute(0, 2, 1) + cam_t
            joints = torch.matmul(cam_R, joints.permute(0, 2, 1)).permute(0, 2, 1) + cam_t


            marker_list.append(markers)
            joint_list.append(joints)


    marker_list = torch.cat(marker_list, dim=0)  # [total_frame, 67, 3]
    marker_list = marker_list.reshape([-1, 100, marker_list.shape[-2], 3])  # [n, 100, 67, 3], n clips in total
    joint_list = torch.cat(joint_list, dim=0)  # [n, 100, 25, 3]
    joint_list = joint_list.reshape([-1, 100, joint_list.shape[-2], 3])  # [n, 100, 25, 3]
    print('total clips from prox:', len(marker_list))

    # normalize motion clips
    for i in range(len(marker_list)):
        joints_cur_clip = joint_list[i]
        marker_cur_clip = marker_list[i]


        ##### transfrom: the first frame: pelvis at origin, face y axis
        joints_frame0 = joints_cur_clip[0].detach()  # [N, 3] joints of first frame
        x_axis = joints_frame0[2, :] - joints_frame0[1, :]  # [3]
        x_axis[-1] = 0
        x_axis = x_axis / torch.norm(x_axis)
        z_axis = torch.tensor([0, 0, 1]).float().to(device)
        y_axis = torch.cross(z_axis, x_axis)
        y_axis = y_axis / torch.norm(y_axis)
        transf_rotmat = torch.stack([x_axis, y_axis, z_axis], dim=1)  # [3, 3]
        joints_cur_clip = torch.matmul(joints_cur_clip - joints_frame0[0], transf_rotmat)  # [T(/bs), 25, 3]
        marker_cur_clip = torch.matmul(marker_cur_clip - joints_frame0[0], transf_rotmat)  # [T(/bs), 67, 3]

        joint_list[i] = joints_cur_clip
        marker_list[i] = marker_cur_clip



    #### position
    marker_list = marker_list.reshape(marker_list.shape[0], 100, -1).detach().cpu().numpy()  # [n, 100, 67*3]
    marker_fft = np.fft.fft(marker_list, axis=1)  # [n, 100, d]
    marker_fft = np.abs(marker_fft) ** 2
    # marker_fft = marker_fft.mean(axis=-1).mean(axis=0)  # [100]
    # marker_fft = (marker_fft + 1e-8) / (marker_fft + 1e-8).sum()
    marker_fft = marker_fft.mean(axis=0) + 1e-8
    marker_fft = marker_fft / marker_fft.sum(axis=0, keepdims=True)

    joint_list = joint_list.reshape(joint_list.shape[0], 100, -1).detach().cpu().numpy()  # [n, 100, 25*3]
    joint_fft = np.fft.fft(joint_list, axis=1)
    joint_fft = np.abs(joint_fft) ** 2
    # joint_fft = joint_fft.mean(axis=-1).mean(axis=0)   # [100]
    # joint_fft = (joint_fft + 1e-8) / (joint_fft + 1e-8).sum()
    joint_fft = joint_fft.mean(axis=0) + 1e-8
    joint_fft = joint_fft / joint_fft.sum(axis=0, keepdims=True)


    #### velocity
    marker_list_vel = marker_list[:, 1:] - marker_list[:, 0:-1]
    marker_fft_vel = np.fft.fft(marker_list_vel, axis=1)  # [n, 99, d]
    marker_fft_vel = np.abs(marker_fft_vel) ** 2
    # marker_fft_vel = marker_fft_vel.mean(axis=-1).mean(axis=0)  # [99]
    # marker_fft_vel = (marker_fft_vel + 1e-8) / (marker_fft_vel + 1e-8).sum()
    marker_fft_vel = marker_fft_vel.mean(axis=0) + 1e-8
    marker_fft_vel = marker_fft_vel / marker_fft_vel.sum(axis=0, keepdims=True)

    joint_list_vel = joint_list[:, 1:] - joint_list[:, 0:-1]
    joint_fft_vel = np.fft.fft(joint_list_vel, axis=1)
    joint_fft_vel = np.abs(joint_fft_vel) ** 2
    # joint_fft_vel = joint_fft_vel.mean(axis=-1).mean(axis=0)   # [99]
    # joint_fft_vel = (joint_fft_vel + 1e-8) / (joint_fft_vel + 1e-8).sum()
    joint_fft_vel = joint_fft_vel.mean(axis=0) + 1e-8
    joint_fft_vel = joint_fft_vel / joint_fft_vel.sum(axis=0, keepdims=True)

    #### accel
    marker_list_acc = marker_list_vel[:, 1:] - marker_list_vel[:, 0:-1]
    marker_fft_acc = np.fft.fft(marker_list_acc, axis=1)  # [n, 99, d]
    marker_fft_acc = np.abs(marker_fft_acc) ** 2
    # marker_fft_acc = marker_fft_acc.mean(axis=-1).mean(axis=0)  # [99]
    # marker_fft_acc = (marker_fft_acc + 1e-8) / (marker_fft_acc + 1e-8).sum()
    marker_fft_acc = marker_fft_acc.mean(axis=0) + 1e-8
    marker_fft_acc = marker_fft_acc / marker_fft_acc.sum(axis=0, keepdims=True)

    joint_list_acc = joint_list_vel[:, 1:] - joint_list_vel[:, 0:-1]
    joint_fft_acc = np.fft.fft(joint_list_acc, axis=1)
    joint_fft_acc = np.abs(joint_fft_acc) ** 2
    # joint_fft_acc = joint_fft_acc.mean(axis=-1).mean(axis=0)   # [99]
    # joint_fft_acc = (joint_fft_acc + 1e-8) / (joint_fft_acc + 1e-8).sum()
    joint_fft_acc = joint_fft_acc.mean(axis=0) + 1e-8
    joint_fft_acc = joint_fft_acc / joint_fft_acc.sum(axis=0, keepdims=True)


    ################################## load AMASS data ##############################
    amass_train_datasets = ['HumanEva', 'MPI_HDM05', 'MPI_mosh', 'Transitions_mocap',
                            'ACCAD', 'BMLhandball', 'BMLmovi', 'BioMotionLab_NTroje', 'CMU',
                            'DFaust_67', 'Eyes_Japan_Dataset', 'MPI_Limits']
    # amass_train_datasets = ['HumanEva', 'MPI_HDM05', 'MPI_mosh', 'Transitions_mocap']
    dataset = AMASS_Loader(clip_len=100)
    dataset.read_data(amass_train_datasets, amass_dir)
    joint_list_amass, marker_list_amass = dataset.create_body_repr(smplx_model_path=smplx_model_path)

    #### position
    marker_list_amass = np.reshape(marker_list_amass, [marker_list_amass.shape[0], 100, -1])  # [n2, 100, 67*3]
    marker_amass_fft = np.fft.fft(marker_list_amass, axis=1)
    marker_amass_fft = np.abs(marker_amass_fft) ** 2
    # marker_amass_fft = marker_amass_fft.mean(axis=-1).mean(axis=0)   # [100]
    # marker_amass_fft = (marker_amass_fft + 1e-8) / (marker_amass_fft + 1e-8).sum()
    marker_amass_fft = marker_amass_fft.mean(axis=0) + 1e-8
    marker_amass_fft = marker_amass_fft / marker_amass_fft.sum(axis=0, keepdims=True)


    joint_list_amass = np.reshape(joint_list_amass, [joint_list_amass.shape[0], 100, -1])  # [n2, 100, 25*3]
    joint_amass_fft = np.fft.fft(joint_list_amass, axis=1)
    joint_amass_fft = np.abs(joint_amass_fft) ** 2
    # joint_amass_fft = joint_amass_fft.mean(axis=-1).mean(axis=0)   # [100]
    # joint_amass_fft = (joint_amass_fft + 1e-8) / (joint_amass_fft + 1e-8).sum()
    joint_amass_fft = joint_amass_fft.mean(axis=0) + 1e-8
    joint_amass_fft = joint_amass_fft / joint_amass_fft.sum(axis=0, keepdims=True)


    #### velocity
    marker_list_amass_vel = marker_list_amass[:, 1:] - marker_list_amass[:, 0:-1]
    marker_amass_fft_vel = np.fft.fft(marker_list_amass_vel, axis=1)
    marker_amass_fft_vel = np.abs(marker_amass_fft_vel) ** 2
    # marker_amass_fft_vel = marker_amass_fft_vel.mean(axis=-1).mean(axis=0)   # [100]
    # marker_amass_fft_vel = (marker_amass_fft_vel + 1e-8) / (marker_amass_fft_vel + 1e-8).sum()
    marker_amass_fft_vel = marker_amass_fft_vel.mean(axis=0) + 1e-8
    marker_amass_fft_vel = marker_amass_fft_vel / marker_amass_fft_vel.sum(axis=0, keepdims=True)


    joint_list_amass_vel = joint_list_amass[:, 1:] - joint_list_amass[:, 0:-1]
    joint_amass_fft_vel = np.fft.fft(joint_list_amass_vel, axis=1)
    joint_amass_fft_vel = np.abs(joint_amass_fft_vel) ** 2
    # joint_amass_fft_vel = joint_amass_fft_vel.mean(axis=-1).mean(axis=0)   # [100]
    # joint_amass_fft_vel = (joint_amass_fft_vel + 1e-8) / (joint_amass_fft_vel + 1e-8).sum()
    joint_amass_fft_vel = joint_amass_fft_vel.mean(axis=0) + 1e-8
    joint_amass_fft_vel = joint_amass_fft_vel / joint_amass_fft_vel.sum(axis=0, keepdims=True)

    #### acc
    marker_list_amass_acc = marker_list_amass_vel[:, 1:] - marker_list_amass_vel[:, 0:-1]
    marker_amass_fft_acc = np.fft.fft(marker_list_amass_acc, axis=1)
    marker_amass_fft_acc = np.abs(marker_amass_fft_acc) ** 2
    # marker_amass_fft_acc = marker_amass_fft_acc.mean(axis=-1).mean(axis=0)   # [100]
    # marker_amass_fft_acc = (marker_amass_fft_acc + 1e-8) / (marker_amass_fft_acc + 1e-8).sum()
    marker_amass_fft_acc = marker_amass_fft_acc.mean(axis=0) + 1e-8
    marker_amass_fft_acc = marker_amass_fft_acc / marker_amass_fft_acc.sum(axis=0, keepdims=True)

    joint_list_amass_acc = joint_list_amass_vel[:, 1:] - joint_list_amass_vel[:, 0:-1]
    joint_amass_fft_acc = np.fft.fft(joint_list_amass_acc, axis=1)
    joint_amass_fft_acc = np.abs(joint_amass_fft_acc) ** 2
    # joint_amass_fft_acc = joint_amass_fft_acc.mean(axis=-1).mean(axis=0)   # [100]
    # joint_amass_fft_acc = (joint_amass_fft_acc + 1e-8) / (joint_amass_fft_acc + 1e-8).sum()
    joint_amass_fft_acc = joint_amass_fft_acc.mean(axis=0) + 1e-8
    joint_amass_fft_acc = joint_amass_fft_acc / joint_amass_fft_acc.sum(axis=0, keepdims=True)


    ############# compute KL divergence
    # kl_dist_marker_1 = (marker_fft * np.log(marker_fft / marker_amass_fft)).mean()
    # kl_dist_marker_2 = (marker_amass_fft * np.log(marker_amass_fft / marker_fft)).mean()
    #
    # kl_dist_joint_1 = (joint_fft * np.log(joint_fft / joint_amass_fft)).mean()
    # kl_dist_joint_2 = (joint_amass_fft * np.log(joint_amass_fft / joint_fft)).mean()
    #
    # kl_dist_marker_vel_1 = (marker_fft_vel * np.log(marker_fft_vel / marker_amass_fft_vel)).mean()
    # kl_dist_marker_vel_2 = (marker_amass_fft_vel * np.log(marker_amass_fft_vel / marker_fft_vel)).mean()
    #
    # kl_dist_joint_vel_1 = (joint_fft_vel * np.log(joint_fft_vel / joint_amass_fft_vel)).mean()
    # kl_dist_joint_vel_2 = (joint_amass_fft_vel * np.log(joint_amass_fft_vel / joint_fft_vel)).mean()
    #
    # kl_dist_marker_acc_1 = (marker_fft_acc * np.log(marker_fft_acc / marker_amass_fft_acc)).mean()
    # kl_dist_marker_acc_2 = (marker_amass_fft_acc * np.log(marker_amass_fft_acc / marker_fft_acc)).mean()
    #
    # kl_dist_joint_acc_1 = (joint_fft_acc * np.log(joint_fft_acc / joint_amass_fft_acc)).mean()
    # kl_dist_joint_acc_2 = (joint_amass_fft_acc * np.log(joint_amass_fft_acc / joint_fft_acc)).mean()

    kl_dist_marker_1 = (marker_fft * np.log(marker_fft / marker_amass_fft)).sum(axis=0).mean()
    kl_dist_marker_2 = (marker_amass_fft * np.log(marker_amass_fft / marker_fft)).sum(axis=0).mean()

    kl_dist_joint_1 = (joint_fft * np.log(joint_fft / joint_amass_fft)).sum(axis=0).mean()
    kl_dist_joint_2 = (joint_amass_fft * np.log(joint_amass_fft / joint_fft)).sum(axis=0).mean()

    kl_dist_marker_vel_1 = (marker_fft_vel * np.log(marker_fft_vel / marker_amass_fft_vel)).sum(axis=0).mean()
    kl_dist_marker_vel_2 = (marker_amass_fft_vel * np.log(marker_amass_fft_vel / marker_fft_vel)).sum(axis=0).mean()

    kl_dist_joint_vel_1 = (joint_fft_vel * np.log(joint_fft_vel / joint_amass_fft_vel)).sum(axis=0).mean()
    kl_dist_joint_vel_2 = (joint_amass_fft_vel * np.log(joint_amass_fft_vel / joint_fft_vel)).sum(axis=0).mean()

    kl_dist_marker_acc_1 = (marker_fft_acc * np.log(marker_fft_acc / marker_amass_fft_acc)).sum(axis=0).mean()
    kl_dist_marker_acc_2 = (marker_amass_fft_acc * np.log(marker_amass_fft_acc / marker_fft_acc)).sum(axis=0).mean()

    kl_dist_joint_acc_1 = (joint_fft_acc * np.log(joint_fft_acc / joint_amass_fft_acc)).sum(axis=0).mean()
    kl_dist_joint_acc_2 = (joint_amass_fft_acc * np.log(joint_amass_fft_acc / joint_fft_acc)).sum(axis=0).mean()


    print(kl_dist_marker_1, kl_dist_marker_2, kl_dist_joint_1, kl_dist_joint_2)
    print(kl_dist_marker_vel_1, kl_dist_marker_vel_2, kl_dist_joint_vel_1, kl_dist_joint_vel_2)
    print(kl_dist_marker_acc_1, kl_dist_marker_acc_2, kl_dist_joint_acc_1, kl_dist_joint_acc_2)


if __name__ == '__main__':
    main()















