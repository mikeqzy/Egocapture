import numpy as np
import torch
import pickle
import smplx
import open3d as o3d
from os.path import join
from glob import glob
from tqdm import tqdm, trange

from IPython import embed

model_path = '/local/home/zhqian/sp/data/smpl/models'
data_root = '/local/home/zhqian/sp/data/HPS/hps_smpl'
device = 'cuda'

import matplotlib.pyplot as plt

def draw_keypoints(joints):
    joints = joints.cpu().numpy()
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(joints[:,0], joints[:,1], joints[:,2])
    ax.axis('equal')
    plt.show()
    plt.close()

def o3d_visualization(joints):
    joints = joints.cpu().numpy()
    joint_pointcloud = o3d.geometry.PointCloud()
    joint_pointcloud.points = o3d.utility.Vector3dVector(joints)

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([mesh_frame, joint_pointcloud])

def process_hps_motion():
    smpl_model = smplx.create(model_path, model_type='smpl').to(device)

    recs = glob(join(data_root, '*.pkl'))
    output_list = []
    for rec in tqdm(recs):
        with open(rec, 'rb') as f:
            data = pickle.load(f)
        n_frames = data['transes'].shape[0]
        n_seq = n_frames // 100
        n_frames = n_seq * 100
        joint_list = []
        for frame in trange(n_frames):
            transl = torch.from_numpy(data['transes'][frame])[None].float().to(device)
            pose = torch.from_numpy(data['poses'][frame]).float()
            global_orient = pose[None, :3].to(device)
            body_pose = pose[None, 3:].to(device)
            joints = smpl_model(transl=transl, global_orient=global_orient, body_pose=body_pose).joints
            joints = joints[:, :22].detach()
            joint_list.append(joints)

        joint_list = torch.cat(joint_list, dim=0)
        joint_list = joint_list.reshape((n_seq, 100, 22, 3))

        for seq in trange(n_seq):
            joints_seq = joint_list[seq]

            # translate to pelvis of first frame
            joints_seq -= joints_seq[0, 0, :].clone()

            # rotate according to frame 0 axis
            joints_frame0 = joints_seq[0]
            x_axis = joints_frame0[2, :] - joints_frame0[1, :]  # [3]
            x_axis[-1] = 0
            x_axis = x_axis / torch.norm(x_axis)
            z_axis = torch.tensor([0, 0, 1]).float().to(device)
            y_axis = torch.cross(z_axis, x_axis)
            y_axis = y_axis / torch.norm(y_axis)
            transf_rotmat = torch.stack([x_axis, y_axis, z_axis], dim=1)  # [3, 3]
            joints_seq = torch.matmul(joints_seq, transf_rotmat)

            joint_list[seq] = joints_seq
            # embed()

        output_list.append(joint_list)

    embed()
    output_list = torch.cat(output_list, dim=0).cpu().numpy()
    np.save(join(data_root, '..', 'hps.npy'), output_list)



if __name__ == '__main__':
    process_hps_motion()