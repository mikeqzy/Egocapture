import numpy as np
import pandas as pd
import torch
import pickle
import smplx
import trimesh
from os.path import join
import os
from glob import glob
from tqdm import tqdm

from IPython import embed

data_root = '/local/home/zhqian/sp/data/egocapture'
model_path = '/local/home/zhqian/sp/data/smpl'
smplx_folder = '/local/home/zhqian/sp/data/meshes/smplx/'
smpl_folder = '/local/home/zhqian/sp/data/meshes/smpl/'
device = 'cuda'

from IPython import embed

def generate_smplx_mesh(gt_dir, gender, output_dir):
    body_model = smplx.create(model_path, model_type='smplx',
                              gender=gender, ext='npz',
                              num_pca_comps=12,
                              create_global_orient=True,
                              create_body_pose=True,
                              create_betas=True,
                              create_transl=True
                              ).to(device)
    gt_frames = glob(join(gt_dir, '*'))
    for gt_frame in tqdm(gt_frames):
        frame = gt_frame.split('/')[-1]
        with open(join(gt_frame, '000.pkl'), 'rb') as f:
            param = pickle.load(f)
        torch_param = {}
        for key in param.keys():
            if key in ['pose_embedding', 'camera_rotation', 'camera_translation', 'gender']:
                continue
            else:
                torch_param[key] = torch.tensor(param[key]).to(device)

        output = body_model(return_verts=True, **torch_param)
        vertices = output.vertices.detach().cpu().numpy().squeeze()
        mesh = trimesh.Trimesh(vertices=vertices, faces=body_model.faces,
                               process=False, maintain_order=True)
        mesh.export(join(output_dir, f'{frame}.obj'))



if __name__ == '__main__':
    recs = glob(join(data_root, 'PROXD_temp_second_person_slide', '*'))
    df = pd.read_csv(join(data_root, 'gt_info.csv'))
    script_file = '/local/home/zhqian/sp/code/smplx/transfer.sh'
    recs = recs[1:]
    np.random.shuffle(recs)
    for rec in recs[:5]:
        rec_name = rec.split('/')[-1]
        info = df.loc[df['recording_name'] == rec_name]
        body_idx = info['body_idx_fpv'].iloc[0][0]
        gender = info['body_idx_fpv'].iloc[0][2:]
        gt_dir = join(rec, f'body_idx_{body_idx}', 'results')
        smplx_dir = join(smplx_folder, rec_name)
        os.makedirs(smplx_dir, exist_ok=True)
        smpl_dir = join(smpl_folder, rec_name)
        os.makedirs(smpl_dir, exist_ok=True)
        generate_smplx_mesh(gt_dir, gender, smplx_dir)
        with open(script_file, 'w') as f:
            cmd = f'python -m transfer_model --exp-cfg config_files/smplx2smpl.yaml ' \
                  f'--exp-opts datasets.mesh_folder.data_folder={smplx_dir} ' \
                  f'body_model.gender={gender} output_folder={smpl_dir}\n'
            f.write(cmd)
