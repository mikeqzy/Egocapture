import os
import os.path as osp
import cv2
import numpy as np
import json
import trimesh
import argparse
# os.environ["PYOPENGL_PLATFORM"] = "egl"
# os.environ["PYOPENGL_PLATFORM"] = "osmesa"
import pyrender
import PIL.Image as pil_img
import pickle
import smplx
import torch
import glob
import ast
import copy
import open3d as o3d
from utils import *

def main(args):
    recording_root = '/mnt/hdd/egocaptures/record_20211004'
    # recording_20210911_s1_01_moh_lam / recording_20211004_s1_02_max_mohit / recording_20210907_s2_01_siwei_zhiyin
    # recording_20211002_s1_02_mert_carlo
    recording_name = 'recording_20211004_s1_02_max_mohit'
    data_folder = os.path.join(recording_root, recording_name)

    # fitting_root = '/mnt/hdd/egocaptures/fit_results/PROXD_init_t/{}'.format(recording_name)
    fitting_root = 'fit_results/PROXD_init_t_undistort_depth_3view/{}'.format(recording_name)
    body_index_name = 'body_idx_0'  # todo: which body idx?
    gender = 'male'  # todo
    fitting_dir = osp.join(fitting_root, body_index_name, 'results')

    fpv_recording_dir = glob.glob(os.path.join(data_folder, '2021*'))[0]
    fpv_color_dir = os.path.join(fpv_recording_dir, 'PV')
    # img_name_list = glob.glob(os.path.join(fpv_color_dir, '*_frame_*.jpg'))

    read_fitting = True

    # read kinect info
    color_calib_path = 'kinect_cam_params/kinect_12/Color.json'
    depth_calib_path = 'kinect_cam_params/kinect_12/IR.json'
    with open(color_calib_path) as calib_file:
        color_cam_main = json.load(calib_file)
    with open(depth_calib_path) as calib_file:
        depth_cam_main = json.load(calib_file)



    # scene_name = 'seminar_h52' # todo
    # scene_dir = os.path.join('../scene_mesh', scene_name)

    # cam2world_dir = os.path.join(data_folder, 'cal_trans/kinect12_to_world')
    # with open(os.path.join(cam2world_dir, scene_name + '.json'), 'r') as f:
    #     trans_main_to_world = np.array(json.load(f)['trans'])
    # trans_world_to_main = np.linalg.inv(trans_main_to_world)

    holo2kinect_dir = os.path.join(data_folder, 'cal_trans/holo_to_kinect12.json')
    with open(holo2kinect_dir, 'r') as f:
        trans_holo2kinect = np.array(json.load(f)['trans'])
    trans_kinect2holo = np.linalg.inv(trans_holo2kinect)

    pv_info_path = glob.glob(os.path.join(fpv_recording_dir, '*_pv.txt'))[0]
    with open(pv_info_path) as f:
        lines = f.readlines()
    cx, cy, w, h = ast.literal_eval(lines[0])  # hololens pv camera infomation

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


    # pkl_files_dir = osp.join(fitting_dir, 'results')
    # scene_name = recording_name.split("_")[0]
    # base_dir = args.base_dir
    # cam2world_dir = osp.join(base_dir, 'cam2world')
    # scene_dir = osp.join(base_dir, 'scenes')
    # recording_dir = osp.join(base_dir, 'recordings', recording_name)
    # color_dir = os.path.join(recording_dir, 'Color')
    # meshes_dir = os.path.join(fitting_dir, 'meshes')
    rendering_dir = os.path.join(data_folder, 'fpv_render_imgs')

    body_model = smplx.create(args.model_folder, model_type='smplx',
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

    # if args.rendering_mode == '3d' or args.rendering_mode == 'both':
    #     static_scene = trimesh.load(osp.join(scene_dir, scene_name + '.obj'), enable_post_processing=True, print_progress=True)
    #     static_scene.apply_transform(trans_world_to_main)  # in kinect main coord
    #

        # body_scene_rendering_dir = os.path.join(fitting_dir, 'renderings')
        # if not osp.exists(body_scene_rendering_dir):
        #     os.mkdir(body_scene_rendering_dir)

    if args.rendering_mode == 'body' or args.rendering_mode == 'both':
        if not osp.exists(rendering_dir):
            os.mkdir(rendering_dir)

    #common
    H, W = h, w
    camera_center = np.array([cx, cy])
    camera_pose = np.eye(4)
    # camera_pose = np.array([1.0, -1.0, -1.0, 1.0]).reshape(-1, 1) * camera_pose
    camera_pose = np.array([1.0, 1.0, 1.0, 1.0]).reshape(-1, 1) * camera_pose
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=4.0)

    if args.body_color == 'pink':
        base_color = (1.0, 193/255, 193/255, 1.0)
    elif args.body_color == 'white':
        base_color = (0.7, 0.7, 0.7, 1.0)
        # base_color = (1.0, 1.0, 0.9, 1.0)

    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        alphaMode='OPAQUE',
        # baseColorFactor=(1.0, 193/255, 193/255, 1.0)
        baseColorFactor=base_color
        )

    for img_name in sorted(os.listdir(fitting_dir))[args.start::args.step]:
        print('viz frame {}'.format(img_name))
        img_main_path = '{}/{}/master/color_img/{}.jpg'.format(recording_root, recording_name, img_name)
        depth_main_path = '{}/{}/master/depth_img/{}.png'.format(recording_root, recording_name, img_name)
        fpv_img_path = glob.glob(os.path.join(fpv_color_dir, '*_{}.jpg').format(img_name))

        # img_name = '01133'
        # img_main_path = '{}/{}/master/color_img/frame_{}.jpg'.format(recording_root, recording_name, img_name)
        # depth_main_path = '{}/{}/master/depth_img/frame_{}.png'.format(recording_root, recording_name, img_name)
        # fpv_img_path = [os.path.join(fpv_color_dir, '132758379300040454.jpg')]

        ############ read lemo fitted body
        if read_fitting:
            with open(osp.join(fitting_dir, img_name, '000.pkl'), 'rb') as f:
                param = pickle.load(f)
            torch_param = {}
            for key in param.keys():
                if key in ['pose_embedding', 'camera_rotation', 'camera_translation', 'gender']:
                    continue
                else:
                    torch_param[key] = torch.tensor(param[key])

            output = body_model(return_verts=True, **torch_param)
            vertices = output.vertices.detach().cpu().numpy().squeeze()
            body = trimesh.Trimesh(vertices, body_model.faces, process=False)  # in main kinect color cam coord

            body_o3d = o3d.geometry.TriangleMesh()
            body_o3d.vertices = o3d.utility.Vector3dVector(vertices)
            body_o3d.triangles = o3d.utility.Vector3iVector(body_model.faces)
            body_o3d.compute_vertex_normals()

        ############### open kinect img
        img_main = cv2.imread(img_main_path).astype(np.float32)[:, :, ::-1] / 255.0  # [1080, 1920, 3]
        depth_im_main = cv2.imread(depth_main_path, flags=-1).astype(float)  # [576, 640]
        depth_im_main = depth_im_main / 8.
        depth_im_main = depth_im_main * 0.001  # mm->m
        depth_im_main[depth_im_main >= 3.86] = 0

        default_color = [1.00, 1.00, 0.80]

        ############# get main kinect point cloud in main color coord
        points_depth_coord_main = unproject_depth_image(depth_im_main, depth_cam_main).reshape(-1, 3)  # point cloud from depth map in depth cam coord [576*640, 3]
        colors_main = np.tile(default_color, [points_depth_coord_main.shape[0], 1])
        colors_main_default = copy.deepcopy(colors_main)
        points_color_coord_main = points_coord_trans(points_depth_coord_main, np.asarray(depth_cam_main['ext_depth2color']))  # point cloud from depth map in color cam coord

        ############# get valid points and colors on main kienct color cam 2D plane
        valid_idx_main, uvs_main = get_valid_idx(points_color_coord_main, color_cam_main)
        points_color_coord_main = points_color_coord_main[valid_idx_main]
        colors_main[valid_idx_main == True, :3] = img_main[uvs_main[:, 1], uvs_main[:, 0]]
        colors_main = colors_main[valid_idx_main]
        colors_main_default = colors_main_default[valid_idx_main]

        # pcd_main = o3d.geometry.PointCloud()
        # pcd_main.points = o3d.utility.Vector3dVector(points_color_coord_main)
        # pcd_main.colors = o3d.utility.Vector3dVector(colors_main)
        # o3d.visualization.draw_geometries([pcd_main, body_o3d])

        ############ kinect main depth point cloud in holo world coord
        points_color_coord_holo_world = points_coord_trans(points_color_coord_main, trans_kinect2holo)
        if read_fitting:
            body_o3d.transform(trans_kinect2holo)  # fitting in holo world coord

        # pcd_main_holo_world_coord = o3d.geometry.PointCloud()
        # pcd_main_holo_world_coord.points = o3d.utility.Vector3dVector(points_color_coord_holo_world)
        # pcd_main_holo_world_coord.colors = o3d.utility.Vector3dVector(colors_main)
        # o3d.visualization.draw_geometries([pcd_main_holo_world_coord, body_o3d])


        ############## read hololens img and world2pv trans
        if len(fpv_img_path) == 0:
            print('{} does not exist')
        else:
            fpv_img_path = fpv_img_path[0]
            if read_fitting:
                pv_timestamp = int(fpv_img_path.split('/')[-1][0:-16])
            else:
                pv_timestamp = int(fpv_img_path.split('/')[-1][0:-4])
            cur_fx = pv_fx_dict[pv_timestamp]
            cur_fy = pv_fy_dict[pv_timestamp]
            cur_pv2world_transform = pv2world_transform_dict[pv_timestamp]
            cur_world2pv_transform = np.linalg.inv(cur_pv2world_transform)

            add_trans = np.array([[1.0, 0, 0, 0],
                              [0, -1, 0, 0],
                              [0, 0, -1, 0],
                              [0, 0, 0, 1]])

            ############# transform kinect main point cloud to holo pv coord (for current frame)
            points_color_coord_holo_pv = points_coord_trans(points_color_coord_holo_world, cur_world2pv_transform)
            points_color_coord_holo_pv = points_coord_trans(points_color_coord_holo_pv, add_trans)
            if read_fitting:
                body_o3d.transform(cur_world2pv_transform)
                body_o3d.transform(add_trans)  # fitting in holo world coord

            # pcd_main_holo_pv_coord = o3d.geometry.PointCloud()
            # pcd_main_holo_pv_coord.points = o3d.utility.Vector3dVector(points_color_coord_holo_pv)
            # pcd_main_holo_pv_coord.colors = o3d.utility.Vector3dVector(colors_main)
            # o3d.visualization.draw_geometries([pcd_main_holo_pv_coord, body_o3d])

            ###### get main kienct point cloud projected on holoelns pv 2D plane
            colors_pv = copy.deepcopy(colors_main)
            colors_pv_default = copy.deepcopy(colors_main_default)

            holo_intrin_mtx = np.array([[cur_fx, 0.0, cx],
                                        [0.0, cur_fy, cy],
                                        [0.0, 0.0, 1.0]])
            v = points_color_coord_holo_pv.reshape((-1, 3)).copy()
            uvs_pv = cv2.projectPoints(v, np.asarray([[0.0, 0.0, 0.0]]), np.asarray([0.0, 0.0, 0.0]),
                                       holo_intrin_mtx,
                                       np.asarray(np.array([[0.0, 0, 0, 0, 0, 0, 0, 0]])))[0].squeeze()
            # uvs = projectPoints(points_color_coord, color_cam)  # [n_depth_points, 2]
            uvs_pv = np.round(uvs_pv).astype(int)
            valid_x = np.logical_and(uvs_pv[:, 1] >= 0, uvs_pv[:, 1] < 1080)  # [n_depth_points], true/false
            valid_y = np.logical_and(uvs_pv[:, 0] >= 0, uvs_pv[:, 0] < 1920)
            valid_idx_pv = np.logical_and(valid_x, valid_y)  # [n_depth_points], true/false
            valid_idx_pv = np.logical_and(valid_idx_pv, points_color_coord_holo_pv[:, 2] > 1e-2)
            uvs_pv = uvs_pv[valid_idx_pv == True]  # valid 2d coords in color img of 3d depth points
            # valid_idx_pv, uvs_pv = get_valid_idx(points_color_coord_holo_pv, color_cam_main)
            points_color_coord_holo_pv = points_color_coord_holo_pv[valid_idx_pv]
            colors_pv = colors_pv[valid_idx_pv]
            colors_pv_default = colors_pv_default[valid_idx_pv]

            # project to holo img
            img = cv2.imread(fpv_img_path)[:, :, ::-1] / 255.0
            img_0 = np.ones(img.shape)
            img_0[uvs_pv[:, 1], uvs_pv[:, 0], :] = colors_pv[:, :3]
            # img[uvs_pv[:, 1], uvs_pv[:, 0], :] = colors_pv_default[:, :3]
            img_out = (img_0 * 255.0).astype(np.uint8)
            img_out = img_out[:, :, ::-1]
            cv2.imshow('img_sub_2', img_out)
            cv2.waitKey(-1)

            img = cv2.imread(fpv_img_path)[:, :, ::-1] / 255.0
            img[uvs_pv[:, 1], uvs_pv[:, 0], :] = colors_pv[:, :3]
            # img[uvs_pv[:, 1], uvs_pv[:, 0], :] = colors_pv_default[:, :3]
            img_out = (img * 255.0).astype(np.uint8)
            img_out = img_out[:, :, ::-1]
            cv2.imshow('img_sub_2', img_out)
            cv2.waitKey(-1)

            # body_o3d.transform(cur_world2pv_transform)
            pcd_main_holo_pv_coord = o3d.geometry.PointCloud()
            pcd_main_holo_pv_coord.points = o3d.utility.Vector3dVector(points_color_coord_holo_pv)
            pcd_main_holo_pv_coord.colors = o3d.utility.Vector3dVector(colors_pv)
            if read_fitting:
                o3d.visualization.draw_geometries([pcd_main_holo_pv_coord, body_o3d])
            else:
                o3d.visualization.draw_geometries([pcd_main_holo_pv_coord])


            camera = pyrender.camera.IntrinsicsCamera(
                fx=cur_fx, fy=cur_fy,
                cx=camera_center[0], cy=camera_center[1])

            body.apply_transform(trans_kinect2holo)  # in hololens world coordinate
            body.apply_transform(cur_world2pv_transform)  # in hololens pv coordinate
            body_mesh = pyrender.Mesh.from_trimesh(body, material=material)


            if read_fitting and args.rendering_mode == 'body' or args.rendering_mode == 'both':
                img = cv2.imread(fpv_img_path)[:, :, ::-1] / 255.0
                # img = cv2.flip(img, 1)

                scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                                       ambient_light=(0.3, 0.3, 0.3))
                scene.add(camera, pose=camera_pose)
                scene.add(light, pose=camera_pose)
                scene.add(body_mesh, 'mesh')

                # static_scene = trimesh.load(osp.join(scene_dir, scene_name + '.obj'), enable_post_processing=True,
                #                             print_progress=True)
                # static_scene.apply_transform(trans_world_to_main)
                # static_scene.apply_transform(trans_kinect2holo)  # in hololens world coordinate
                # static_scene.apply_transform(cur_world2pv_transform)  # in hololens wold coordinate
                # static_scene_mesh = pyrender.Mesh.from_trimesh(static_scene)
                # scene.add(static_scene_mesh, 'mesh')



                r = pyrender.OffscreenRenderer(viewport_width=W,
                                               viewport_height=H,
                                               point_size=1.0)
                color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
                color = color.astype(np.float32) / 255.0

                valid_mask = (color[:, :, -1] > 0)[:, :, np.newaxis]
                input_img = img
                output_img = (color[:, :, :-1] * valid_mask +
                              (1 - valid_mask) * input_img)

                img = pil_img.fromarray((output_img * 255).astype(np.uint8))
                img.show()
                # img = img.resize((int(W / 2), int(H / 2)))
                # img.save(os.path.join(rendering_dir, img_name + '_output.jpg'))


            # if args.rendering_mode == '3d' or args.rendering_mode == 'both':
            #     static_scene = trimesh.load(osp.join(scene_dir, scene_name + '.obj'), enable_post_processing=True, print_progress=True)
            #     static_scene.apply_transform(trans_world_to_main)
            #     static_scene.apply_transform(trans_kinect2holo)  # in hololens world coordinate
            #     static_scene.apply_transform(cur_world2pv_transform)  # in hololens wold coordinate
            #     static_scene_mesh = pyrender.Mesh.from_trimesh(static_scene)
            #
            #     scene_o3d = o3d.io.read_triangle_mesh(osp.join(scene_dir, scene_name + '.obj'), enable_post_processing=True, print_progress=True)
            #     scene_o3d.compute_vertex_normals()
            #     scene_o3d.transform(trans_world_to_main)
            #     scene_o3d.transform(trans_kinect2holo)
            #
            #     body_o3d = o3d.geometry.TriangleMesh()
            #     body_o3d.vertices = o3d.utility.Vector3dVector(vertices)
            #     body_o3d.triangles = o3d.utility.Vector3iVector(body_model.faces)
            #     body_o3d.compute_vertex_normals()
            #     body_o3d.transform(trans_kinect2holo)
            #
            #     depth_pv = o3d.io.read_point_cloud('/mnt/hdd/egocaptures/record_20210911/recording_20210911_s1_02_moh_lam/2021-09-11-150137/Depth Long Throw/132758389535689301.ply')
            #     o3d.visualization.draw_geometries([depth_pv])
            #     # depth_pv.transform(cur_world2pv_transform)
            #     # depth_pv.transform(np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]]))
            #
            #     mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
            #     o3d.visualization.draw_geometries([depth_pv, mesh_frame])
            #     o3d.visualization.draw_geometries([body_o3d, scene_o3d, mesh_frame])
            #     o3d.visualization.draw_geometries([depth_pv, body_o3d, scene_o3d, mesh_frame])
            #
            #
            #     scene = pyrender.Scene()
            #     scene.add(camera, pose=camera_pose)
            #     scene.add(light, pose=camera_pose)
            #
            #     scene.add(static_scene_mesh, 'mesh')
            #     body_mesh = pyrender.Mesh.from_trimesh(
            #         body, material=material)
            #     scene.add(body_mesh, 'mesh')
            #
            #     r = pyrender.OffscreenRenderer(viewport_width=W,
            #                                    viewport_height=H)
            #     color, _ = r.render(scene)
            #     color = color.astype(np.float32) / 255.0
            #     img = pil_img.fromarray((color * 255).astype(np.uint8))
            #     # img.show(())
            #     # img = img.resize((int(W / 2), int(H / 2)))
            #     # img.save(os.path.join(body_scene_rendering_dir, img_name + '.jpg'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # /local/home/szhang/temp_prox/fit_results_15217_S3_infill_2_contact_0_1_fric_1_1_slide_other/PROXD
    # other:
    # MPH112_00150_01, MPH112_00157_01,
    # MPH11_00034_01, MPH11_00151_01, MPH11_03515_01,
    # N0SittingBooth_03301_01, N0SittingBooth_03403_01,
    # N0Sofa_00034_02, N0Sofa_00141_01, N0Sofa_00145_01,
    # N3OpenArea_00157_01, N3OpenArea_00158_01,
    # Werkraum_03301_01, Werkraum_03516_01, Werkraum_03516_02

    # todo:
    # N3Library_00157_01, N3Library_03301_01, N3Library_03301_02, N3Library_03375_01
    # N3Office_00034_01, N3Office_00150_01, N3Office_00153_01, N3Office_00159_01, N3Office_03301_01

    # not other:
    # BasementSittingBooth_00142_01, MPH112_00034_01, MPH11_00150_01, MPH16_00157_01, MPH1Library_00034_01, MPH8_00168_01
    # N0SittingBooth_00169_01, N0Sofa_00034_01, N3Library_00157_02, N3Office_00139_01, N3OpenArea_00157_02, Werkraum_03403_01

    # fit_results_15217_adam_other_slide / fit_results_15217_S3_infill_2_contact_0_1_fric_1_1_slide_other
    # parser.add_argument('--fitting_dir', type=str, default='/local/home/szhang/temp_prox/fit_results_15217_adam_other_slide/PROXD/N3OpenArea_00157_01')
    parser.add_argument('--body_color', type=str, default='pink', choices=['pink', 'white'])

    # parser.add_argument('--base_dir', type=str, default='/mnt/hdd/PROX', help='recording dir')
    parser.add_argument('--start', type=int, default=0, help='id of the starting frame')  # 13
    parser.add_argument('--step', type=int, default=100, help='id of the starting frame')
    parser.add_argument('--model_folder', default='/mnt/hdd/PROX/body_models/smplx_model', type=str, help='')
    parser.add_argument('--num_pca_comps', type=int, default=12,help='')
    # parser.add_argument('--save_meshes', default=False, type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument('--rendering_mode', default='body', type=str,
                choices=['body', '3d', 'both'],
                help='')

    args = parser.parse_args()
    main(args)
