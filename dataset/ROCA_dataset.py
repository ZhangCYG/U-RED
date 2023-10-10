from torch.utils.data import Dataset
import numpy as np
import os

from dataset.dataset_utils import *
import json
from scipy.spatial.transform import Rotation as R
import cv2
import pickle


class ROCA_dataset(Dataset):
    def __init__(self, config):
        
        category = config["category"]
        if category == 'chair':
            self.category_id = 5
        elif category == 'storagefurniture':
            self.category_id = 3
        elif category == 'table':
            self.category_id = 8
        self.data_root = '/data3/zcyg/ROCA-data/Data'
        self.width = 480
        self.height = 360
        instances_file = open(os.path.join(self.data_root, 'Dataset', 'scan2cad_instances_val.json'))
        inst_json = json.load(instances_file)
        instances_file.close()
        scene_all = os.listdir(os.path.join(self.data_root, 'Images/tasks/scannet_frames_25k'))
        self.pred_depth_dir = '/data3/zrd/results/scan2cad_swinl_27_22k'

        pred_depth_pkl = open('/data3/zrd/results/depth_preds.pkl', 'rb')
        self.pred_depth_pkl = pickle.load(pred_depth_pkl)
        pred_depth_pkl.close()

        # select by category id
        self.annos = [inst_json['annotations'][i] for i in range(len(inst_json['annotations'])) if inst_json['annotations'][i]['category_id'] == self.category_id]
        self.images = inst_json['images']

        self.target_points, self.ids, self.align_ids, self.t_means, self.imgs = self.get_target_points()
        self.n_samples = self.target_points.shape[0]

        print("Number of targets: " + str(self.n_samples))

    def quaternion_rotation_matrix(self, Q):
        """
        Covert a quaternion into a full three-dimensional rotation matrix.
    
        Input
        :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3) 
    
        Output
        :return: A 3x3 element matrix representing the full 3D rotation matrix. 
                This rotation matrix converts a point in the local reference 
                frame to a point in the global reference frame.
        """
        # Extract the values from Q
        q0 = Q[0]
        q1 = Q[1]
        q2 = Q[2]
        q3 = Q[3]
        
        # First row of the rotation matrix
        r00 = 2 * (q0 * q0 + q1 * q1) - 1
        r01 = 2 * (q1 * q2 - q0 * q3)
        r02 = 2 * (q1 * q3 + q0 * q2)
        
        # Second row of the rotation matrix
        r10 = 2 * (q1 * q2 + q0 * q3)
        r11 = 2 * (q0 * q0 + q2 * q2) - 1
        r12 = 2 * (q2 * q3 - q0 * q1)
        
        # Third row of the rotation matrix
        r20 = 2 * (q1 * q3 - q0 * q2)
        r21 = 2 * (q2 * q3 + q0 * q1)
        r22 = 2 * (q0 * q0 + q3 * q3) - 1
        
        # 3x3 rotation matrix
        rot_matrix = np.array([[r00, r01, r02],
                            [r10, r11, r12],
                            [r20, r21, r22]])
                                
        return rot_matrix

    def get_target_points(self):
        x, y = np.meshgrid(list(range(self.width)), list(range(self.height)))
        target_points = []
        ids = []
        align_ids = []
        t_means = []
        imgs = []
        for anno in self.annos:
            intrinsics = anno['intrinsics']
            q = anno['q']
            s = anno['s']
            t = anno['t']
            R_mat = self.quaternion_rotation_matrix(q)
            image_list = [self.images[i] for i in range(len(self.images)) if self.images[i]['id'] == anno['image_id']]
            assert len(image_list) == 1
            image = image_list[0]
            image_file = image['file_name'][25:]
            img = cv2.imread(os.path.join(self.data_root, 'Images/tasks/scannet_frames_25k', image_file))
            # print(img.shape)

            # depth_file = image_file.replace('color', 'depth')
            # depth_file = depth_file.replace('jpg', 'png')
            # depth = cv2.imread(os.path.join(self.data_root, 'Rendering', depth_file), cv2.IMREAD_UNCHANGED)
            # depth = depth / 1000.

            # pred_depth_filename = image_file[:-4] + '.npy'
            # # print(pred_depth_filename)
            # pred_depth_file = os.path.join(self.pred_depth_dir, pred_depth_filename)
            # pred_depth = np.load(pred_depth_file)
            # pred_depth = pred_depth.squeeze()
            # pred_depth = cv2.resize(pred_depth, [480, 360])
            # depth = pred_depth

            pred_depth = self.pred_depth_pkl['/data/zrd/datasets/ROCA-data/Data/Images/tasks/scannet_frames_25k/' + image_file]
            pred_depth = pred_depth.numpy().squeeze()
            pred_depth = cv2.resize(pred_depth, [480, 360])
            depth = pred_depth

            fx = intrinsics[0][0]
            fy = intrinsics[1][1]
            ux = intrinsics[0][2]
            uy = intrinsics[1][2]
            pc_x = depth * (x - ux) / fx
            pc_y = depth * (y - uy) / fy
            pc = np.concatenate((np.expand_dims(pc_x, 2), np.expand_dims(pc_y, 2), np.expand_dims(depth, 2)), axis=2)  # pc in camera coordinate
            # mask
            mask_file = image_file.replace('color', 'instance')
            mask_file = mask_file.replace('jpg', 'png')
            mask = cv2.imread(os.path.join(self.data_root, 'Rendering', mask_file), cv2.IMREAD_UNCHANGED)
            mask = mask == anno['alignment_id']
            pc_masked = pc[mask]
            # img = img * np.expand_dims(mask, 2)
            # random down sampling to 2048
            if pc_masked.shape[0] < 2048:
                continue
            rdm_idx = np.random.choice(pc_masked.shape[0], 2048, replace=False)
            # transform pc into object coord
            pc_final = pc_masked[rdm_idx]
            t_mean = np.mean(pc_final, axis=0)
            # pc_final = pc_final - t_mean
            pc_final = pc_final - t
            pc_final = (np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]).dot(R_mat.T).dot(pc_final.T)).T
            pc_final = pc_final / s
            target_points.append(pc_final)
            ids.append(anno['image_id'])
            align_ids.append(anno['alignment_id'])
            t_means.append(t_mean)
            imgs.append(img)
            # print(img.shape)
        return np.stack(target_points, 0), np.stack(ids), np.stack(align_ids), np.stack(t_means), np.stack(imgs)

    def __getitem__(self, index):
        points = self.target_points[index]  # size 2048 x 3
        # note that ids and labels  are only used in visualization and retrieval
        # focalization
        points_occ_mean = np.mean(points, axis=0, keepdims=True)
        points_occ = points - points_occ_mean
        
        return points_occ, self.ids[index], self.align_ids[index], points_occ, self.t_means[index], self.imgs[index]

    def __len__(self):
        return self.n_samples
