import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

from dataset.dataset_utils import *
from PIL import Image
from dataset.gen_occ_point import generate_occ_point_ball, generate_occ_point_slice, generate_occ_point_random, generate_occ_point_part
from train_utils.random_rot import rotation_matrix_3d


class shapenet_dataset(Dataset):
    def __init__(self, config):
        if config["complementme"]:

            filename = os.path.join(config["base_dir"], "generated_datasplits_complementme", config["middle_name"],
                                    "generated_datasplits_complementme",
                                    config["category"] + '_' + str(config["num_source"])
                                    + '_' + config["mode"] + ".h5")
        else:
            filename = os.path.join(config["base_dir"], "generated_datasplits", config["middle_name"],
                                    "generated_datasplits", config["category"] + '_' + str(config["num_source"])
                                    + '_' + config["mode"] + ".h5")

        self.dis_mat_path = os.path.join(config["base_dir"], "dis_mat", config["category"])

        all_target_points, all_target_labels, all_target_semantics, all_target_model_id = load_h5(filename)
        self.target_points = all_target_points
        self.target_labels = all_target_labels
        self.target_semantics = all_target_semantics
        self.target_ids = all_target_model_id

        self.n_samples = all_target_points.shape[0]
        self.random_rot = config["random_rot"]

        print("Number of targets: " + str(self.n_samples))

    def __getitem__(self, index):
        # occlusion handling
        points = self.target_points[index]  # size 2048 x 3
        # note that ids and labels  are only used in visualization and retrieval
        ids = self.target_ids[index]  # 1
        labels = self.target_labels[index]  # 2048   view label, from which view
        semantics = self.target_semantics[index]  # 2048  part segementation
        ##  randomly generate occ points
        choose_one_occ = np.random.rand()
        if choose_one_occ < 0.3:
            points_occ, points_occ_mask = generate_occ_point_ball(points, ids, save_pth=self.dis_mat_path)
        elif choose_one_occ < 0.6:
            points_occ, points_occ_mask = generate_occ_point_random(points)
        elif choose_one_occ < 0.9:
            points_occ, points_occ_mask = generate_occ_point_slice(points)
        else:
            points_occ, points_occ_mask = generate_occ_point_part(points, semantics)
        # focalization
        ori_point_occ = points_occ
        points_occ_mean = np.mean(points_occ, axis=0, keepdims=True)
        points_occ = points_occ - points_occ_mean
        #  numpy check if there is none or inf
        '''
        if ((True in np.isnan(points_occ)) or (True in np.isnan(points_occ_mask))
                or (True in np.isinf(points_occ)) or (True in np.isinf(points_occ_mask))):
            print(str(ids), str(index), str(choose_one_occ))
            return self.__getitem__((index + 1) % self.__len__())
        if (True in np.isnan(points)):
            print(str(1024), str(ids), str(index), str(choose_one_occ))
            return self.__getitem__((index + 1) % self.__len__())
        '''
        if self.random_rot:
            angle = np.random.uniform(low=-10.0, high=10.0, size=6)
            #R_full = rotation_matrix_3d(angle[0], angle[1], angle[2])[:3, :3]
            R_partial = rotation_matrix_3d(angle[3], angle[4], angle[5])[:3, :3]
            #points = (np.matmul(R_full, points.T)).T
            points_occ = (np.matmul(R_partial, points_occ.T)).T
        # print(np.mean(points, axis=0))
        return points, ids, labels[points_occ_mask], semantics[points_occ_mask], points_occ, points_occ_mask, ori_point_occ

    def __len__(self):
        return self.n_samples


if __name__ == "__main__":
    import json
    import time

    np.random.seed(int(time.time()))
    config_path = '../config/unused/config_dm.json'
    config = json.load(open(config_path, 'rb'))
    dataset = shapenet_dataset(config)
    points, ids, labels, semantics, points_occ, points_occ_mask = dataset.__getitem__(1200)

    import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D

    ref_points = points
    fig = plt.figure()
    ax = fig.add_subplot(222, projection='3d')
    ax.scatter(ref_points[:, 0], ref_points[:, 1], ref_points[:, 2], marker='.')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points_occ[:, 0], points_occ[:, 1], points_occ[:, 2], marker='.')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()
    plt.show()
