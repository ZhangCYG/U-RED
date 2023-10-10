import numpy as np
import os
import pickle

import pytorch3d.ops
import torch
import pytorch3d


def generate_occ_point_ball(points, ids, save_pth=''):
    # input 2048 x 3
    # output: occ_points  1024 x 3,  point_occ_mask  1024
    # indicates the idx of selected points
    nofp = points.shape[0]
    assert nofp == 2048

    # circle based occlusion point generation
    dis_mat_path = os.path.join(save_pth, str(int(ids)) + '.pickle')
    if os.path.exists(dis_mat_path):
        with open(dis_mat_path, 'rb') as dis_mat_file:
            try:
                dis_mat = pickle.load(dis_mat_file)
            finally:
                return generate_occ_point_random(points)
    else:
        dis_mat_dir = os.path.dirname(dis_mat_path)
        os.makedirs(dis_mat_dir, exist_ok=True)
        with open(dis_mat_path, 'wb') as dis_mat_file:
            points_t = torch.from_numpy(points)
            points_t = points_t.unsqueeze(0)
            knn_cal = pytorch3d.ops.knn_points(points_t, points_t, K=nofp // 2, return_sorted=True)
            knn_cal = knn_cal.idx
            dis_mat = knn_cal.squeeze().numpy()
            pickle.dump(dis_mat, dis_mat_file)

    # until here we get the dis_mat
    center_p_choice = np.array([1, 2, 4, 8])
    nofcenter = np.random.choice(center_p_choice)  # select a number from center_p_choice
    selected_points = np.random.choice(nofp, size=nofcenter, replace=False)
    cancel_num = nofp // 2 // nofcenter  # for each selected points, we need to cancel cancel_num_points
    center_p_dis = dis_mat[selected_points, :]  # nofcenter x 1024
    cancel_candidate = center_p_dis[:, :cancel_num]
    cancel_candidate = cancel_candidate.reshape(-1)
    point_occ_mask = np.arange(nofp)
    point_occ_mask[cancel_candidate] = -1
    selected_occ_mask = point_occ_mask[point_occ_mask >= 0]
    nofoccp = len(selected_occ_mask)
    if nofoccp > nofp // 2:
        new_cal_selection = np.random.choice(nofoccp, size=nofoccp - nofp // 2, replace=False)
        selected_occ_mask[new_cal_selection] = -1
        selected_occ_mask = selected_occ_mask[selected_occ_mask >= 0]
    occ_points = points[selected_occ_mask, :]
    return occ_points, selected_occ_mask


def generate_occ_point_slice(points):
    nofp = points.shape[0]
    assert nofp == 2048

    # select a point as center
    center_ids = np.random.randint(nofp, size=1)
    center_pts = points[center_ids, :]  # 3
    # generate a plane normal
    selected_direction = np.random.uniform(low=1e-3, high=1.0, size=3)  # generate a point
    selected_direction = selected_direction / np.linalg.norm(selected_direction)
    # calculate the distance from each point to the defined plane
    points_res = points - center_pts.reshape(1, 3)
    dis_mat = np.abs(np.dot(points_res, selected_direction.reshape(3, 1)))  # nx1
    dis_idx = np.argsort(dis_mat.reshape(-1))
    selected_occ_mask = dis_idx[nofp // 2 - 1:-1]
    occ_points = points[selected_occ_mask, :]
    return occ_points, selected_occ_mask


def generate_occ_point_random(points):
    nofp = points.shape[0]
    assert nofp == 2048
    selected_points = np.random.choice(nofp, size=nofp // 2, replace=False)
    point_occ_mask = selected_points
    point_occ = points[selected_points, :]
    return point_occ, point_occ_mask


def generate_occ_point_part(points, semantics):
    nofp = points.shape[0]
    assert nofp == 2048
    selected_semantics = np.random.choice(semantics.shape[0], size=1, replace=False)
    point_occ_mask = (semantics != semantics[selected_semantics])
    point_occ = points[point_occ_mask, :]
    if point_occ.shape[0] > nofp // 2:
        true_indices = np.where(point_occ_mask == 1)[0]
        selected_indices = np.random.choice(true_indices, size=nofp // 2, replace=False)
        point_occ_mask = selected_indices
        point_occ = points[point_occ_mask, :]
    else:
        selected_points = np.random.choice(nofp, size=nofp // 2, replace=False)
        point_occ_mask = selected_points
        point_occ = points[selected_points, :]

    return point_occ, point_occ_mask