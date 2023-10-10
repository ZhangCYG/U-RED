# test the performance
# only train the deformation and metching network
# the source labels are selected randomly
import os
import json
import torch
import pytorch3d.ops
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import datetime
import time
import torch.utils.data.dataloader
import sys
sys.path.append('./')
from train_utils.load_sources import load_sources

from tensorboardX import SummaryWriter
from dataset.dataset_utils import get_all_selected_models_pickle, get_random_labels, get_source_points
from dataset.dataset_utils import get_source_info, get_source_latent_codes_fixed, get_shape, get_source_info_mesh, get_source_info_visualization, get_shape_numpy, output_visualization_mesh
from dataset.shapenet_dataset import shapenet_dataset
from network.simple_encoder import TargetEncoder as simple_encoder
from network.deformation_net import DeformNet_MatchingNet as DM_decoder
from network.deformation_net import re_residual_net

from loss.chamfer_loss import compute_cm_loss


def main(cfg):
    # torch.autograd.set_detect_anomaly(True)

    writer = SummaryWriter(logdir=cfg["log_path"])
    if cfg["mode"] == 'train':
        DATA_SPLIT = 'train'
        bs = cfg["batch_size"]
    else:
        DATA_SPLIT = 'test'
        bs = cfg["batch_size"]  # must be 2

    dataset = shapenet_dataset(cfg)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=bs,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        shuffle=(True if cfg["mode"] == 'train' else False),
    )
    print("loading sources")
    SOURCE_MODEL_INFO, SOURCE_SEMANTICS, _, _, \
    MAX_NUM_PARAMS, MAX_NUM_PARTS = load_sources(cfg)

    print("sources loaded")
    # construct deformation and matching network
    target_encoder_full = simple_encoder()  #
    target_encoder_partial = simple_encoder()
    param_decoder_full = DM_decoder(cfg['source_latent_dim'] * 3, graph_dim=cfg['source_latent_dim'],
                                    max_num_parts=MAX_NUM_PARTS, matching=False)
    param_decoder_partial = DM_decoder(cfg['source_latent_dim'] * 3, graph_dim=cfg['source_latent_dim'],
                                       max_num_parts=MAX_NUM_PARTS, matching=False)
    recon_decoder_full = re_residual_net(cfg['target_latent_dim'] * 2)
    recon_decoder_partial = re_residual_net(cfg['target_latent_dim'] * 2)

    src_encoder_all = simple_encoder()
    recon_decoder_src = re_residual_net(cfg['source_latent_dim'] * 2)

    if cfg["init_dm"]:
        fname = os.path.join(cfg["dm_model_path"])
        state_dict = torch.load(fname)
        target_encoder_full.load_state_dict(state_dict["target_encoder_full"])
        target_encoder_partial.load_state_dict(state_dict["target_encoder_partial"])

        param_decoder_partial.load_state_dict(state_dict["param_decoder_partial"])
        param_decoder_full.load_state_dict(state_dict["param_decoder_full"])

        recon_decoder_full.load_state_dict(state_dict["recon_decoder_full"])
        recon_decoder_partial.load_state_dict(state_dict["recon_decoder_partial"])

        src_encoder_all.load_state_dict(state_dict["src_encoder_all"])
        recon_decoder_src.load_state_dict(state_dict["recon_decoder_src"])

        print("Initialize the dmnet, done!")

    target_encoder_full.to(cfg['device'], dtype=torch.float)
    target_encoder_partial.to(cfg["device"], dtype=torch.float)

    param_decoder_full.to(cfg['device'], dtype=torch.float)
    param_decoder_partial.to(cfg['device'], dtype=torch.float)

    recon_decoder_full.to(cfg['device'], dtype=torch.float)
    recon_decoder_partial.to(cfg['device'], dtype=torch.float)

    src_encoder_all.to(cfg['device'], dtype=torch.float)
    recon_decoder_src.to(cfg['device'], dtype=torch.float)

    # construct retrieval decoder
    re_order_decoder_full = re_residual_net(cfg['source_latent_dim'] + cfg['target_latent_dim'] * 2)
    re_order_decoder_partial = re_residual_net(cfg['source_latent_dim'] + cfg['target_latent_dim'] * 2)
    if cfg["init_re"]:
        fname_re = os.path.join(cfg["re_model_path"])
        state_dict = torch.load(fname_re)
        re_order_decoder_full.load_state_dict(state_dict["re_residual_net_full"])
        re_order_decoder_partial.load_state_dict(state_dict["re_residual_net_partial"])
    re_order_decoder_full = re_order_decoder_full.to(cfg['device'], dtype=torch.float)
    re_order_decoder_partial = re_order_decoder_partial.to(cfg['device'], dtype=torch.float)

    # turn all branches into test mode
    target_encoder_full.eval()
    target_encoder_partial.eval()
    param_decoder_full.eval()
    param_decoder_partial.eval()
    recon_decoder_full.eval()
    recon_decoder_partial.eval()
    re_order_decoder_full.eval()
    re_order_decoder_partial.eval()
    src_encoder_all.eval()
    recon_decoder_src.eval()

    np.random.rand(int(time.time()))

    # get src latent codes
    print("calculating source labels for retrieval")
    source_labels_all = np.expand_dims(np.arange(len(SOURCE_MODEL_INFO)), axis=1)
    source_labels_all = np.reshape(source_labels_all, (-1))
    source_labels_all = source_labels_all[:500]
    print(source_labels_all.shape)

    '''
     re_src_latent_codes_template = get_source_latent_codes_fixed(source_labels, RETRIEVAL_SOURCE_LATENT_CODES,
                                                                 device=cfg['device'])
     '''

    src_mats, src_default_params, \
    src_connectivity_mat = get_source_info(source_labels_all, SOURCE_MODEL_INFO,
                                           MAX_NUM_PARAMS,
                                           use_connectivity=cfg["use_connectivity"])

    src_points_cloud, point_labels, num_parts = get_source_points(source_labels_all, SOURCE_MODEL_INFO,
                                                                  device=cfg["device"])
    with torch.no_grad():
        src_latent_codes_all, src_per_point_f = src_encoder_all(src_points_cloud)

    src_per_point_f = src_per_point_f.permute(0, 2, 1)
    part_latent_codes_all = []
    for w in range(src_latent_codes_all.shape[0]):
        src_f_now = []
        for v in range(num_parts[w]):
            src_f_now.append(torch.mean(src_per_point_f[w, point_labels[w, :] == v, :], dim=0))
        src_f_now = torch.stack(src_f_now)
        part_latent_codes_all.append(src_f_now)

    best_cd_loss_full = []
    best_cd_loss_partial = []
    best_re_loss_full = []
    best_re_loss_partial = []
    best_re_cd_loss_full = []
    best_re_cd_loss_partial = []

    for i, batch in enumerate(loader):
        print(str(i), '/', str(len(loader)))
        target_shapes, target_ids, target_labels, semantics, point_occ, point_occ_mask, ori_point_occ = batch
        # forward pass the deformation and matching network to get the loss
        if cfg["complementme"]:
            target_shapes[:, :, 2] = -target_shapes[:, :, 2]
            point_occ[:, :, 2] = -point_occ[:, :, 2]

        x = [x.to(cfg["device"], dtype=torch.float) for x in target_shapes]
        x_p = [x_p.to(cfg["device"], dtype=torch.float) for x_p in point_occ]
        x = torch.stack(x)
        x_p = torch.stack(x_p)
        # pdb.set_trace()

        # transfer data to gpu

        mat = [mat.to(cfg["device"], dtype=torch.float) for mat in src_mats]
        def_param = [def_param.to(cfg["device"], dtype=torch.float) for def_param in src_default_params]

        mat_all = torch.stack(mat)
        def_param_all = torch.stack(def_param)
        print(def_param_all.shape)

        if cfg["use_connectivity"]:
            conn_mat = [conn_mat.to(cfg["device"], dtype=torch.float) for conn_mat in src_connectivity_mat]
            conn_mat_all = torch.stack(conn_mat)

        # forward pass
        # here maybe some problems
        with torch.no_grad():
            target_latent_codes_full, per_point_full = target_encoder_full(x)
            target_latent_codes_partial, per_point_partial = target_encoder_partial(x_p)

        cd_loss_full_list = []
        cd_loss_partial_list = []
        re_loss_full_list = []
        re_loss_partial_list = []
        para_all = []
        # pdb.set_trace()
        # print(src_latent_codes_all.shape[0])
        sample_full_target_code = torch.rand(cfg["batch_size"], cfg['target_latent_dim']).to(cfg["device"], dtype=torch.float)
        sample_full_target_code_norm = F.normalize(sample_full_target_code, dim=1)
        for j in range(src_latent_codes_all.shape[0]):
            with torch.no_grad():
                src_latent_codes = src_latent_codes_all[j, :].view(1, -1).repeat(cfg["batch_size"], 1)
                part_latent_codes = [part_latent_codes_all[j], part_latent_codes_all[j]]
                # print(mat_all.shape)
                # print(def_param_all.shape)
                # print(conn_mat_all.shape)
                mat = mat_all[j].unsqueeze(0)
                # print(mat.shape)
                def_param = def_param_all[j].unsqueeze(0)
                # print(def_param.shape)
                conn_mat = conn_mat_all[j].unsqueeze(0)
                # print(conn_mat.shape)

                # construct the input of the retrieval residual network
                # here only consider the partial part
                nofp = per_point_partial.shape[-1]
                re_target_latent_codes = target_latent_codes_partial.unsqueeze(1).repeat(1, nofp, 1)
                re_source_latent_codes = src_latent_codes.unsqueeze(1).repeat(1, nofp, 1)
                re_input_codes = torch.cat([per_point_partial.permute(0, 2, 1),
                                            re_target_latent_codes,
                                            re_source_latent_codes], dim=-1)

                re_residuals_partial = re_order_decoder_partial(torch.cat([re_input_codes, sample_full_target_code_norm.unsqueeze(1).repeat(1, nofp, 1)], dim=-1))  # bs x 1024 x 3

                # for target recon
                # recon_partial_p = recon_decoder_partial(recon_input_codes_partial)

                nofp = per_point_full.shape[-1]
                recon_target_codes_full = target_latent_codes_full.unsqueeze(1).repeat(1, nofp, 1)
                recon_input_codes_full = torch.cat([per_point_full.permute(0, 2, 1),
                                                    recon_target_codes_full], dim=-1)
                # recon_full_p = recon_decoder_full(recon_input_codes_full)

                # re_residuals  full
                re_source_latent_codes_full = src_latent_codes.unsqueeze(1).repeat(1, nofp, 1)
                re_input_codes_full = torch.cat([recon_input_codes_full, re_source_latent_codes_full], dim=-1)

                re_residuals_full = re_order_decoder_full(re_input_codes_full)

                loss_re_full, _ = torch.max(torch.sum(torch.abs(re_residuals_full), dim=-1), dim=-1)
                loss_re_partial, _ = torch.max(torch.sum(torch.abs(re_residuals_partial), dim=-1), dim=-1)

                '''
                loss_re_full, _ = torch.topk(torch.sum(torch.abs(re_residuals_full), dim=-1),
                                             k=cfg['top_k'], dim=-1, largest=False)
                loss_re_partial, _ = torch.topk(torch.sum(torch.abs(re_residuals_partial), dim=-1),
                                                k=cfg['top_k'], dim=-1, largest=False)
                '''

                re_loss_full_list.append(loss_re_full)
                re_loss_partial_list.append(loss_re_partial)

                params_full, _ = param_decoder_full(src_latent_codes,
                                                    target_latent_codes_full,
                                                    part_latent_codes,
                                                    per_point_full)
                params_partial, _ = param_decoder_partial(src_latent_codes,
                                                          target_latent_codes_partial,
                                                          part_latent_codes,
                                                          per_point_partial)

            params_partial = torch.stack(params_partial)
            params_full = torch.stack(params_full)
            para_all.append(params_partial)

            # using params to get deformed shape
            # def param, original bounding box info
            # pdb.set_trace()
            if cfg["use_connectivity"]:
                output_pc_from_full = get_shape(mat, params_full, def_param,
                                                cfg["alpha"], connectivity_mat=conn_mat)
                output_pc_from_partial = get_shape(mat, params_partial, def_param,
                                                   cfg["alpha"], connectivity_mat=conn_mat)
            else:
                output_pc_from_full = get_shape(mat, params_full, def_param, cfg["alpha"])
                output_pc_from_partial = get_shape(mat, params_partial, def_param, cfg["alpha"])

            cd_loss_partial = compute_cm_loss(output_pc_from_partial.detach(), x, batch_reduction=None)
            cd_loss_full = compute_cm_loss(output_pc_from_full.detach(), x, batch_reduction=None)

            cd_loss_full_list.append(cd_loss_full)
            cd_loss_partial_list.append(cd_loss_partial)

        # process data for each batch
        cd_loss_partial_list = torch.stack(cd_loss_partial_list)  # 500 x 2
        cd_loss_full_list = torch.stack(cd_loss_full_list)  # 500 x 2

        re_loss_full_list = torch.stack(re_loss_full_list)
        re_loss_partial_list = torch.stack(re_loss_partial_list)
        para_all = torch.stack(para_all)
        # pdb.set_trace()
        for q in range(cfg['batch_size']):
            cd_loss_full_list_now = cd_loss_full_list[:, q]
            cd_loss_partial_list_now = cd_loss_partial_list[:, q]
            best_cd_loss_full.append(torch.min(cd_loss_full_list_now))
            best_cd_loss_partial.append(torch.min(cd_loss_partial_list_now))

            re_loss_full_list_now = re_loss_full_list[:, q]
            re_loss_partial_list_now = re_loss_partial_list[:, q]

            min_re, idx_re = torch.topk(re_loss_full_list_now, k=cfg['top_k'], largest=False, dim=0)
            best_re_loss_full.append(torch.min(min_re))
            best_re_cd_loss_full.append(torch.min(cd_loss_full_list_now[idx_re]))

            min_re, idx_re = torch.topk(re_loss_partial_list_now, k=cfg['top_k'], largest=False, dim=0)
            best_re_loss_partial.append(torch.min(min_re))
            best_re_cd_loss_partial.append(torch.min(cd_loss_partial_list_now[idx_re]))

            # mesh rendering
            para_re = para_all[:, q, :, :].unsqueeze(1)
            para_re = para_re.squeeze(3)
            vis_re_idx = idx_re[0]
            retrieved_idx_repeated = vis_re_idx.unsqueeze(0).unsqueeze(-1).repeat(1, 1, para_re.shape[-1])
            params_retrieved = torch.gather(para_re, 0, retrieved_idx_repeated)
            params_retrieved = params_retrieved.to("cpu")
            params_retrieved = params_retrieved.detach().numpy()[0]

            target_labels = target_labels.to("cpu")
            target_labels = target_labels.detach().numpy()
            target_ids = target_ids.to("cpu")
            target_ids = target_ids.detach().numpy()
            vis_re_idx = vis_re_idx.to("cpu").unsqueeze(0)
            vis_re_idx = vis_re_idx.detach().numpy()

            # pdb.set_trace()
            src_vertices_mats, src_default_params_mesh, src_conn_mat = get_source_info_mesh(vis_re_idx, SOURCE_MODEL_INFO, MAX_NUM_PARAMS, use_connectivity=cfg["use_connectivity"])
            src_points, src_labels, src_ids, _, src_vertices, src_faces, src_face_labels = get_source_info_visualization(vis_re_idx, SOURCE_MODEL_INFO, mesh=True)
            # pdb.set_trace()
            curr_param = np.expand_dims(params_retrieved, -1)
            curr_mat = src_vertices_mats[0].detach().numpy()
            
            if (cfg["use_connectivity"]):
                curr_conn_mat = src_conn_mat[0].detach().numpy()
            else:
                curr_conn_mat = None

            curr_default_param = src_default_params_mesh[0].detach().numpy().T
            output_vertices = get_shape_numpy(curr_mat, curr_param, curr_default_param, 0.1, connectivity_mat=curr_conn_mat)

            if cfg["complementme"]:
                output_vertices[:, 2] = -output_vertices[:, 2]
                curr_src_vertices = src_vertices[0].copy()
                curr_src_vertices[:, 2] = -src_vertices[0][:, 2]
                target_shapes[0][:, 2] = -target_shapes[0][:, 2]
            else:
                curr_src_vertices = src_vertices[0]
            # pdb.set_trace()
            print(target_ids[0])
            output_visualization_mesh(output_vertices, curr_src_vertices, src_faces[0], ori_point_occ[0].to("cpu").detach().numpy(), src_face_labels[0], target_labels[0], src_ids[0], target_ids[0], cfg['log_path'])
    # pdb.set_trace()
    best_re_cd_loss_full = torch.stack(best_re_cd_loss_full)
    best_re_cd_loss_partial = torch.stack(best_re_cd_loss_partial)
    best_cd_loss_full = torch.stack(best_cd_loss_full)
    best_cd_loss_partial = torch.stack(best_cd_loss_partial)
    best_re_loss_full = torch.stack(best_re_loss_full)
    best_re_loss_partial = torch.stack(best_re_loss_partial)

    print("best full cd loss from retrieval=" + str(torch.mean(best_re_cd_loss_full).cpu().numpy()),
          "best partial cd loss from retrieval=" + str(torch.mean(best_re_cd_loss_partial).cpu().numpy()),
          "best full cd loss=" + str(torch.mean(best_cd_loss_full).cpu().numpy()),
          "best partial cd loss=" + str(torch.mean(best_cd_loss_partial).cpu().numpy()),
          "best full re loss=" + str(torch.mean(best_re_loss_full).cpu().numpy()),
          "best partial re loss=" + str(torch.mean(best_re_loss_partial).cpu().numpy()),
          )


if __name__ == '__main__':
    import json

    config_path = sys.argv[1]
    config = json.load(open(config_path, 'rb'))
    main(config)
