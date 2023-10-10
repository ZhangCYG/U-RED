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
import sys
sys.path.append('./')

import torch.utils.data.dataloader
from train_utils.load_sources import load_sources
from train_utils.optimizer_dm import define_optimizer_dm_re_recon

from tensorboardX import SummaryWriter
from dataset.dataset_utils import get_all_selected_models_pickle, get_random_labels, get_source_points
from dataset.dataset_utils import get_source_info, get_source_latent_codes_fixed, get_shape
from dataset.dataset_utils import get_symmetric
from dataset.shapenet_dataset import shapenet_dataset
from network.simple_encoder import TargetEncoder as simple_encoder
from network.deformation_net import DeformNet_MatchingNet as DM_decoder
from network.deformation_net import re_residual_net

from loss.chamfer_loss import compute_cm_loss
from loss.basic_loss import point_loss_matching, residual_retrieval_loss
from loss.basic_consistency_loss import compute_pc_consistency, compute_param_consistency


def main(cfg):
    # torch.autograd.set_detect_anomaly(True)

    writer = SummaryWriter(logdir=cfg["log_path"])
    if cfg["mode"] == 'train':
        DATA_SPLIT = 'train'
        bs = cfg["batch_size"]
    else:
        DATA_SPLIT = 'test'
        bs = 2

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
    re_order_decoder_partial = re_residual_net(cfg['source_latent_dim'] + cfg['target_latent_dim'] * 3)
    if cfg["init_re"]:
        fname_re = os.path.join(cfg["re_model_path"])
        state_dict = torch.load(fname_re)
        re_order_decoder_full.load_state_dict(state_dict["re_residual_net_full"])
        re_order_decoder_partial.load_state_dict(state_dict["re_residual_net_partial"])
    re_order_decoder_full = re_order_decoder_full.to(cfg['device'], dtype=torch.float)
    re_order_decoder_partial = re_order_decoder_partial.to(cfg['device'], dtype=torch.float)

    # define optimizer and scheduler
    optimizer, scheduler = define_optimizer_dm_re_recon(target_encoder_full, target_encoder_partial,
                                                        param_decoder_full, param_decoder_partial,
                                                        recon_decoder_full, recon_decoder_partial,
                                                        re_order_decoder_full, re_order_decoder_partial,
                                                        src_encoder_all, recon_decoder_src,
                                                        cfg)

    # start training
    if cfg["mode"] == 'train':
        target_encoder_full.train()
        target_encoder_partial.train()
        param_decoder_full.train()
        param_decoder_partial.train()
        recon_decoder_full.train()
        recon_decoder_partial.train()
        re_order_decoder_full.train()
        re_order_decoder_partial.train()
        src_encoder_all.train()
        recon_decoder_src.train()

        for epoch in range(cfg["epochs"]):
            np.random.seed(int(time.time()))
            start = datetime.datetime.now()
            print(str(start), 'training epoch', str(epoch))
            for i, batch in enumerate(loader):
                target_shapes, _, target_labels, semantics, point_occ, point_occ_mask, ori_point_occ = batch
                if cfg["complementme"]:
                    target_shapes[:, :, 2] = -target_shapes[:, :, 2]
                    point_occ[:, :, 2] = -point_occ[:, :, 2]
                # pre_train, thus select random source for each target to train
                source_label_shape = torch.zeros(target_shapes.shape[0])  # batchsize
                source_labels = get_random_labels(source_label_shape, len(SOURCE_MODEL_INFO))
                #writer.add_scalar('source_labels', source_labels[0], global_step=epoch * len(loader) + i)
                # get specific information of selected source shapes
                src_mats, src_default_params, \
                src_connectivity_mat = get_source_info(source_labels, SOURCE_MODEL_INFO,
                                                       MAX_NUM_PARAMS,
                                                       use_connectivity=cfg["use_connectivity"])
                # get latent codes of the selected source shapes
                src_points_cloud, point_labels, num_parts = get_source_points(source_labels, SOURCE_MODEL_INFO,
                                                                              device=cfg["device"])
                src_latent_codes, src_per_point_f = src_encoder_all(src_points_cloud)
                # for src reconstrcution
                recon_src_latent_codes = src_latent_codes.unsqueeze(2).repeat(1, 1, src_per_point_f.shape[-1])
                recon_src_input_f = torch.cat([recon_src_latent_codes, src_per_point_f], dim=1)
                recon_src_p = recon_decoder_src(recon_src_input_f.permute(0, 2, 1))

                ### use src_per_point_f to generate per part features
                src_per_point_f = src_per_point_f.permute(0, 2, 1)
                part_latent_codes = []

                for w in range(src_points_cloud.shape[0]):  # dont use batchsize, in loading, there may be samller batch
                    src_f_now = []
                    for v in range(num_parts[w]):
                        src_f_now.append(torch.mean(src_per_point_f[w, point_labels[w, :] == v, :], dim=0))
                    src_f_now = torch.stack(src_f_now)
                    part_latent_codes.append(src_f_now)

                # transfer data to cpu

                x = [x.to(cfg["device"], dtype=torch.float) for x in target_shapes]
                x_p = [x_p.to(cfg["device"], dtype=torch.float) for x_p in point_occ]
                occ_mask = [occ_mask.to(cfg["device"], dtype=torch.int64) for occ_mask in point_occ_mask]
                mat = [mat.to(cfg["device"], dtype=torch.float) for mat in src_mats]
                def_param = [def_param.to(cfg["device"], dtype=torch.float) for def_param in src_default_params]

                x = torch.stack(x)
                x_p = torch.stack(x_p)
                occ_mask = torch.stack(occ_mask)
                mat = torch.stack(mat)
                def_param = torch.stack(def_param)
                '''

                if True in torch.isnan(x) or True in torch.isnan(x_p):
                    print("wrong in dataloader1")
                    #assert False

                if True in torch.isnan(mat) or True in torch.isnan(occ_mask):
                    print("wrong in dataloader2")
                    #assert False
                '''

                if cfg["use_connectivity"]:
                    conn_mat = [conn_mat.to(cfg["device"], dtype=torch.float) for conn_mat in src_connectivity_mat]
                    conn_mat = torch.stack(conn_mat)

                # forward pass
                # here maybe some problems
                target_latent_codes_full, per_point_full = target_encoder_full(x)
                target_latent_codes_partial, per_point_partial = target_encoder_partial(x_p)

                # for target recon
                nofp = per_point_partial.shape[-1]
                re_target_latent_codes = target_latent_codes_partial.unsqueeze(1).repeat(1, nofp, 1)
                recon_input_codes_partial = torch.cat([per_point_partial.permute(0, 2, 1),
                                                       re_target_latent_codes], dim=-1)
                recon_partial_p = recon_decoder_partial(recon_input_codes_partial)

                nofp = per_point_full.shape[-1]
                recon_target_codes_full = target_latent_codes_full.unsqueeze(1).repeat(1, nofp, 1)
                recon_input_codes_full = torch.cat([per_point_full.permute(0, 2, 1),
                                                    recon_target_codes_full], dim=-1)
                recon_full_p = recon_decoder_full(recon_input_codes_full)

                # re_residuals  full
                re_source_latent_codes_full = src_latent_codes.unsqueeze(1).repeat(1, nofp, 1)
                re_input_codes_full = torch.cat([recon_input_codes_full, re_source_latent_codes_full], dim=-1)
                re_residuals_full = re_order_decoder_full(re_input_codes_full)

                # construct the input of the retrieval residual network
                # here only consider the partial part
                nofp = per_point_partial.shape[-1]
                re_source_latent_codes = src_latent_codes.unsqueeze(1).repeat(1, nofp, 1)
                re_target_latent_codes_full = (target_latent_codes_full / torch.norm(target_latent_codes_full)).unsqueeze(1).repeat(1, nofp, 1)
                re_input_codes = torch.cat([per_point_partial.permute(0, 2, 1),
                                            re_target_latent_codes,
                                            re_source_latent_codes,
                                            re_target_latent_codes_full,], dim=-1)  # OTM
                re_residuals_partial = re_order_decoder_partial(re_input_codes)  # bs x 1024 x 3

                params_full, _ = param_decoder_full(src_latent_codes,
                                                    target_latent_codes_full,
                                                    part_latent_codes,
                                                    per_point_full)
                params_partial, _ = param_decoder_partial(src_latent_codes,
                                                          target_latent_codes_partial,
                                                          part_latent_codes,
                                                          per_point_partial)

                # observe the change of match_full and match_partial
                '''
                if epoch % 3 == 0 and i % 10 == 0:
                    p1, _ = torch.max(match_full[0, :, :], dim=-1, keepdim=True)
                    match_full_0 = match_full[0, :, :] / p1
                    writer.add_image("match_full", match_full_0, dataformats='HW',
                                     global_step=epoch * len(loader) + i)
                    p2, _ = torch.max(match_partial[0, :, :], dim=-1, keepdim=True)
                    match_partial_0 = match_partial[0, :, :] / p2
                    writer.add_image("match_partial", match_partial_0, dataformats='HW',
                                     global_step=epoch * len(loader) + i)
                '''

                params_partial = torch.stack(params_partial)
                params_full = torch.stack(params_full)

                '''

                if ((True in torch.isnan(params_partial)) or (True in torch.isinf(params_partial)))\
                        or (True in torch.isnan(params_partial)) or (True in torch.isinf(params_partial)):
                    print("wrong in network")
                    #assert False
                '''

                # using params to get deformed shape
                # def param, original bounding box info
                if cfg["use_connectivity"]:
                    output_pc_from_full = get_shape(mat, params_full, def_param,
                                                    cfg["alpha"], connectivity_mat=conn_mat)
                    output_pc_from_partial = get_shape(mat, params_partial, def_param,
                                                       cfg["alpha"], connectivity_mat=conn_mat)
                else:
                    output_pc_from_full = get_shape(mat, params_full, def_param, cfg["alpha"])
                    output_pc_from_partial = get_shape(mat, params_partial, def_param, cfg["alpha"])
                #  compute losses
                loss_all = 0.0

                # back translate and rotate x_p
                ori_x_p = [ori_x_p.to(cfg["device"], dtype=torch.float) for ori_x_p in ori_point_occ]
                ori_x_p = torch.stack(ori_x_p)

                if cfg["use_chamfer_loss"] > 0.0:
                    # note that here, the deformed shape, no matter from partial or full, should be align to x
                    cd_loss_partial = compute_cm_loss(output_pc_from_partial, x)
                    cd_loss_full = compute_cm_loss(output_pc_from_full, x)
                    loss_all += cd_loss_full * cfg["use_chamfer_loss"]
                    loss_all += cd_loss_partial * cfg["use_chamfer_loss"]
                    # print("cd_loss_full", str(cd_loss_full.item()), "cd_loss_partial", str(cd_loss_partial.item()))
                    writer.add_scalar('cd_loss_full', cd_loss_full.item(), global_step=epoch * len(loader) + i)
                    writer.add_scalar('cd_loss_partial', cd_loss_partial.item(), global_step=epoch * len(loader) + i)
                if cfg["use_symmetry_loss"] > 0.0:
                    ref_pc_partial = get_symmetric(output_pc_from_partial)
                    ref_pc_full = get_symmetric(output_pc_from_full)
                    ref_cd_loss_partial = compute_cm_loss(ref_pc_partial, x)
                    ref_cd_loss_full = compute_cm_loss(ref_pc_full, x)
                    loss_all += ref_cd_loss_full * cfg["use_symmetry_loss"]
                    loss_all += ref_cd_loss_partial * cfg["use_symmetry_loss"]
                    # print("ref_cd_loss_full", str(ref_cd_loss_full.item()), "cd_loss_partial", str(ref_cd_loss_partial.item()))
                    writer.add_scalar('ref_cd_loss_full', ref_cd_loss_full.item(), global_step=epoch * len(loader) + i)
                    writer.add_scalar('ref_cd_loss_partial', ref_cd_loss_partial.item(),
                                      global_step=epoch * len(loader) + i)

                if cfg["use_deformed_pc_consistency"] > 0.0:
                    dpc_con_loss = compute_pc_consistency(output_pc_from_full, output_pc_from_partial)
                    loss_all += dpc_con_loss * cfg["use_deformed_pc_consistency"]
                    # print("dpc_con_loss", str(dpc_con_loss.item()))
                    writer.add_scalar('dpc_con_loss', dpc_con_loss.item(), global_step=epoch * len(loader) + i)

                if cfg["use_residuals_reg"] > 0.0 and epoch > cfg['init_p_m_loss']:
                    re_residual_loss_full, reg_residual_full = residual_retrieval_loss(x,
                                                                                       output_pc_from_full.detach(),
                                                                                       re_residuals_full)
                    re_residual_loss_partial, reg_residual_partial = residual_retrieval_loss(ori_x_p,
                                                                                             output_pc_from_partial.detach(),
                                                                                             re_residuals_partial)

                    loss_all += re_residual_loss_partial * cfg["use_residuals_reg"]
                    loss_all += re_residual_loss_full * cfg["use_residuals_reg"]
                    loss_all += reg_residual_full * cfg["use_residuals_reg"] * 0.01
                    loss_all += reg_residual_partial * cfg["use_residuals_reg"] * 0.01

                    writer.add_scalar('re_reg_loss_full',
                                      re_residual_loss_full.item(), global_step=epoch * len(loader) + i)
                    writer.add_scalar('re_reg_loss_partial', re_residual_loss_partial.item(),
                                      global_step=epoch * len(loader) + i)
                    writer.add_scalar('reg_loss_partial', reg_residual_partial.item(),
                                      global_step=epoch * len(loader) + i)
                    writer.add_scalar('reg_loss_full', reg_residual_full.item(),
                                      global_step=epoch * len(loader) + i)

                if cfg["use_recon"] > 0.0:
                    recon_loss_full = compute_pc_consistency(recon_full_p, x)
                    recon_loss_partial = compute_pc_consistency(recon_partial_p, ori_x_p)
                    loss_all += recon_loss_full * cfg["use_recon"]
                    loss_all += recon_loss_partial * cfg["use_recon"]
                    writer.add_scalar('recon_loss_full',
                                      recon_loss_full.item(), global_step=epoch * len(loader) + i)
                    writer.add_scalar('recon_loss_partial', recon_loss_partial.item(),
                                      global_step=epoch * len(loader) + i)

                    recon_loss_src = compute_pc_consistency(recon_src_p, src_points_cloud)
                    loss_all += recon_loss_src * cfg["use_recon"]
                    writer.add_scalar('recon_loss_src', recon_loss_src.item(),
                                      global_step=epoch * len(loader) + i)

                '''
                # back propagation
                if (True in torch.isnan(loss_all)) or (True in torch.isinf(loss_all)):
                    print("cd_loss_full", str(cd_loss_full.item()), "cd_loss_partial", str(cd_loss_partial.item()))
                    print("ref_cd_loss_full", str(ref_cd_loss_full.item()), "cd_loss_partial",
                          str(ref_cd_loss_partial.item()))
                    print("pm_loss_full", str(pm_loss_full.item()), "pm_loss_partial", str(pm_loss_partial.item()))
                    print("mm_reg_loss_full", str(mm_reg_loss_full.item()), "cd_loss_partial",
                          str(mm_reg_loss_partial.item()))
                    print("mm_con_loss", str(mm_con_loss.item()))
                    print("dpc_con_loss", str(dpc_con_loss.item()))
                    print("param_con_loss", str(param_con_loss.item()))

                    #assert False
                '''
                writer.add_scalar('all_loss', loss_all.item(), global_step=epoch * len(loader) + i)
                optimizer.zero_grad()
                loss_all.backward()
                torch.nn.utils.clip_grad_norm_(target_encoder_full.parameters(), 5.0)
                torch.nn.utils.clip_grad_norm_(target_encoder_partial.parameters(), 5.0)
                torch.nn.utils.clip_grad_norm_(param_decoder_partial.parameters(), 5.0)
                torch.nn.utils.clip_grad_norm_(param_decoder_full.parameters(), 5.0)
                torch.nn.utils.clip_grad_norm_(re_order_decoder_partial.parameters(), 5.0)
                torch.nn.utils.clip_grad_norm_(re_order_decoder_full.parameters(), 5.0)
                torch.nn.utils.clip_grad_norm_(recon_decoder_full.parameters(), 5.0)
                torch.nn.utils.clip_grad_norm_(recon_decoder_partial.parameters(), 5.0)
                torch.nn.utils.clip_grad_norm_(recon_decoder_src.parameters(), 5.0)
                torch.nn.utils.clip_grad_norm_(src_encoder_all.parameters(), 5.0)
                optimizer.step()
            scheduler.step()  # decrease the learning rate by 0.7 every 30 epochs

            # save model
            if ((epoch + 1) % 10 == 0):
                # Summary after each epoch
                summary = {}
                now = datetime.datetime.now()
                duration = (now - start).total_seconds()
                log = "> {} | Epoch [{:04d}/{:04d}] | duration: {:.1f}s |"
                log = log.format(now.strftime("%c"), epoch, cfg["epochs"], duration)

                fname = os.path.join(cfg["log_path"], "checkpoint_{:04d}.pth".format(epoch))
                print("> Saving model to {}...".format(fname))
                model = {"target_encoder_full": target_encoder_full.state_dict(),
                         "target_encoder_partial": target_encoder_partial.state_dict(),
                         "param_decoder_full": param_decoder_full.state_dict(),
                         "param_decoder_partial": param_decoder_partial.state_dict(),
                         "re_residual_net_partial": re_order_decoder_partial.state_dict(),
                         "re_residual_net_full": re_order_decoder_full.state_dict(),
                         "recon_decoder_full": recon_decoder_full.state_dict(),
                         "recon_decoder_partial": recon_decoder_partial.state_dict(),
                         "src_encoder_all": src_encoder_all.state_dict(),
                         "recon_decoder_src": recon_decoder_src.state_dict()}
                torch.save(model, fname)

                fname = os.path.join(cfg["log_path"], "train.log")
                with open(fname, "a") as fp:
                    fp.write(log + "\n")

                print(log)
                print("--------------------------------------------------------------------------")

    else:
        print('go to the test file: test_deform_match!')


if __name__ == '__main__':
    config_path = sys.argv[1]
    config = json.load(open(config_path, 'rb'))
    main(config)
