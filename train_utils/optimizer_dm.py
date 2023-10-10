import torch

import torch.optim as optim


def define_optimizer_dm(target_encoder_full, target_encoder_partial, param_decoder_full, param_decoder_partial, cfg):
    target_encoder_full_params = list(target_encoder_full.parameters())
    target_encoder_partial_params = list(target_encoder_partial.parameters())
    decoder_full_params = list(param_decoder_full.parameters())
    decoder_partial_params = list(param_decoder_partial.parameters())

    all_parameters = (target_encoder_full_params + target_encoder_partial_params +
                      decoder_partial_params + decoder_full_params)

    if cfg["optimizer"] == 'sgd':
        optimizer = torch.optim.SGD(
            all_parameters,
            lr=cfg["learning_rate"],
            momentum=cfg["momentum"],
            weight_decay=cfg["weight_decay"]
        )
    elif cfg["optimizer"] == 'adam':
        optimizer = torch.optim.Adam(
            all_parameters,
            lr=cfg["learning_rate"],
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=cfg["weight_decay"]
        )
    else:
        return None
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg["lr_stepsize"], gamma=cfg["lr_decay"])
    return optimizer, scheduler


def define_optimizer_dm_re(target_encoder_full, target_encoder_partial, param_decoder_full,
                           param_decoder_partial, re_net, cfg):
    target_encoder_full_params = list(target_encoder_full.parameters())
    target_encoder_partial_params = list(target_encoder_partial.parameters())
    decoder_full_params = list(param_decoder_full.parameters())
    decoder_partial_params = list(param_decoder_partial.parameters())
    re_net_params = list(re_net.parameters())

    all_parameters = (target_encoder_full_params + target_encoder_partial_params +
                      decoder_partial_params + decoder_full_params + re_net_params)

    if cfg["optimizer"] == 'sgd':
        optimizer = torch.optim.SGD(
            all_parameters,
            lr=cfg["learning_rate"],
            momentum=cfg["momentum"],
            weight_decay=cfg["weight_decay"]
        )
    elif cfg["optimizer"] == 'adam':
        optimizer = torch.optim.Adam(
            all_parameters,
            lr=cfg["learning_rate"],
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=cfg["weight_decay"]
        )
    else:
        return None
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg["lr_stepsize"], gamma=cfg["lr_decay"])
    return optimizer, scheduler


def define_optimizer_dm_re_recon(target_encoder_full, target_encoder_partial, param_decoder_full,
                                 param_decoder_partial,
                                 recon_full, recon_partial,
                                 re_net_full, re_net_partial,
                                 src_encoder, recon_src,
                                 cfg):
    target_encoder_full_params = list(target_encoder_full.parameters())
    target_encoder_partial_params = list(target_encoder_partial.parameters())
    decoder_full_params = list(param_decoder_full.parameters())
    decoder_partial_params = list(param_decoder_partial.parameters())
    re_net_full_params = list(re_net_full.parameters())
    re_net_partial_params = list(re_net_partial.parameters())
    recon_full_params = list(recon_full.parameters())
    recon_partial_params = list(recon_partial.parameters())
    src_encoder_params = list(src_encoder.parameters())
    recon_src_params = list(recon_src.parameters())

    all_parameters = (target_encoder_full_params + target_encoder_partial_params +
                      decoder_partial_params + decoder_full_params + re_net_full_params + re_net_partial_params +
                      recon_partial_params + recon_full_params + src_encoder_params + recon_src_params)

    if cfg["optimizer"] == 'sgd':
        optimizer = torch.optim.SGD(
            all_parameters,
            lr=cfg["learning_rate"],
            momentum=cfg["momentum"],
            weight_decay=cfg["weight_decay"]
        )
    elif cfg["optimizer"] == 'adam':
        optimizer = torch.optim.Adam(
            all_parameters,
            lr=cfg["learning_rate"],
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=cfg["weight_decay"]
        )
    else:
        return None
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg["lr_stepsize"], gamma=cfg["lr_decay"])
    return optimizer, scheduler


def define_optimizer_re(re_target_encoder, re_order_decoder, re_src_encoder, variance, sigma, cfg):
    params_emb = []
    if re_target_encoder is not None:
        params_emb = list(re_target_encoder.parameters())
    if re_src_encoder is not None:
        params_emb = params_emb + list(re_src_encoder.parameters())
    if re_order_decoder is not None:
        params_emb = params_emb + list(re_order_decoder.parameters())

    if cfg["optimizer"] == 'sgd':
        optimizer = torch.optim.SGD(
            params_emb,
            lr=cfg["learning_rate"],
            momentum=cfg["momentum"],
            weight_decay=cfg["weight_decay"]
        )
    elif cfg["optimizer"] == 'adam':
        optimizer = torch.optim.Adam(
            params_emb,
            lr=cfg["learning_rate"],
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=cfg["weight_decay"]
        )
    else:
        assert False
    if variance is not None:
        optimizer.add_param_group({"params": variance})
    if sigma is not None:
        optimizer.add_param_group({"params": sigma, "lr": 0.01})  # larger than the initial lr
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg["lr_stepsize"], gamma=cfg["lr_decay"])

    return optimizer, scheduler
