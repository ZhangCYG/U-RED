import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np

from dataset.dataset_utils import get_all_selected_models_pickle, get_random_labels, get_all_source_labels
from dataset.dataset_utils import get_source_info, get_source_latent_codes_fixed, get_shape
from dataset.dataset_utils import get_symmetric


def compute_mahalanobis(query_vecs, mus, sigmas, activation_fn=None, clip_vec=False):
    if not activation_fn == None and sigmas is not None:
        sigmas = activation_fn(sigmas) + 1.0e-6

    if clip_vec:
        query_vecs = query_vecs.clamp(-100.0, 100.0)

    if sigmas is not None:
        queries_normalized = torch.square(torch.mul((query_vecs - mus), sigmas))
    else:
        queries_normalized = torch.square((query_vecs - mus))

    distances = torch.sum(queries_normalized, dim=-1)

    return distances



