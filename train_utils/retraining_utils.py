import torch
import torch.nn as nn
from dataset.dataset_utils import get_all_selected_models_pickle, get_random_labels, get_all_source_labels
from dataset.dataset_utils import get_source_info, get_source_latent_codes_fixed, get_shape
from dataset.dataset_utils import get_symmetric

def get_sources_accordingly(source_labels, mode, nofsource, epoch, re_epoch=30):
    if mode == "exhaustive":
        source_labels = get_all_source_labels(source_labels, nofsource)
        return source_labels, None
    elif mode == "random":
        source_labels = get_random_labels(source_labels, nofsource)
        return source_labels, None
    elif mode == "retrieval_candidates" and epoch > re_epoch:
        return None, None
    else:
        source_labels = get_random_labels(source_labels, nofsource)
        return source_labels, None


