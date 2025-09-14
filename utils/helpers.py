"""
Collection of helper functions
"""

import copy
import glob
import yaml
import os
import os.path
import errno
import shutil
import random
import logging
import torch
import numpy as np
from torch import nn, Tensor
from logging import Logger
from typing import Optional, Tuple, List, Union
from tqdm import tqdm

from ..losses.dtw import dtw


class ConfigurationError(Exception):
    """ Custom exception for misspecifications of configuration """

def make_model_dir(model_dir: str, overwrite=False, model_continue=False) -> str:
    """
    Create a new directory for the model.

    :param model_dir: path to model directory
    :param overwrite: whether to overwrite an existing directory
    :param model_continue: whether to continue from a checkpoint
    :return: path to model directory
    """
    # If model already exists
    if os.path.isdir(model_dir):

        # If model continuing from checkpoint
        if model_continue:
            # Return the model_dir
            return model_dir

        # If set to not overwrite, this will error
        if not overwrite:
            raise FileExistsError("Model directory exists and overwriting is disabled.")

        # If over-write, recursively delete previous directory to start with empty dir again
        for file in os.listdir(model_dir):
            file_path = os.path.join(model_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        shutil.rmtree(model_dir, ignore_errors=True)

    # If model directly doesn't exist, make it and return
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    return model_dir


def make_logger(model_dir: str, log_file: str = "train.log") -> Logger:
    """
    Create a logger for logging the training process.

    :param model_dir: path to logging directory
    :param log_file: path to logging file
    :return: logger object
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.DEBUG)
    fh = logging.FileHandler("{}/{}".format(model_dir, log_file))
    fh.setLevel(level=logging.DEBUG)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(message)s')
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    logging.getLogger("").addHandler(sh)
    logger.info("Progressive Transformers for End-to-End SLP")
    return logger


def log_cfg(cfg: dict, logger: Logger, prefix: str = "cfg") -> None:
    """
    Write configuration to log.

    :param cfg: configuration to log
    :param logger: logger that defines where log is written to
    :param prefix: prefix for logging
    """
    for k, v in cfg.items():
        if isinstance(v, dict):
            p = '.'.join([prefix, k])
            log_cfg(v, logger, prefix=p)
        else:
            p = '.'.join([prefix, k])
            logger.info("{:34s} : {}".format(p, v))


def clones(module: nn.Module, n: int) -> nn.ModuleList:
    """
    Produce N identical layers. Transformer helper function.

    :param module: the module to clone
    :param n: clone this many times
    :return cloned modules
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def subsequent_mask(size: int) -> Tensor:
    """
    Mask out subsequent positions (to prevent attending to future positions)
    Transformer helper function.

    :param size: size of mask (2nd and 3rd dim)
    :return: Tensor with 0s and 1s of shape (1, size, size)
    """
    mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')

    return torch.from_numpy(mask) == 0 # Turns it into True and False's


# Subsequent mask of two sizes
def uneven_subsequent_mask(x_size: int, y_size: int) -> Tensor:
    """
    Mask out subsequent positions (to prevent attending to future positions)
    Transformer helper function.

    :param size: size of mask (2nd and 3rd dim)
    :return: Tensor with 0s and 1s of shape (1, size, size)
    """
    mask = np.triu(np.ones((1, x_size, y_size)), k=1).astype('uint8')
    return torch.from_numpy(mask) == 0  # Turns it into True and False's


def set_seed(seed: int) -> None:
    """
    Set the random seed for modules torch, numpy and random.

    :param seed: random seed
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_config(path="configs/default.yaml") -> dict:
    """
    Loads and parses a YAML configuration file.

    :param path: path to YAML configuration file
    :return: configuration dictionary
    """
    with open(path, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg


def bpe_postprocess(string) -> str:
    """
    Post-processor for BPE output. Recombines BPE-split tokens.

    :param string:
    :return: post-processed string
    """
    return string.replace("@@ ", "")


def get_latest_checkpoint(ckpt_dir, post_fix="_every" ) -> Optional[str]:
    """
    Returns the latest checkpoint (by time) from the given directory, of either every validation step or best
    If there is no checkpoint in this directory, returns None

    :param ckpt_dir: directory of checkpoint
    :param post_fixe: type of checkpoint, either "_every" or "_best"

    :return: latest checkpoint file
    """
    # Find all the every validation checkpoints
    list_of_files = glob.glob("{}/*{}.ckpt".format(ckpt_dir,post_fix))
    latest_checkpoint = None
    if list_of_files:
        latest_checkpoint = max(list_of_files, key=os.path.getctime)
    return latest_checkpoint


def load_checkpoint(path: str, use_cuda: bool = True) -> dict:
    """
    Load model from saved checkpoint.

    :param path: path to checkpoint
    :param use_cuda: using cuda or not
    :return: checkpoint (dict)
    """
    assert os.path.isfile(path), "Checkpoint %s not found" % path
    checkpoint = torch.load(path, map_location='cuda' if use_cuda else 'cpu')
    return checkpoint


def freeze_params(module: nn.Module) -> None:
    """
    Freeze the parameters of this module,
    i.e. do not update them during training

    :param module: freeze parameters of this module
    """
    for _, p in module.named_parameters():
        p.requires_grad = False


def symlink_update(target, link_name):
    try:
        os.symlink(target, link_name)
    except FileExistsError as e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(target, link_name)
        else:
            raise e


# Find the best timing match between a reference and a hypothesis, using DTW
def calculate_dtw(
        references,
        hypotheses,
        cost_func_type="euclidian_norm",
        skel_structure: Tuple[Tuple[int, int, int], ...] = None,
        return_mbae: bool = False,
) -> Union[List[float], Tuple[List[float], List[float]]]:
    """
    Calculate the DTW costs between a list of references and hypotheses

    :param references: list of reference sequences to compare against
    :param hypotheses: list of hypothesis sequences to fit onto the reference
    :param cost_func_type: str indicating which cost function to take for DTW. Available options are:
                 - 'euclidian_norm'
                 - 'angle_error' (N.B: scores will be in radians)
    :param skel_structure: (optional) skeletal structure indicating (parent, child) node IDs
    in references/hypotheses poses
    :param return_mbae: (optional) whether to return the DTW corrected Mean Bone Angle error (in degrees) for
    each pair of sequences of the batch. If True, these values will be returned in a list as the second element
    of an output tuple

    :return: dtw_scores: list of DTW costs
    """
    # Cost function
    if cost_func_type == "euclidian_norm":
        cost_func = lambda x, y: np.sum(np.abs(x - y))
    elif cost_func_type == "angle_error":
        cost_func = lambda x, y: angle_error(x, y, skel_structure=skel_structure, reduction="sum")
    else:
        raise ValueError(f"{cost_func_type}' is not a valid type for `cost_func_type`")

    dtw_scores = []
    mbae_scores = []

    # Remove the BOS frame from the hypothesis
    hypotheses = hypotheses[:, 1:]

    # For each reference in the references list
    for i, ref in tqdm(enumerate(references), total=len(references)):
        # Cut the reference down to the max count value
        _ , ref_max_idx = torch.max(ref[:, -1], 0)
        if ref_max_idx == 0: ref_max_idx += 1
        # Cut down frames by to the max counter value, and chop off counter from joints
        ref_count = ref[:ref_max_idx,:-1].cpu().numpy()

        # Cut the hypothesis down to the max count value
        hyp = hypotheses[i]
        _, hyp_max_idx = torch.max(hyp[:, -1], 0)
        if hyp_max_idx == 0: hyp_max_idx += 1
        # Cut down frames by to the max counter value, and chop off counter from joints
        hyp_count = hyp[:hyp_max_idx,:-1].cpu().numpy()

        # Calculate DTW of the reference and hypothesis, using chosen cost function
        d, cost_matrix, acc_cost_matrix, path = dtw(ref_count, hyp_count, dist=cost_func)

        if return_mbae:
            # --- mean bone angle error (in degrees °)
            # [average over temporal path length and number of bones (BAE-DTW / (T*N_bones) )]
            mbae_scores.append((180/np.pi) * d / (len(path[0]) * len(skel_structure)))

        # Normalise the dtw cost by sequence length
        d = d/acc_cost_matrix.shape[0]

        dtw_scores.append(d)

    # Return dtw scores
    if return_mbae:
        return dtw_scores, mbae_scores
    else:
        return dtw_scores


def angle_error(
        x: np.ndarray,  # shape=(N_pts * 3,)
        y: np.ndarray,  # shape=(N_pts * 3,)
        skel_structure: Tuple[Tuple[int, int, int], ...],
        reduction: str ="sum",
) -> float:
    # 1) computing bones vectors
    x_pts, y_pts = x.reshape((-1, 3)), y.reshape((-1, 3))
    ux = np.array([x_pts[iC] - x_pts[iP] for iP, iC, _ in skel_structure])
    uy = np.array([y_pts[iC] - y_pts[iP] for iP, iC, _ in skel_structure])

    # 2) computing angles between bones vectors
    # --- dot products and norms
    dot_products = np.einsum('ij,ij->i', ux, uy)
    norms_ux = np.linalg.norm(ux, axis=1)
    norms_uy = np.linalg.norm(uy, axis=1)
    # --- compute <ux_k,uy_k> / |ux_k| |uy_k| avoiding division by zero
    cos_angles = np.clip(dot_products / (norms_ux * norms_uy + 1e-8), -1.0, 1.0)
    # --- angles in radians
    angles = np.arccos(cos_angles)

    if reduction == "sum":
        return angles.sum()
    elif reduction == "mean":
        return angles.mean()
    else:
        return angles


def get_bounding_boxes_radii(seq: np.ndarray) -> np.ndarray:
    """
    With `seq` a sequence of skeletal poses of shape (T, N_pts, 3) or (T, N_pts, 2), computes the
    largest radii of the (x, y) bounding boxes of all N_pts joints.
    Thus, output will be of shape (T, N_pts).
    """
    min_xys, max_xys = np.min(seq[:, :, :2], axis=0), np.max(seq[:, :, :2], axis=0)
    return np.linalg.norm(max_xys - min_xys, axis=1)


def pck(seq1: np.ndarray, seq2: np.ndarray, alpha: float = 0.2) -> float:
    """
    Computes Probability of Correct Key-point (PCK) between ref and hypothesis sequences
    of skeletal poses of same shape (T, nPoints, k) where
    k is generally 2 or 3 (for joints 2D or 3D Euclidean coordinates).

    Parameters
    ----------
    seq1: np.ndarray
        The ref (ground truth) sequence of skeletal T skeletal poses of the form:
            np.array([ [ [x0, y0, z0], ..., [x_nPoints, y_nPoints, z_nPoints] ]_{t=0},
                            ...
                         [x0, y0, z0], ..., [x_nPoints, y_nPoints, z_nPoints] ]_{t=T} ] ])

    seq2: np.ndarray
        The hypothesis (predicted) sequence of T skeletal poses of the same form as `seq1`.

    alpha: float, optional
        The ratio of the largest dimension of a seq1 joint J_seq1 bounding box to consider as the radius
        of the sphere within which a joint J_seq2 has two fall to be considered correct.
        I.e. {J_seq2 is correct} <=> |J_seq2 - J_seq1| < alpha * d_largest
        `alpha` takes values between 0 and 1.

    Returns
    -------
    float
        The PCK - mean score - between `seq1` (ref) and `seq2` (hypothesis).
    """
    assert len(seq1) == len(seq2), "Sequences of skeletal poses `seq1` and `seq2` must be of same length"

    distances: np.ndarray = np.linalg.norm(seq1 - seq2, axis=2)  # shape: (T, nPoints)
    keypoint_overlap = (distances <= get_bounding_boxes_radii(seq1) * alpha)

    return np.mean(keypoint_overlap)


# Apply DTW to the produced sequence, so it can be visually compared to the reference sequence
def alter_DTW_timing(pred_seq,ref_seq):

    # Define a cost function
    euclidean_norm = lambda x, y: np.sum(np.abs(x - y))

    # Cut the reference down to the max count value
    _ , ref_max_idx = torch.max(ref_seq[:, -1], 0)
    if ref_max_idx == 0: ref_max_idx += 1
    # Cut down frames by counter
    ref_seq = ref_seq[:ref_max_idx,:].cpu().numpy()

    # Cut the hypothesis down to the max count value
    _, hyp_max_idx = torch.max(pred_seq[:, -1], 0)
    if hyp_max_idx == 0: hyp_max_idx += 1
    # Cut down frames by counter
    pred_seq = pred_seq[:hyp_max_idx,:].cpu().numpy()

    # Run DTW on the reference and predicted sequence
    d, cost_matrix, acc_cost_matrix, path = dtw(ref_seq[:,:-1], pred_seq[:,:-1], dist=euclidean_norm)

    # Normalise the dtw cost by sequence length
    d = d / acc_cost_matrix.shape[0]

    # Initialise new sequence
    new_pred_seq = np.zeros_like(ref_seq)
    # j tracks the position in the reference sequence
    j = 0
    skips = 0
    squeeze_frames = []
    for (i, pred_num) in enumerate(path[0]):

        if i == len(path[0]) - 1:
            break

        if path[1][i] == path[1][i + 1]:
            skips += 1

        # If a double coming up
        if path[0][i] == path[0][i + 1]:
            squeeze_frames.append(pred_seq[i - skips])
            j += 1
        # Just finished a double
        elif path[0][i] == path[0][i - 1]:
            new_pred_seq[pred_num] = avg_frames(squeeze_frames)
            squeeze_frames = []
        else:
            new_pred_seq[pred_num] = pred_seq[i - skips]

    return new_pred_seq, ref_seq, d

# Find the average of the given frames
def avg_frames(frames):
    frames_sum = np.zeros_like(frames[0])
    for frame in frames:
        frames_sum += frame

    avg_frame = frames_sum / len(frames)
    return avg_frame