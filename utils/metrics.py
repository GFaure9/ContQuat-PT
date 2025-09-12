"""
Functions to compute different metrics for the evaluation of SLP results against ground truth data.
"""

import numpy as np
from typing import List, Tuple
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer


def mpje(seq1: np.ndarray, seq2: np.ndarray) -> float:
    """
    Computes Mean Per Joint Error (MPJE) between two sequences of skeletal poses of same shape (T, nPoints, k)
    where k is generally 2 or 3 (for joints 2D or 3D Euclidean coordinates).

    Parameters
    ----------
    seq1: np.ndarray
        A sequence of skeletal T skeletal poses of the form:
            np.array([ [ [x0, y0, z0], ..., [x_nPoints, y_nPoints, z_nPoints] ]_{t=0},
                            ...
                         [x0, y0, z0], ..., [x_nPoints, y_nPoints, z_nPoints] ]_{t=T} ] ])

    seq2: np.ndarray
        Another sequence of T skeletal poses of the same form as `seq1`.

    Returns
    -------
    float
        The MPJE between `seq1` and `seq2`.
    """
    assert len(seq1) == len(seq2), "Sequences of skeletal poses `seq1` and `seq2` must be of same length"

    distances: np.ndarray = np.linalg.norm(seq1 - seq2, axis=2)  # shape: (T, nPoints)
    return distances.mean()

def compute_dtw_and_align(seq1: np.ndarray, seq2: np.ndarray):
    """
    Parameters
    ----------
    seq1 : np.ndarray
        Skeleton sequence 1 of shape (T1, N, 3).
    seq2 : np.ndarray
        Skeleton sequence 2 of shape (T2, N, 3).

    Returns
    -------
    dtw_distance : float
        The DTW alignment cost between seq1 and seq2.
    aligned_seq2 : np.ndarray
        Sequence seq2 realigned to match seq1's time steps, shape (T1, N, 3).
    """
    T1, N, D = seq1.shape
    T2, _, _ = seq2.shape

    # Compute DTW with L2 distance between skeleton frames
    dtw_distance, path = fastdtw(seq1.reshape(T1, -1), seq2.reshape(T2, -1), dist=euclidean)

    # Warp seq2 to align with seq1
    aligned_seq2 = np.array([seq2[j] for i, j in sorted(path)])

    return dtw_distance, aligned_seq2

def dtw(seq1: np.ndarray, seq2: np.ndarray) -> float:
    """
    Computes Dynamic Time Warping (DTW) between two sequences of skeletal poses of same shape (T, nPoints, k)
    where k is generally 2 or 3 (for joints 2D or 3D Euclidean coordinates).
    Note that DTW is computed with L2 distance here.

    Parameters
    ----------
    seq1: np.ndarray
        A sequence of skeletal T skeletal poses of the form:
            np.array([ [ [x0, y0, z0], ..., [x_nPoints, y_nPoints, z_nPoints] ]_{t=0},
                            ...
                         [x0, y0, z0], ..., [x_nPoints, y_nPoints, z_nPoints] ]_{t=T} ] ])

    seq2: np.ndarray
        Another sequence of T skeletal poses of the same form as `seq1`.

    Returns
    -------
    float
        The DTW between `seq1` and `seq2`.
    """
    T1, N, D = seq1.shape
    T2, _, _ = seq2.shape

    return fastdtw(seq1.reshape(T1, -1), seq2.reshape(T2, -1), dist=euclidean)[0]

def dtw_mpje(seq1: np.ndarray, seq2: np.ndarray) -> float:
    """
    Does the following:
    1) First re-align `seq2` on `seq1` using DTW optimal path
    2) Compute MPJE between `seq1` and aligned `seq2`

    Parameters
    ----------
    seq1: np.ndarray
        A sequence of skeletal T skeletal poses of the form:
            np.array([ [ [x0, y0, z0], ..., [x_nPoints, y_nPoints, z_nPoints] ]_{t=0},
                            ...
                         [x0, y0, z0], ..., [x_nPoints, y_nPoints, z_nPoints] ]_{t=T} ] ])

    seq2: np.ndarray
        Another sequence of T skeletal poses of the same form as `seq1`.

    Returns
    -------
    float
        The MPJE between `seq1` and `seq2` after re-alignment with L2 distance-based DTW.
    """
    _, aligned_seq2 = compute_dtw_and_align(seq1, seq2)
    return mpje(seq1, aligned_seq2[: len(seq1)])

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

def bleu_n(seq1: List[str], seq2: List[str], n: int) -> float:
    """
    Computes BLEU-n score between the ref sequence `seq1` and the hypothesis sequence `seq2`.

    Parameters
    ----------
    seq1: List[str]
        Reference sequence of words (sentence) as a list of words.
        E.g. `seq1 = ['My', 'name', 'is', 'Joe']`.

    seq2: List[str]
        Reference sequence of words (sentence) as a list of words.
        E.g. `seq2 = ["I", "am", "Joe"]`.

    Returns
    -------
    float
        The BLEU-n score between `seq1` (ref) and `seq2` (hypothesis).
    """
    assert n in [1, 2, 3, 4], "`n` must be an integer between 1 and 4"
    weights = {
        1: (1., 0., 0., 0.),  # BLEU-1
        2: (0.5, 0.5, 0., 0.),  # BLEU-2
        3: (0.33, 0.33, 0.33, 0.),  # BLEU-3
        4: (0.25, 0.25, 0.25, 0.25)  # BLEU-4
    }
    return sentence_bleu([seq1], seq2, weights=weights[n])

def rouge_n(seq1: List[str], seq2: List[str], n: int) -> Tuple[float, float, float]:
    """
    Computes ROUGE-n precision, recall and F1 score between
    the ref sequence `seq1` and the hypothesis sequence `seq2`.

    Parameters
    ----------
    seq1: List[str]
        Reference sequence of words (sentence) as a list of words.
        E.g. `seq1 = ['My', 'name', 'is', 'Joe']`.

    seq2: List[str]
        Reference sequence of words (sentence) as a list of words.
        E.g. `seq2 = ["I", "am", "Joe"]`.

    Returns
    -------
    Tuple[float, float, float]
        The ROUGE-n precision, recall and F1 score between `seq1` (ref) and `seq2` (hypothesis).
        In the format: `(precision, recall, F1)`.
    """
    assert n in [1, 2, 3, 4], "`n` must be an integer between 1 and 4"
    score_type = f'rouge{n}'
    scorer = rouge_scorer.RougeScorer([score_type], use_stemmer=True)
    scores = scorer.score(" ".join(seq1), " ".join(seq2))[score_type]
    return scores.precision, scores.recall, scores.fmeasure
