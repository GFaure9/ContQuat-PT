"""
Functions to facilitate the computation of quaternion-based representation of
bones rotations from a skeletal T-pose and target 3D joints coordinates, and inversely.
Allows for computing these representations in a "parallel" fashion for sequences (or batch) of poses
by leveraging numpy operations on arrays.
"""


import numpy as np
from typing import Tuple


def apply_quaternion_rotation(q, v):
    """
    Function to apply an array of quaternions to an array of 3D coordinates vectors:

                                quaternion * vector * quaternion^-1
    """
    # q = (w, x, y, z)
    q_conj = np.hstack((q[:, [0]], -q[:, [1]], -q[:, [2]], -q[:, [3]]))  # conjugates of the quaternions | shape=(M, 4)
    v_quat = np.hstack(
        (np.zeros((len(v), 1)), v)  # N.B: v must be of shape (M, 3)
    )  # convert vectors of v to pure quaternions (0, x, y, z) | shape=(M, 4)

    # q * v * v^-1 (Hamilton product)
    qv = quaternion_multiply(quaternion_multiply(q, v_quat), q_conj)  # shape=(M, 4)

    return qv[:, 1:]  # returning only the vectors parts (x, y, z) | shape=(M, 3)

def quaternion_multiply(q1, q2):
    """Function for piece-wise multiplication of two arrays of quaternions"""
    w1, x1, y1, z1 = (q1[:, [i]] for i in range(4))  # w, x, y, z each one of shape=(M, 1)
    w2, x2, y2, z2 = (q2[:, [i]] for i in range(4))
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2  # shape=(M, 1)
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2  # shape=(M, 1)
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2  # shape=(M, 1)
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2  # shape=(M, 1)
    return np.hstack((w, x, y, z))

def cartesian_to_quaternion_pose(
        skel_pose: np.ndarray,  # array of poses arrays in cartesian coordinates --> shape=(N_skels, N_pts, 3)
        skel_resting_pose: np.ndarray,  # array of T poses arrays in cartesian coordinates --> shape(N_skels, N_pts, 3)
        skel_structure: Tuple[Tuple[int, int, int], ...],  # /!\ order respects tree structure
) -> Tuple[np.ndarray, np.ndarray]:

    # Ensure the input arrays have consistent shapes
    assert len(skel_pose) == len(skel_resting_pose)
    assert len(skel_pose.shape) == len(skel_resting_pose.shape)
    if len(skel_pose.shape) == 2:
        # --- to handle the case where one skel pose only is given as input: shape=(N_pts, 3) --> (1, N_pts, 3)
        skel_pose = np.expand_dims(skel_pose, axis=0)
        skel_resting_pose = np.expand_dims(skel_resting_pose, axis=0)
    # --- check that the skel poses and the resting poses have the same root joint position (Oth point)
    assert np.sum(np.linalg.norm(skel_pose[:, 0, :] - skel_resting_pose[:, 0, :], axis=1)) < 1e-8

    M, N_bones =  skel_pose.shape[0], len(skel_structure)
    quaternions = np.zeros((M, N_bones, 4))
    bones_lengths = np.zeros((M, N_bones, 1))

    for k_bone, (i_parent, i_child, _) in enumerate(skel_structure):

        xc, xp = skel_pose[:, i_child], skel_pose[:, i_parent]  # shape=(M, 3)
        xc0, xp0 = skel_resting_pose[:, i_child], skel_resting_pose[:, i_parent]  # shape=(M, 3)


        b_len, b_len0 = (np.linalg.norm(xc - xp, axis=1, keepdims=True),  # shape=(M, 1)
                         np.linalg.norm(xc0 - xp0, axis=1, keepdims=True))  # shape=(M, 1)

        # temp debug to avoid zero division --------
        v, v0 = (xc - xp) / np.maximum(b_len, 1e-8), (xc0 - xp0) / np.maximum(b_len0, 1e-8)
        # ------------------------------------------

        # 1) compute u vectors
        # --- cross product
        u = np.cross(v0, v, axis=1)
        # --- normalization
        # i/ compute norms
        norms = np.linalg.norm(u, axis=1, keepdims=True)  # shape=(M, 1)
        # ii/ avoid division by zero by normalizing only valid rows
        valid = norms > 1e-6
        u = np.where(valid, u / norms, np.array([1, 0, 0]))  # handle edge case (180° rotation) | shape=(M, 3)

        # 2) computing thetas
        dot_products = np.einsum('ij,ij->i', v, v0)  # shape=(M,)
        theta = np.arccos(dot_products)[:, np.newaxis]  # shape=(M, 1)

        # 3) computing quaternions
        quaternions[:, k_bone, :] = np.hstack((np.cos(theta/2), np.sin(theta/2) * u))  # shape=(M, 4)
        bones_lengths[:, k_bone] = b_len  # shape=(M, 1)

    return quaternions, bones_lengths

def quaternion_to_cartesian_pose(
        skel_quaternions: np.ndarray,  # array of quaternions representations arrays --> shape=(N_skels, N_bones, 4)
        skel_resting_pose: np.ndarray,  # array of T poses arrays in cartesian coordinates --> shape(N_skels, N_pts, 3)
        skel_structure: Tuple[Tuple[int, int, int], ...],  # /!\ order respects tree structure
) -> np.ndarray:

    # Ensure the input arrays have consistent shapes
    assert len(skel_quaternions.shape) == len(skel_resting_pose.shape)
    if len(skel_quaternions.shape) == 2:
        # to handle the case where one skel pose only is given as input: shape=(N_bones, 4) --> (1, N_bones, 4)
        skel_quaternions = np.expand_dims(skel_quaternions, axis=0)
        skel_resting_pose = np.expand_dims(skel_resting_pose, axis=0)

    # Initialize the skeleton poses with root joints at T-poses' root joints
    M, N_pts = skel_resting_pose.shape[:2]
    skel_pose = np.zeros((M, N_pts, 3))
    skel_pose[:, 0, :] = skel_resting_pose[:, 0, :].copy()  # roots (0th joint)

    for k_bone, (i_parent, i_child, _) in enumerate(skel_structure):
        q = skel_quaternions[:, k_bone]  # get the quaternions for the current bone | shape=(M, 4)

        # 1) get vectors from parent to child joint in resting poses
        v0 = skel_resting_pose[:, i_child] - skel_resting_pose[:, i_parent]  # shape=(M, 4)

        # 2) apply the quaternion rotation to the vectors of v0
        rotated_v = apply_quaternion_rotation(q, v0)  # shape=(M, 3)

        # 3) calculate the new position of the child joint
        skel_pose[:, i_child] = skel_pose[:, i_parent] + rotated_v  # shape=(M, 3)

    return skel_pose  # shape=(M, N_pts, 3)

