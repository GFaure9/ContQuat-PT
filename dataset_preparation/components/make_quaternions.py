import io
import numpy as np

from tqdm import tqdm
from typing import List, Tuple, Dict

from utils.skeleletal_structures_helper import ORIGINAL_S2SL_SKEL, generate_t_pose
from utils.skeletal_representations import cartesian_to_quaternion_pose
from xfeatsl.data_extraction.audio.audio_extractor import logger


def load_skel_sequences(dataset_folder: str, subset: str) -> List[np.ndarray]:
    skels_path = f"{dataset_folder}/{subset}.skels"

    skel_sequences = []
    with io.open(skels_path, mode='r', encoding='utf-8') as skels_file:

        for skels_line in skels_file:
            # ---- strip away the "\n" at the end of the line
            skels_line = skels_line.strip()

            # ---- split skeletal poses into joint coordinate values
            skels_line = skels_line.split(" ")

            # ---- turn each joint into a float value, with 1e-8 for numerical stability
            skels_line = [(float(joint) + 1e-8) for joint in skels_line]

            # ---- turn into a numpy array + remove counter + reshape
            skels_line = np.array(skels_line).reshape(-1, 151)[:, :-1].reshape(-1, 50, 3)

            # ---- append to the list
            skel_sequences.append(np.array(skels_line))

    return skel_sequences


def cart_to_quat(
        skel_sequences: List[np.array],  # M arrays of shape (T_{seq}, N_pts, 3)
        skel_structure: Tuple[Tuple[int, int, int], ...] = ORIGINAL_S2SL_SKEL,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    To go from Cartesian coordinates to Quaternion-based rotations representation of skel sequences.
    """
    quat_sequences = []
    root_points_sequences = []

    for skel_seq in tqdm(skel_sequences, total=len(skel_sequences)):
        # ---- get bones lengths for each pose of the sequence
        bones_vectors = np.array(
            [skel_seq[:, iChild] - skel_seq[:, iParent] for iChild, iParent, _ in skel_structure]
        ).transpose(1, 0, 2)  # shape=(T_{seq}, N_bones, 3)
        bones_lengths = np.linalg.norm(bones_vectors, axis=2)  # shape=(T_{seq}, N_bones)

        # ---- get root joints coordinates for each pose of the sequence
        root_points = skel_seq[:, 0].copy()  # shape=(T_{seq}, 3)

        resting_poses = generate_t_pose(
            skel_name="original_s2sl_skel",  # /!\ Warning: this must match `skel_structure`
            bones_lengths=bones_lengths,
            root_pt=root_points,
        )

        quaternions, _ = cartesian_to_quaternion_pose(
            skel_pose=skel_seq,
            skel_resting_pose=resting_poses,
            skel_structure=skel_structure,
        )  # shape=(T_{seq}, N_bones, 4)

        quat_sequences.append(quaternions)
        root_points_sequences.append(root_points)

    return root_points_sequences, quat_sequences


def write_quat_file(
        quat_sequences: List[np.ndarray],  # each element has a shape=(T_{seq}, N_bones, 4)
        dataset_folder: str,
        subset: str,
        with_counter: bool = True,
        root_points_sequences: List[np.ndarray] = None,
):
    """
    Write dataset's `subset`.quat file.
    Each line is a sequence of skeleton's limbs rotations in the quaternions' space.
    Line {k} in the file matches line {k} in `subset`.skels file (and the other parallel files).
    N.B:
         - without root points, a line is of the form:
                q1(t=0) q2(t=0) ... qNbones(t=0) ... q1(t=0) q2(t=T) ... qNbones(t=T)  with qi = qi1 qi2 qi3 qi4
        - with root points, a line is of the form:
                XRoot(t=0) q1(t=0) ... qNbones(t=0) ... xRoot(t=T) q1(t=T) ... qNbones(t=T)  with xRoot = x y z
    """
    with open(f"{dataset_folder}/{subset}.quat", "w", encoding="utf-8") as f:
        for i, quat_seq in tqdm(enumerate(quat_sequences), total=len(quat_sequences)):  # quat_seq.shape=(T_{seq}, 50, 4)
            # ---- reshape quaternions sequence to be T_{seq} lines
            N_bones, dims = quat_seq.shape[1:]
            quat_seq = quat_seq.reshape(-1, N_bones * dims)
            # ---- add counter value at the end of each quaternion 'pose' of the sequence
            if with_counter:
                counter = np.linspace(0, 1, len(quat_seq)).reshape(-1, 1)
                quat_seq = np.hstack((quat_seq, counter))
            # ---- add at the beginning the 3D coordinates of each pose in the sequence
            if root_points_sequences:
                quat_seq = np.hstack((root_points_sequences[i], quat_seq))
            # ---- flatten the sequence
            quat_seq = quat_seq.flatten().tolist()  # if with roots shape=(T_{seq} * (3 + N_bones * 4)) else (T_{seq} * N_bones * 4)
            # ---- write the line in the quaternions file
            f.write(" ".join(map(str, quat_seq)) + "\n")


def compute_mean_bones_lengths(
        dataset_folder: str,
        subsets: List[str] = None,
        skel_structure: Tuple[Tuple[int, int, int], ...] = ORIGINAL_S2SL_SKEL,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    if subsets is None:
        subsets = ["dev", "test", "train"]

    mean_bones_lengths = {}
    for subset in subsets:
        poses_t0 = np.array(
            [seq[0, :, :] for seq in load_skel_sequences(dataset_folder, subset)])  # shape=(N_data, N_pts, 3)
        bones_vectors = np.array(
            [poses_t0[:, iChild] - poses_t0[:, iParent] for iChild, iParent, _ in skel_structure]
        ).transpose(1, 0, 2)  # shape=(N_data, N_bones, 3)
        bones_lengths = np.linalg.norm(bones_vectors, axis=2)  # shape=(N_data, N_bones)

        mean_bones_lengths[subset] = bones_lengths.mean(axis=0)  # shape=(N_bones,)

        logger.info(f"Computed mean bones lengths for '{subset}' data!")

    global_mean_bones_lengths = np.stack([arr for arr in mean_bones_lengths.values()], axis=0).mean(
        axis=0)  # shape=(N_bones,)
    np.savetxt(f'{dataset_folder}/mean_bones_lengths.txt', global_mean_bones_lengths, fmt='%.6f')

    return mean_bones_lengths, global_mean_bones_lengths
