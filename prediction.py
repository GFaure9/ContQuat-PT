import numpy as np
import math

import torch
from torchtext.data import Dataset
from typing import Tuple
from tqdm import tqdm

from .utils.helpers import calculate_dtw
from .architecture.model import Model
from .data.batch import Batch
from .data.data import make_data_iter
from .data.constants import PAD_TOKEN
from .utils.plot_videos import alter_DTW_timing

from utils.metrics import pck
from utils.skeletal_representations import quaternion_to_cartesian_pose, cartesian_to_quaternion_pose
from utils.skeleletal_structures_helper import generate_t_pose, ORIGINAL_S2SL_SKEL


# Validate epoch given a dataset
def validate_on_data(model: Model,
                     data: Dataset,
                     batch_size: int,
                     max_output_length: int,
                     eval_metric: str,
                     loss_function: torch.nn.Module = None,
                     batch_type: str = "sentence",
                     type = "val",
                     BT_model = None,
                     bones_lengths: np.ndarray = None,
                     body_ids: Tuple[int, ...] = None,
                     left_hand_ids: Tuple[int, ...] = None,
                     right_hand_ids: Tuple[int, ...] = None,
                     body_aligned_joint_id: int = 1,
                     hand_aligned_joint_id: int = 0,
                     skel_structure: Tuple[Tuple[int, int, int], ...] = None,
                     only_training_metrics: bool = False):

    valid_iter = make_data_iter(
        dataset=data, batch_size=batch_size, batch_type=batch_type,
        shuffle=True, train=False)

    pad_index = model.src_vocab.stoi[PAD_TOKEN]
    # disable dropout
    model.eval()
    # don't track gradients during validation
    with (torch.no_grad()):
        valid_hypotheses = []
        valid_references = []
        valid_inputs = []
        file_paths = []
        all_dtw_scores = []
        all_pcks = []  # added by GF based on S2SL
        all_dtw_mje = []  # added by GF
        all_dtw_scores_by_part = {"body": [], "left_hand": [], "right_hand": []}  # added by GF
        all_dtw_mje_by_part = {"body": [], "left_hand": [], "right_hand": []}  # added by GF
        all_bae_dtw_scores = []  # added by GF
        all_mbae_scores = []  # added by GF

        valid_loss = 0
        total_ntokens = 0
        total_nseqs = 0

        batches = 0
        for valid_batch in iter(valid_iter):
            # Extract batch
            batch = Batch(torch_batch=valid_batch,
                          pad_index = pad_index,
                          model = model)
            targets = batch.trg

            # run as during training with teacher forcing
            if loss_function is not None and batch.trg is not None:
                # Get the loss for this batch
                batch_loss, _ = model.get_loss_for_batch(batch, loss_function=loss_function)

                valid_loss += batch_loss
                total_ntokens += batch.ntokens
                total_nseqs += batch.nseqs

            # If not just count in, run inference to produce translation videos
            if not model.just_count_in:
                # Run batch through the model in an auto-regressive format
                output, attention_scores = model.run_batch(batch=batch, max_output_length=max_output_length)

            # If future prediction
            if model.future_prediction != 0:
                # Cut to only the first frame prediction + add the counter
                train_output = torch.cat((train_output[:, :, :train_output.shape[2] // (model.future_prediction)], train_output[:, :, -1:]),dim=2)
                # Cut to only the first frame prediction + add the counter
                targets = torch.cat((targets[:, :, :targets.shape[2] // (model.future_prediction)], targets[:, :, -1:]),dim=2)

            # For just counter, the inference is the same as GTing
            if model.just_count_in:
                output = train_output

            # Added by me (GF) / not sure about that
            output = output[:, :-1, :]  # cut off the percent_tok (?)
            assert targets.shape == output.shape  # if skels: (N_batch, T, N_pts * 3 + 1[counter])

            # Added by me (GF) --- handle 'skels' case with imposed bones lengths (will convert to Quat)
            # =============================================================================
            if (bones_lengths is not None) and not model.trg_is_quat:  # if `bones_lengths` is not None and targets are Cartesian coordinates
                N_batch, T = targets.shape[:2]
                N_bones = len(ORIGINAL_S2SL_SKEL)
                N_pts = N_bones + 1

                targets_quat = torch.zeros((N_batch, T, 3 + N_bones * 4 + 1))  # originally (N_batch, T, 200) with the counter
                outputs_quat = torch.zeros((N_batch, T, 3 + N_bones * 4 + 1))

                print("Converting skeletal poses sequences from cartesian coordinates to quaternions...")

                for i in tqdm(range(len(targets))):
                    sequence_target, sequence_output = targets[i], output[i]  # shape=(T, 3 * N_pts + 1)
                    assert sequence_output.shape[0] == T

                    # 0/ get the frame id after which all frames are zero
                    non_zero = np.any(sequence_target.cpu().numpy() != 0.0, axis=1)
                    last_t_non_zero = -1 if not np.any(non_zero) else len(non_zero) - np.argmax(non_zero[::-1]) - 1

                    # 1/ extracting roots and counter values + computing bones lengths
                    poses_trg = sequence_target[:, :-1].reshape(T, -1, 3)  # shape=(T, N_pts, 3)
                    poses_out = sequence_output[:, :-1].reshape(T, -1, 3)
                    assert poses_trg.shape[1] == poses_out.shape[1] == N_pts

                    # --- roots + counter
                    root_pts_trg = poses_trg[:, 0]  # shape=(T, 3)
                    root_pts_out = poses_out[:, 0]

                    counter_trg = sequence_target[:, -1:]  # shape=(T, 1)
                    counter_out = sequence_output[:, -1:]

                    # --- bones lengths
                    bones_vectors_trg = np.array(
                        [poses_trg.cpu()[:, iChild] - poses_trg.cpu()[:, iParent] for iChild, iParent, _ in ORIGINAL_S2SL_SKEL]
                    ).transpose(1, 0, 2)  # shape=(T, N_bones, 3)
                    bones_lengths_trg = np.linalg.norm(bones_vectors_trg, axis=2)  # shape=(T, N_bones)

                    bones_vectors_out = np.array(
                        [poses_out.cpu()[:, iChild] - poses_out.cpu()[:, iParent] for iChild, iParent, _ in ORIGINAL_S2SL_SKEL]
                    ).transpose(1, 0, 2)  # shape=(T, N_bones, 3)
                    bones_lengths_out = np.linalg.norm(bones_vectors_out, axis=2)  # shape=(T, N_bones)

                    # 2/ computing binding poses
                    resting_poses_trg = generate_t_pose(
                        skel_name="original_s2sl_skel",
                        bones_lengths=bones_lengths_trg,
                        root_pt=root_pts_trg.cpu().numpy(),
                    )  # shape=(T, N_pts, 3)

                    resting_poses_out = generate_t_pose(
                        skel_name="original_s2sl_skel",
                        bones_lengths=bones_lengths_out,
                        root_pt=root_pts_out.cpu().numpy(),
                    )

                    # 3/ getting quaternions from skeletal poses and resting poses
                    quaternions_trg = cartesian_to_quaternion_pose(
                        skel_pose=poses_trg.cpu().numpy(),
                        skel_resting_pose=resting_poses_trg,
                        skel_structure=ORIGINAL_S2SL_SKEL,
                    )[0]  # shape=(T, N_bones, 4)

                    quaternions_out = cartesian_to_quaternion_pose(
                        skel_pose=poses_out.cpu().numpy(),
                        skel_resting_pose=resting_poses_out,
                        skel_structure=ORIGINAL_S2SL_SKEL,
                    )[0]

                    # 4/ going back to (T, 3 + N_bones * 4 + 1) format
                    targets_seq = torch.from_numpy(quaternions_trg.reshape(T, -1))
                    targets_seq = torch.cat((root_pts_trg.cpu(), targets_seq, counter_trg.cpu()), dim=1)

                    outputs_seq = torch.from_numpy(quaternions_out.reshape(T, -1))
                    outputs_seq = torch.cat((root_pts_out.cpu(), outputs_seq, counter_out.cpu()), dim=1)

                    # 5/ adding to `targets_quat` and `outputs_quat`
                    targets_quat[i, :last_t_non_zero + 1] = targets_seq[:last_t_non_zero + 1]
                    outputs_quat[i] = outputs_seq

                # 6/ updating `targets` and `outputs` to be in the quat format
                #    (will be then put in skel format with given `bones_lengths` in the next IF statement)
                targets = targets_quat.clone()  # (N_batch, T, 3 + N_bones * 4 + 1)
                output = outputs_quat.clone()

                print("Conversion finished!\nPoses will be recomputed with given `bones_lengths`...")
            # =============================================================================

            # Added by me (GF) --- handle 'quat' case (OR Cartesian transformed to Quat before)
            # =============================================================================
            # (in this case we transform `targets` and `outputs` into cartesian coordinates)
            if model.trg_is_quat or (bones_lengths is not None):

                # output = output[:, :-1, :]  # cut off the percent_tok (?)
                # assert targets.shape == output.shape  # N.B: `targets` shape is (N_batch, T, 3 + N_bones * 4 + 1)

                N_batch, T = targets.shape[:2]
                N_bones = len(ORIGINAL_S2SL_SKEL)
                N_pts = N_bones + 1

                targets_skel = torch.zeros((N_batch, T, N_pts * 3 + 1))  # originally (N_batch, T, 151) with the counter
                outputs_skel = torch.zeros((N_batch, T, N_pts * 3 + 1))

                bones_lengths = bones_lengths if bones_lengths is not None else DEFAULT_BONES_LENGTHS  # shape=(N_bones,)

                for i in range(len(targets)):
                    sequence_target, sequence_output = targets[i], output[i]  # shape=(T, 3 + N_bones * 4 + 1)
                    assert sequence_output.shape[0] == T

                    # 1/ extracting roots, quaternions and counter values
                    quaternions_trg = sequence_target[:, 3:-1].reshape(T, -1, 4)  # shape=(T, N_bones, 4)
                    quaternions_out = sequence_output[:, 3:-1].reshape(T, -1, 4)
                    assert quaternions_trg.shape[1] == quaternions_out.shape[1] == N_bones

                    root_pts_trg = sequence_target[:, :3]  # shape=(T, 3)
                    root_pts_out = sequence_output[:, :3]

                    counter_trg = sequence_target[:, -1:]  # shape=(T, 1)
                    counter_out = sequence_output[:, -1:]

                    # 2/ computing binding poses
                    repeated_bones_lengths = np.tile(bones_lengths, (T, 1))

                    resting_poses_trg = generate_t_pose(
                        skel_name="original_s2sl_skel",
                        bones_lengths=repeated_bones_lengths,
                        root_pt=root_pts_trg.cpu().numpy(),
                    )  # shape=(T, N_pts, 3)

                    resting_poses_out = generate_t_pose(
                        skel_name="original_s2sl_skel",
                        bones_lengths=repeated_bones_lengths,
                        root_pt=root_pts_out.cpu().numpy(),
                    )

                    # 3/ getting skeletal poses from quaternions and resting poses
                    skel_poses_trg = quaternion_to_cartesian_pose(
                        skel_quaternions=quaternions_trg.cpu().numpy(),
                        skel_resting_pose=resting_poses_trg,
                        skel_structure=ORIGINAL_S2SL_SKEL,
                    )  # shape=(T, N_pts, 3)

                    skel_poses_out = quaternion_to_cartesian_pose(
                        skel_quaternions=quaternions_out.cpu().numpy(),
                        skel_resting_pose=resting_poses_out,
                        skel_structure=ORIGINAL_S2SL_SKEL,
                    )

                    # 4/ going back to (T, N_pts * 3 + 1) format
                    targets_seq = torch.from_numpy(skel_poses_trg.reshape(T, -1))
                    targets_seq = torch.cat((targets_seq, counter_trg.cpu()), dim=1)

                    outputs_seq = torch.from_numpy(skel_poses_out.reshape(T, -1))
                    outputs_seq = torch.cat((outputs_seq, counter_out.cpu()), dim=1)

                    # 5/ adding to `targets_skel` and `outputs_skel`
                    targets_skel[i] = targets_seq
                    outputs_skel[i] = outputs_seq

                # 6/ updating `targets` and `outputs` to be in the skel format
                targets = targets_skel.clone()  # (N_batch, T, N_pts * 3 + 1)
                output = outputs_skel.clone()
            # =============================================================================

            # Add references, hypotheses and file paths to list
            valid_references.extend(targets)
            valid_hypotheses.extend(output)
            file_paths.extend(batch.file_paths)
            # Add the source sentences to list, by using the model source vocab and batch indices
            valid_inputs.extend([[model.src_vocab.itos[batch.src[i][j]] for j in range(len(batch.src[i]))] for i in
                                 range(len(batch.src))])

            # Calculate the full Dynamic Time Warping score - for evaluation
            dtw_score = calculate_dtw(targets, output)
            all_dtw_scores.extend(dtw_score)

            # Calculate BAE (Bone Angle Error)-based DTW scores - added by me (GF)
            # + calculate MBAE (Mean Bone Angle Error) after aligning with BAE-based DTW optimal path - added by me (GF)
            if not only_training_metrics:
                print("Computing BAE-based DTW score (and MBAE -mean bones angles error-) for each sequence...")
                bae_dtw_score, mbae_scores = calculate_dtw(
                    targets,
                    output,
                    cost_func_type="angle_error",
                    skel_structure=skel_structure,
                    return_mbae=True,
                )
                all_bae_dtw_scores.extend(bae_dtw_score)
                all_mbae_scores.extend(mbae_scores)
                print("BAE-based DTW score (and MBAE) computation terminated!")

            # Calculate the DTW separately for body, left hand and right hand - without the counter! (added by GF)
            # + calculate the DTW-MJE for each part (added by GF)
            if not only_training_metrics:
                body_ids = body_ids if body_ids is not None else DEFAULT_BODY_IDS
                left_hand_ids = left_hand_ids if left_hand_ids is not None else DEFAULT_LEFT_HAND_IDS
                right_hand_ids = right_hand_ids if right_hand_ids is not None else DEFAULT_RIGHT_HAND_IDS

                for part, ids in zip(["body", "left_hand", "right_hand"], [body_ids, left_hand_ids, right_hand_ids]):
                    # --- retrieve only ids of interest (corresponding to the `part` joints)
                    targets_skel_part = targets[:, :, :-1].reshape(targets.shape[0], targets.shape[1], -1, 3)[:, :, torch.tensor(ids), :]  # shape=(N_batch, T, n_ids, 3)
                    output_skel_part = output[:, :, :-1].reshape(output.shape[0], output.shape[1], -1, 3)[:, :, torch.tensor(ids), :]

                    # --- align neck joints for body OR wrist joints for hands
                    aligned_joint_id = 0  # default
                    if "body" in part:
                        aligned_joint_id = body_aligned_joint_id  # neck id
                    if "hand" in part:
                        aligned_joint_id = hand_aligned_joint_id  # wrist id
                    output_skel_part += (targets_skel_part[:, :, [aligned_joint_id], :] - output_skel_part[:, :, [aligned_joint_id], :])

                    # --- stack the counter back
                    targets_part = torch.cat([targets_skel_part.reshape(targets.shape[0], targets.shape[1], -1), targets[:, :, -1:]], dim=-1)
                    output_part = torch.cat([output_skel_part.reshape(output.shape[0], output.shape[1], -1), output[:, :, -1:]], dim=-1)

                    # --- computing scores
                    dtw_score_part = calculate_dtw(targets_part, output_part)
                    all_dtw_scores_by_part[part].extend(dtw_score_part)

                    # --- compute MJE after aligning with DTW optimal path
                    for b in range(output.shape[0]):  # iterate over batch sequences
                        hyp_part, ref_part, _ = alter_DTW_timing(targets_part[b], output_part[b])
                        T = ref_part.shape[0]
                        diff = ref_part[:, :-1].reshape(T, -1, 3) - hyp_part[:, :-1].reshape(T, -1, 3)  # numpy arrays
                        dtw_mje = np.linalg.norm(diff, axis=2).mean()
                        all_dtw_mje_by_part[part].append(float(dtw_mje))

            # Compute DTW-MJE and PCK (added by GF)
            if not only_training_metrics:
                for b in range(output.shape[0]):
                    hyp, ref, _ = alter_DTW_timing(output[b], targets[b])

                    # Calculate DTW-MJE (mean joint error after euclidian norm-based DTW alignment)
                    dtw_mje = np.linalg.norm(
                        ref[:, :-1].reshape(-1, 50, 3) - hyp[:, :-1].reshape(-1, 50, 3) ,
                        axis=2
                    ).mean()
                    all_dtw_mje.append(float(dtw_mje))

                    # Calculate the PCK for entire vid sequence (added by GF based on S2SL)
                    pck_score = pck(ref[:,:-1].reshape(-1, 50, 3), hyp[:,:-1].reshape(-1, 50, 3))
                    all_pcks.append(pck_score)

            # Can set to only run a few batches
            # if batches == math.ceil(20/batch_size):
            #     break
            batches += 1

        # Dynamic Time Warping scores
        current_valid_score = np.mean(all_dtw_scores)

    return current_valid_score, valid_loss, valid_references, valid_hypotheses, \
           valid_inputs, all_dtw_scores, file_paths, \
           all_pcks, all_dtw_scores_by_part, all_dtw_mje_by_part, \
           all_bae_dtw_scores, all_mbae_scores, \
           all_dtw_mje


DEFAULT_BODY_IDS = tuple(i for i in range(8))
DEFAULT_LEFT_HAND_IDS = tuple(i for i in range(8, 21))
DEFAULT_RIGHT_HAND_IDS = tuple(i for i in range(21, 50))

DEFAULT_BONES_LENGTHS = np.array([
    # ----- neck
    0.3,
    # ----- left shoulder + left arm (upper + fore)
    0.5,
    0.4,
    0.4,
    # ----- right shoulder + right arm (upper + fore)
    0.5,
    0.4,
    0.4,
    # ----- left hand
    # left hand - wrist
    0.05,
    # left hand - palm
    0.08,
    0.08,
    0.08,
    0.08,
    0.08,
    # left hand - 1st finger
    0.05,
    0.05,
    0.05,
    # left hand - 2nd finger
    0.06,
    0.06,
    0.06,
    # left hand - 3rd finger
    0.08,
    0.08,
    0.08,
    # left hand - 4th finger
    0.07,
    0.07,
    0.07,
    # left hand - 5th finger
    0.04,
    0.04,
    0.04,
    # ----- right hand
    # right hand - wrist
    0.05,
    # right hand - palm
    0.08,
    0.08,
    0.08,
    0.08,
    0.08,
    # right hand - 1st finger
    0.05,
    0.05,
    0.05,
    # right hand - 2nd finger
    0.06,
    0.06,
    0.06,
    # right hand - 3rd finger
    0.08,
    0.08,
    0.08,
    # right hand - 4th finger
    0.07,
    0.07,
    0.07,
    # right hand - 5th finger
    0.04,
    0.04,
    0.04,
])

