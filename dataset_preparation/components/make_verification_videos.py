import os
import io
import json
import logging

import numpy as np
from typing import List
from utils.visualization import make_skel_video, stack_videos
from utils.skeleletal_structures_helper import ORIGINAL_S2SL_SKEL, generate_t_pose
from utils.skeletal_representations import quaternion_to_cartesian_pose

logger = logging.getLogger(__name__)


def make_verif_videos(
        verif_examples_ids: List[str],
        output_folder: str,
        dataset_folder: str,
        subset: str,
        original_videos_folder: str,
        quaternions_verif: bool = False,
        quaternions_with_roots: bool = True,
):
    """
    Resulting output folder will be of the form:
        > `output_folder`
            > exampleID1
                - sample.mp4  (original video + video of animated skeleton with signed text above)
                - text.txt  (the signed text)
                - gloss.txt  (the gloss version of the signed text)
                - skels.json  (the skeletal poses of the Sign Language sequence)
            > ...
            > exampleID{k}
    """
    # 0) Create the general folder for outputs subfolders + the target, source and files filepaths
    os.makedirs(output_folder, exist_ok=True) or logging.info(
        f"'{subset}' dataset verif videos folder: {output_folder}")
    extensions = [".text", ".gloss", ".skels", ".files"]
    text_path, gloss_path, skels_path, files_path = tuple(f"{dataset_folder}/{subset}" + x for x in extensions)

    # ---- for quaternions computation verification
    quat_path = f"{dataset_folder}/{subset}.quat" if quaternions_verif else None

    def safe_open(fpath, mode, encoding):
        if fpath:
            return io.open(fpath, mode=mode, encoding=encoding)
        else:
            return None

    with (
        io.open(text_path, mode='r', encoding='utf-8') as text_file,
        io.open(gloss_path, mode='r', encoding='utf-8') as gloss_file,
        io.open(skels_path, mode='r', encoding='utf-8') as skels_file,
        io.open(files_path, mode='r', encoding='utf-8') as files_file,
        safe_open(quat_path, mode='r', encoding='utf-8') as quat_file
    ):
        # some verifications (safeguards)
        n_files = len(files_file.readlines())
        n_text = len(text_file.readlines())
        n_gloss = len(gloss_file.readlines())
        n_skels = len(skels_file.readlines())
        assert n_files == n_text == n_gloss == n_skels
        if quaternions_verif:
            n_quat = len(quat_file.readlines())
            assert n_quat == n_files
        logger.info("Matching number of examples for all modalities!")
        logger.info(f"Total number of {subset} examples = {n_files}")

        # /!\ making sure the cursor is put back at the first line of each file
        text_file.seek(0)
        gloss_file.seek(0)
        skels_file.seek(0)
        files_file.seek(0)
        if quat_file:
            quat_file.seek(0)

        for lines in zip(
                text_file,
                gloss_file,
                skels_file,
                files_file,
                *(quat_file,) if quat_file is not None else ()  # unpacking trick to dynamically include `quat_file`
        ):
            text_line, gloss_line, skels_line, files_line = lines[:4]

            if quaternions_verif:
                quat_line = lines[4].strip()

            # ---- strip away the "\n" at the end of the line
            text_line, gloss_line, skels_line, files_line = (
                text_line.strip(),
                gloss_line.strip(),
                skels_line.strip(),
                files_line.strip()
            )
            line_id = files_line.split("/")[-1]
            if line_id in verif_examples_ids:
                # ---- split skeletal poses into joint coordinate values
                skels_line = skels_line.split(" ")
                # ---- turn each joint into a float value, with 1e-8 for numerical stability
                skels_line = [(float(joint) + 1e-8) for joint in skels_line]

                # 1) Create a subfolder to store results
                subfolder = f"{output_folder}/{line_id}"
                if not os.path.isdir(subfolder):
                    os.mkdir(subfolder)
                    logger.info(f"Created sample verification folder at: {subfolder}")
                else:
                    logger.info(f"{subfolder} already exists. Using it")

                # 2) Save the text and gloss
                with open(f"{subfolder}/text.txt", "w", encoding="utf-8") as f:
                    f.write(text_line)
                with open(f"{subfolder}/gloss.txt", "w", encoding="utf-8") as f:
                    f.write(gloss_line)

                # 3) Prepare the skeletal poses in the correct format (a numpy array of shape (T, N_pts, 3))
                # ---- assuming that last value is the counter and that the skel has 50 joints
                poses = np.array(skels_line).reshape(-1, 151)[:, :-1].reshape(-1, 50, 3)

                # 4) Save the skeletal poses
                with open(f"{subfolder}/skels.json", "w") as f:
                    json.dump(poses.tolist(), f, indent=4)

                # 5) Creating a video with skeletal motion with signed text
                make_skel_video(
                    skel_sequence=poses,
                    output_folder=subfolder,
                    output_name="poses",
                    structure=ORIGINAL_S2SL_SKEL,
                    fps=30,
                    scale=1 / 4,
                    attach_text=text_line,
                )
                temp_poses_video = f"{subfolder}/poses.mp4"
                logger.info(f"Successfully created skeletal poses video in {subfolder}!")

                # 6) Stacking the previous video to the right of the original clip
                original_video_fpath = f"{original_videos_folder}/{line_id.replace("-", "_")}.mp4"
                stack_videos(
                    vid1=original_video_fpath,
                    vid2=temp_poses_video,
                    output_folder=subfolder,
                    output_name="sample",
                    width=800,
                    height=800,
                    frame_rate=30,
                    txt1="Original Clip",
                    txt2="Dataset Sample",
                )
                logger.info(f"Successfully created skeletal final sample video in {subfolder}!")

                # 7) Performing quaternions computation verification if needed
                if quaternions_verif:
                    # ---- get list of str values + turning them into array of floats (+1e-8 for num. stability)
                    quat_line = quat_line.split(" ")
                    quat_line = [(float(joint) + 1e-8) for joint in quat_line]

                    # ---- assuming that last value is the counter and that the skel has 49 bones
                    if quaternions_with_roots:
                        # ---- 3 first values of each skel representation are the root joint coordinates
                        quaternions = np.array(quat_line).reshape(-1, 3 + 49 * 4 + 1)[:, 3:-1].reshape(-1, 49, 4)
                    else:
                        quaternions = np.array(quat_line).reshape(-1, 49 * 4 + 1)[:, :-1].reshape(-1, 49, 4)
                    assert quaternions.shape[1] == len(ORIGINAL_S2SL_SKEL), "Mismatch between quaternions and skel structure lengths"

                    # ---- get bones lengths for each pose of the sequence
                    bones_vectors = np.array(
                        [poses[:, iChild] - poses[:, iParent] for iChild, iParent, _ in ORIGINAL_S2SL_SKEL]
                    ).transpose(1, 0, 2)  # shape=(T_{seq}, N_bones, 3)
                    bones_lengths = np.linalg.norm(bones_vectors, axis=2)  # shape=(T_{seq}, N_bones)

                    # ---- get root joints coordinates for each pose of the sequence
                    if quaternions_with_roots:
                        root_points = np.array(quat_line).reshape(-1, 3 + 49 * 4 + 1)[:, :3]
                    else:
                        root_points = poses[:, 0].copy()

                    # ---- computing corresponding T-poses (using the sequence of skeletal pose)
                    resting_poses = generate_t_pose(
                        skel_name="original_s2sl_skel",
                        bones_lengths=bones_lengths,
                        root_pt=root_points
                    )

                    # ---- getting poses back from quaternions
                    reconstructed_poses = quaternion_to_cartesian_pose(
                        skel_quaternions=quaternions,
                        skel_resting_pose=resting_poses,
                        skel_structure=ORIGINAL_S2SL_SKEL,
                    )

                    # ---- making a video from the reconstructed poses (cartesian original VS cartesian reconstructed)
                    make_skel_video(
                        skel_sequence=reconstructed_poses,
                        output_folder=subfolder,
                        output_name="reconstructed_poses_from_quat",
                        structure=ORIGINAL_S2SL_SKEL,
                        fps=30,
                        scale=1 / 4,
                    )
                    logger.info(f"Successfully created reconstructed (from quaternions) poses video in {subfolder}!")

                    stack_videos(
                        vid1=temp_poses_video,
                        vid2=f"{subfolder}/reconstructed_poses_from_quat.mp4",
                        output_folder=subfolder,
                        output_name="reconstructed_poses_from_quat",
                        width=800,
                        height=800,
                        frame_rate=30,
                        txt1="Original Poses",
                        txt2="Reconstructed Poses (applying quaternions to T-poses)",
                    )
                    logger.info(f"Successfully created original VS quaternion-based poses video in {subfolder}!")

                # 8) Removing poses.mp4 (which was just an intermediary file to produce other videos)
                if os.path.exists(temp_poses_video):
                    os.remove(temp_poses_video)
                    logger.info(f"Removed temporary file: {temp_poses_video}")
