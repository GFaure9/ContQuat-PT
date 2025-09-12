import subprocess
import logging
import json
import numpy as np
from typing import List, Tuple

from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool

from xfeatsl.features_extraction.video.video_features_extractor import VideoFeaturesExtractor


logger = logging.getLogger(__name__)


def make_videos(examples_ids: List[str], src_images_folder: str, videos_folder: str, fr: int = 30):
    """
    Make MP4 videos from the sequences of images PNG files in corresponding examples IDs source images folders.

    I.e. from folders:

        > example_id1
            - image0001.png
            - image0002.png
            ...
            - imageN.png
        > example_id2.png
        ...
        > example_idM.png

    Generate files:

        - example_id1.mp4
        - example_id2.mp4
        ...
        - example_idM.mp4
    """
    for example_id in examples_ids:
        video_images_folder = f"{src_images_folder}/{example_id}"
        output_video_file = f"{videos_folder}/{example_id.replace('-', '_')}.mp4"
        img_to_vid_ffmpeg_commang = [
            "ffmpeg",
            "-y",  # enforce overwriting if file already exists
            "-framerate", str(fr),  # frame rate
            "-i", f"{video_images_folder}/images%04d.png",  # input file pattern (image0001.png, image0002.png, etc.)
            "-c:v", "libx264",  # video codec
            "-pix_fmt", "yuv420p",  # pixel format for compatibility
            output_video_file  # output video file
        ]
        try:
            subprocess.run(img_to_vid_ffmpeg_commang)
        except Exception as e:
            logger.warning(
                f"The MP4 video for '{example_id}' was not correctly created due to the following exception: {e}"
            )


def write_poses(videos_paths: List[str], fts_dir, cfg, save_filepaths: bool = True, save_errors: bool = True):
    videos_names = [Path(f).stem for f in videos_paths]
    pose_extractor = VideoFeaturesExtractor(extraction_dir=fts_dir)
    pose_extractor.extract(
        filepaths=videos_paths,
        names=videos_names,
        configs=cfg,
        save_filepaths=save_filepaths, save_errors=save_errors,
    )


def extract_skel_poses(
        ext_dir: str,
        videos_filepaths: List[str],
        cfg: dict,
        parallel: bool = False,
        cpu: int = 1
):
    # /!\ N.B : here the parallelization is done over the videos filepaths
    # (since we have only one folder containing all the videos for PHOENIX14T)

    if parallel:
        logger.info(f"Launching parallel computing on {cpu} CPU cores")

        n = len(videos_filepaths)
        cpu = min(n, cpu)
        k = n // cpu

        for i in tqdm(range(k + 1)):
            # 1) Creating sub-list of videos filepaths
            parallel_videos_fpath = videos_filepaths[i * cpu: min((i + 1) * cpu, n)]
            names = [Path(video_fpath).stem for video_fpath in parallel_videos_fpath]

            # 2) Defining the list of features directories (the same repeated)
            parallel_features_dir = [ext_dir] * len(parallel_videos_fpath)

            # 3) Creating the arguments for pool.starmap function on write_poses()
            args_poses = [
                (
                    [video_fpath],  # /!\ N.B: list format is important even if single element
                    features_dir,
                    cfg,
                    False, False  # /!\ N.B: to deactivate writing the same file in parallel
                ) for video_fpath, features_dir in zip(parallel_videos_fpath, parallel_features_dir)
            ]

            # multiprocessing on write_poses()
            with Pool(processes=cpu) as pool:
                logger.info(f"\n\nProcessing the {len(names)} videos:\n{names}")
                pool.starmap(write_poses, args_poses)
                logger.info(f"Finished writing JSON poses files corresponding to the videos sublist!")

    else:
        logger.info(f"Processing (sequentially) the videos to extract poses...")
        write_poses(videos_filepaths, ext_dir, cfg)
        logger.info(f"Finished writing JSON poses files to: {ext_dir}")


def normalize_sequence(
        skel_seq: np.ndarray,  # shape=(T_{seq}, N_pts, 3) where N_pts is the number of joints of the skeleton
        shoulder_to_shoulder_ref_distance: float = 1.,
        shoulder_ids: Tuple[int, int] = (2, 5),  # (leftShoulder_ID, rightShoulder_ID)
        neck_id: int = 1,
):
    """
    Normalize a sequence of skeletons following the pipeline given in the section 3.2 (Gloss to Skeletal Pose Mapping)
    of Stoll at al. BMVC 2018 paper: Sign Language Production using NMT and GANs
    (https://bmva-archive.org.uk/bmvc/2018/contents/papers/0906.pdf)
    """
    # 1) translate by (NeckRef - NeckSkel) with NeckRef = (0, 0, 0)
    skel_seq_trans = skel_seq + (np.zeros(3) - skel_seq[:, neck_id])[:, np.newaxis, :]  # (T_{seq}, N_pts, 3)

    # 2) compute scaling factor: Shoulder_To_Shoulder_Ref / |LeftShoulder_Skel(t=0) - RightShoulder_Skel(t=0)|
    i1, i2 = shoulder_ids
    skel_shoulder_to_shoulder_dist = np.linalg.norm(skel_seq[0, i1] - skel_seq[0, i2])
    scaling_factor = shoulder_to_shoulder_ref_distance / skel_shoulder_to_shoulder_dist

    # 3) compute: NeckRef + (Skel_Trans - NeckRef) * ScalingFactor
    skel_norm = np.zeros(3) + (skel_seq_trans - np.zeros(3)) * scaling_factor

    return skel_norm


def write_skels_file(
        examples_ids: List[str],
        subset: str,
        poses_folder: str,
        dataset_folder: str,
        with_counter: bool = True,
        normalize: bool = True,
        **normalize_kwargs,
):
    """
    Write dataset's `subset`.skels file.
    Each line of the resulting file is a sequence of skeletal poses (cartesian coordinates + counter)
    corresponding to the example ID at the same line in `examples_ids` list.
    """
    if normalize:
        logger.info(f"Skeletal poses will be normalized just before writing them to {subset}.skels file")

    with open(f"{dataset_folder}/{subset}.skels", "w", encoding="utf-8") as f:
        for example_id in tqdm(examples_ids, total=len(examples_ids)):
            json_skel_sequence = f"{poses_folder}/{example_id.replace('-', '_')}.json"
            with open(json_skel_sequence, "r", encoding="utf-8") as file:
                skel_sequence = np.array(json.load(file))  # shape=(T_{seq}, N_pts * 3) i.e. originally (T_{seq}, 150)
                # ---- normalize the sequence
                if normalize:
                    N_pts = skel_sequence.shape[1] // 3  # should be divisible by 3
                    skel_sequence = normalize_sequence(
                        skel_seq=skel_sequence.reshape(-1, N_pts, 3),
                        **normalize_kwargs,
                    ).reshape(-1, N_pts * 3)
                # ---- add counter value at the end of each pose of the sequence
                if with_counter:
                    counter = np.linspace(0, 1, len(skel_sequence)).reshape(-1, 1)
                    skel_sequence = np.hstack((skel_sequence, counter))
                # ---- flatten the sequence
                skel_sequence = skel_sequence.flatten().tolist()
            f.write(" ".join(map(str, skel_sequence)) + "\n")
