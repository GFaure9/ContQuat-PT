import os
import glob
import random
import logging

from typing import List, Dict

from utils.textual_embeddings import load_pretrained_sentence_embeddings_model

from components import (
load_annotations,
make_videos,
extract_skel_poses,
write_files_file, write_text_file, write_gloss_file, write_skels_file,
load_skel_sequences, cart_to_quat, write_quat_file,
write_sbert_file,
make_verif_videos,
compute_mean_bones_lengths,
)

logger = logging.getLogger(__name__)


class Phoenix14TPTPipeline:

    default_steps = [
        "get_examples_ids",
        "make_videos",
        "extract_poses",
        "prepro_save_text_file",
        "prepro_save_gloss_file",
        "prepro_save_files_file",
        "prepro_save_skels_file",
        "prepro_save_quat_file",
        "prepro_save_sbert_file",
        "make_verif_videos",
        "compute_mean_bones_lengths",
    ]

    default_subsets = [
        "dev",
        "test",
        "train",
    ]

    default_paths = {
        # ---------- outputs ---------
        "examples_ids_folder": "examples_ids",
        "videos_folder": "videos",
        "poses_folder": "skel_sequences",
        "dataset_folder": "dataset",
        "verif_videos_folder": "verif_videos",
        # ---------- inputs ----------
        "src_images_folder": "path/to/raw/videos/images",
        "src_annotations_folder": "path/to/raw/annotations",
    }

    default_skel_extraction_cfg = {
        "method": "zelinka2020",
        "show_progress_bar": True,
        "export_video": False,
        "verbose": True,
    }

    default_n_verif_videos = {
        "dev": 10,
        "test": 10,
        "train": 20,
    }

    def __init__(
            self,
            output_folder: str = "./data",
            steps: List[str] = None,
            subsets: List[str] = None,
            paths: Dict[str, str] = None,
            skel_extraction_cfg: dict = None,
            n_verif_videos: Dict[str, int] = None,
            quaternions_verif: bool = False,
            sentences_embeddings_pretrained_model: str = None,
    ):

        self.steps = steps if steps else self.default_steps
        self.subsets = subsets if subsets else self.default_subsets
        self.paths = paths if paths else self.default_paths
        self.output_folder = output_folder
        for k in [
            "examples_ids_folder",
            "videos_folder",
            "poses_folder",
            "dataset_folder",
            "verif_videos_folder",
        ]:
            if k in self.paths.keys():
                self.paths[k] = f"{self.output_folder}/{self.paths[k]}"
            else:
                self.paths[k] = f"{self.output_folder}/{self.default_paths[k]}"

        assert set(self.steps).issubset(self.default_steps)
        assert set(self.subsets).issubset(self.default_subsets)
        assert set(self.paths.keys()).issubset(self.default_paths.keys())

        self.skel_extraction_cfg = skel_extraction_cfg if skel_extraction_cfg else self.default_skel_extraction_cfg
        self.n_verif_videos = n_verif_videos if n_verif_videos else self.default_n_verif_videos
        self.quaternions_verif = quaternions_verif

        if "prepro_save_sbert_file" in self.steps:
            default_model = "all-MiniLM-L6-v2"
            if sentences_embeddings_pretrained_model is None:
                logger.warning(
                    f"No pretrained model provided for sentences embeddings. Taking '{default_model}' by default"
                )
        self.sentences_embeddings_pretrained_model = sentences_embeddings_pretrained_model


    def __repr__(self):

        # Retrieving information from attributes
        steps_txt = "Running the following steps:\n" + "\n".join([f"    - {s}" for s in self.steps])
        subsets_txt = f"To build the following subsets: {self.subsets}"
        output_folder_txt = f"All outputs will be stored at: {self.output_folder}"
        paths_txt = f"Outputs and inputs files/folders paths are:\n" + "\n".join(
            [f"    - {k}: {v}" for k, v in self.paths.items()]
        )
        skel_ext_cfg_txt = f"Using the following configuration to estimate skeletal poses from videos:\n{self.skel_extraction_cfg}"

        # Putting all together
        repr_txt = 35 * "*-" + "\n" + "\n\n".join([
            steps_txt,
            subsets_txt,
            output_folder_txt,
            paths_txt,
            skel_ext_cfg_txt,
        ]) + "\n" + 35 * "*-"

        return repr_txt


    def save_pipeline_info(self, output_fpath: str):
        with open(output_fpath, "w") as f:
            f.write(str(self))


    def run(self, parallelize_skel_extraction: bool = True, cpu: int = 5):

        logger.info("Running pipeline for the preparation of ISL News dataset for S2SL model...")
        logger.info(f"\n{self}")

        os.makedirs(self.output_folder, exist_ok=True) or logging.info(f"Output folder: {self.output_folder}")
        os.makedirs(self.paths['examples_ids_folder'], exist_ok=True) or logging.info(f"Examples IDs folder: {self.paths['examples_ids_folder']}")
        os.makedirs(self.paths['videos_folder'], exist_ok=True) or logging.info(f"Videos folder: {self.paths['videos_folder']}")
        os.makedirs(self.paths['poses_folder'], exist_ok=True) or logging.info(f"Poses folder: {self.paths['poses_folder']}")
        os.makedirs(self.paths['verif_videos_folder'], exist_ok=True) or logging.info(f"Verification videos folder: {self.paths['verif_videos_folder']}")

        self.save_pipeline_info(output_fpath=f"{self.output_folder}/dataset_preparation_pipeline_info.txt")

        dataset_folder = self.paths["dataset_folder"]  # final folder for the dataset (prepared for PT)
        os.makedirs(dataset_folder, exist_ok=True) or logging.info(f"Datasets folder: {dataset_folder}")

        sbert_model = load_pretrained_sentence_embeddings_model(self.sentences_embeddings_pretrained_model)

        for subset in self.subsets:

            logger.info(f"Starting pipeline for '{subset}' subset...")

            # 0) Set paths + load annotations
            # -------------------------------------------------------------

            # =========== Sources paths
            src_images_folder = f"{self.paths['src_images_folder']}/{subset}"
            src_annotations_file = f"{self.paths['src_annotations_folder']}/PHOENIX-2014-T.{subset}.corpus.csv"

            # =========== Outputs paths
            examples_ids_file = f"{self.paths['examples_ids_folder']}/{subset}.txt"
            videos_folder = f"{self.paths['videos_folder']}/{subset}"
            poses_folder = f"{self.paths['poses_folder']}/{subset}"
            verif_videos_folder = f"{self.paths['verif_videos_folder']}/{subset}"

            # =========== Loading annotations
            examples_ids, annotations = load_annotations(src_annotations_file)

            # 1) Retrieve + save the examples ids
            # -------------------------------------------------------------
            if "get_examples_ids" in self.steps:
                logger.info("Writing examples IDs to a .txt file...")

                with open(examples_ids_file, "w", encoding="utf-8") as f:
                    f.writelines(f"{example_id}\n" for example_id in examples_ids)

                logger.info(f"Finished writing examples IDs to a .txt file: {examples_ids_file}")

            # 2) Build SL videos from images
            # -------------------------------------------------------------
            if "make_videos" in self.steps:
                logger.info("Making MP4 videos...")

                os.makedirs(videos_folder, exist_ok=True) or f"'{subset}' videos folder: {videos_folder}"

                make_videos(
                    examples_ids=examples_ids,
                    src_images_folder=src_images_folder,
                    videos_folder=videos_folder,
                    fr=30,  # frame rate (default to 30 FPS for now)
                )

                logger.info("Finished making MP4 videos!")

            # 3) Extract the poses
            # -------------------------------------------------------------
            if "extract_poses" in self.steps:
                logger.info("Extracting poses from videos...")

                assert os.path.isdir(videos_folder), "You must run 'make_videos' step first"
                videos_filepaths = glob.glob(f"{videos_folder}/*.mp4")

                assert (videos_filepaths is not None), f"No .mp4 in {videos_folder}. Re-run 'make_videos' step"

                extract_skel_poses(
                    ext_dir=poses_folder,
                    videos_filepaths=videos_filepaths,
                    cfg={"features": {"poses": self.skel_extraction_cfg}},
                    parallel=parallelize_skel_extraction,
                    cpu=cpu,
                )

                logger.info("Finished extracting skel poses (saved in JSON files)!")

            # 4) Prepare + save the .files file
            # -------------------------------------------------------------
            if "prepro_save_files_file" in self.steps:
                logger.info(f"Writing {subset}.files file...")

                write_files_file(examples_ids, subset, dataset_folder)

                logger.info(f"Finished writing to: {dataset_folder}/{subset}.files")

            # 5) Prepare + save the .text file (from the annotations)
            # -------------------------------------------------------------
            if "prepro_save_text_file" in self.steps:
                logger.info(f"Writing {subset}.text file...")

                write_text_file(annotations, examples_ids, subset, dataset_folder)

                logger.info(f"Finished writing to: {dataset_folder}/{subset}.text")

            # 6) Prepare + save the .gloss file
            # -------------------------------------------------------------
            if "prepro_save_gloss_file" in self.steps:
                logger.info(f"Writing {subset}.gloss file...")

                write_gloss_file(annotations, examples_ids, subset, dataset_folder)

                logger.info(f"Finished writing to: {dataset_folder}/{subset}.gloss")

            # 7) Prepare + save the .skels file
            # -------------------------------------------------------------
            if "prepro_save_skels_file" in self.steps:
                logger.info(f"Writing {subset}.skels file...")

                write_skels_file(
                    examples_ids,
                    subset,
                    f"{poses_folder}/poses",
                    dataset_folder,
                    with_counter=True,
                    normalize=True,
                    # --- **normalize_kwargs ---
                    shoulder_to_shoulder_ref_distance=1.,
                    shoulder_ids=(2, 5),
                    neck_id=1,
                )

                logger.info(f"Finished writing to: {dataset_folder}/{subset}.skels")

            # 8) Prepare + save the .quat files
            # -------------------------------------------------------------
            if "prepro_save_quat_file" in self.steps:
                logger.info(f"Starting loading {subset}.skels file...")
                skel_sequences = load_skel_sequences(dataset_folder, subset)  # N.B: it removes the counter
                logger.info(f"Loaded {subset}.skels file!")

                logger.info(f"Starting computing quaternions for each skeletal sequence...")
                root_pts_sequences, quat_sequences = cart_to_quat(skel_sequences)  # skel_structure is `ORIGINAL_S2SL_SKEL` by default
                logger.info("Finished computing quaternions for all sequences!")

                logger.info(f"Starting writing quaternions to: {dataset_folder}/{subset}.quat")
                write_quat_file(
                    quat_sequences,
                    dataset_folder,
                    subset,
                    with_counter=True,
                    root_points_sequences=root_pts_sequences
                )
                logger.info(f"Finished writing quaternions to: {dataset_folder}/{subset}.quat")

            # 9) Compute sentences embeddings with a given `sentence-transformers` pre-trained model
            #     (https://www.sbert.net/docs/sentence_transformer/pretrained_models.html)
            # -------------------------------------------------------------
            if "prepro_save_sbert_file" in self.steps:
                logger.info("Computing sentences embeddings from prepared dataset sentences using...")
                logger.info(f"Using pre-trained model: '{self.sentences_embeddings_pretrained_model}'")

                assert (sbert_model is not None), "`sbert_model` is None. Please provide a model name for `sentences_embeddings_pretrained_model`"
                write_sbert_file(
                    dataset_folder,
                    subset,
                    sentence_transformer_pretrained_model=sbert_model,
                    batch_size=128,
                    show_progress_bar=True,
                )

                logger.info(f"Saved sentences embeddings at: {dataset_folder}/{subset}.sbert")

            logger.info(f"Finished running all steps of pipeline for '{subset}' subset!")

            # 10) Produce some videos to check that dataset is correct
            # -------------------------------------------------------------
            if "make_verif_videos" in self.steps:
                logger.info(f"Generating verification videos from '{subset}' dataset samples...")

                # Select `n_verif_videos` samples randomly
                random.seed(42)
                verif_examples_ids = random.sample(examples_ids, k=self.n_verif_videos[subset])
                logger.info(f"Samples for which to generate verif videos:\n{verif_examples_ids}")

                make_verif_videos(
                    verif_examples_ids=verif_examples_ids,
                    output_folder=verif_videos_folder,
                    dataset_folder=dataset_folder,
                    subset=subset,
                    original_videos_folder=videos_folder,
                    quaternions_verif=self.quaternions_verif, quaternions_with_roots=True,
                )
                logger. info("Finished generating verification videos!")

        # 11) Compute mean bones lengths from prepared dataset
        # -------------------------------------------------------------
        if "compute_mean_bones_lengths" in self.steps:
            logger.info("Computing mean bones lengths from prepared dataset...")

            compute_mean_bones_lengths(dataset_folder, subsets=self.subsets)

            logger.info(f"Saved mean bones lengths at: {dataset_folder}/mean_bones_lengths.txt")

        logger.info("Finished running pipeline on all subsets!")
