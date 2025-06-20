# ============= temporary =============
import sys

sys.path.append("../../../../../sl-features-extractor")  # change to the path to the root path where xfeat-sl is
sys.path.append("../../../..")  # root of the sign-language production repo

# N.B: you should make sure before that all dependencies of xfeat-sl have been installed

# Remark: xfeat-sl will be made an installable package in the future...

# Remark2: in PyCharm, to make it recognizable, I do the following:
#           1) Open File → Settings (or Preferences on macOS)
#           2) Navigate to Project: <your_project> → Project Structure
#           3) Click Add Content Root and select ../../../../../sl-features-extractor
#           4) Click Apply and OK
# =====================================


import yaml
import argparse
import logging
from pipeline import Phoenix14TPTPipeline
from xfeatsl.logging_config import setup_logger


def main(
        output_folder,
        steps,
        subsets,
        paths,
        skel_extraction_cfg=None,
        parallelize_skel_extraction=False,
        cpu=1,
        n_verif_videos=None,
        quaternions_verif=False,
        sentences_embeddings_pretrained_model=None,
):
    pipeline = Phoenix14TPTPipeline(
        output_folder=output_folder,
        steps=steps,
        subsets=subsets,
        paths=paths,
        skel_extraction_cfg=skel_extraction_cfg,
        n_verif_videos=n_verif_videos,
        quaternions_verif=quaternions_verif,
        sentences_embeddings_pretrained_model=sentences_embeddings_pretrained_model,
    )
    pipeline.run(parallelize_skel_extraction=parallelize_skel_extraction, cpu=cpu)


if __name__ == "__main__":
    cfg_filepath = "./configs/pipeline_preparation_dataset.yaml"

    parser = argparse.ArgumentParser(description="Run the pipeline for Phoenix14T dataset preparation for PT model.")
    parser.add_argument("--cfg", type=str, default=cfg_filepath, help="Path to configuration file")
    args = parser.parse_args()

    with open(args.cfg, "r") as file:
        cfg = yaml.safe_load(file)

    logger = setup_logger(logging.INFO)
    main(**cfg)