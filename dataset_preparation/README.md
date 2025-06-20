## General Information

`dataset_preparation` is based on the description of the dataset format given in the **Data** section of the 
[original Progressive Transformers Model repo](https://github.com/BenSaunders27/ProgressiveTransformersSLP)
and uses `xfeatsl` toolbox for skeletal poses extraction.

---
### How to prepare the *Phoenix14T* dataset for PT?

Either open a terminal and run from the repo root:

```commandline
cd scripts/bash_scripts
bash prepare_phoenix14t_pt_dataset.sh --cfg='path/to/your/pipeline/config.yaml'
```

Or inside this `datasets/phoenix14t/phoenix14t_prog_trans/dataset_preparation` folder,
run:

```commandline
python __main__.py --cfg='path/to/your/pipeline/config.yaml'
```

Providing `--cfg` argument is optional: if not provided, 
default `configs/pipeline_preparation_dataset.yaml` will be taken.

---
*Note*:
a typical `config.yaml` file for Phoenix14T dataset preparation for PT model looks
like this:

```yaml
output_folder: "../data"

steps: [
  "get_examples_ids",
  "make_videos",
  "extract_poses",
  "prepro_save_text_file",
  "prepro_save_gloss_file",
  "prepro_save_files_file",
  "prepro_save_skels_file",
  "prepro_save_quat_file",  # optional
  "prepro_save_sbert_file",  # optional
  "make_verif_videos",
  "compute_mean_bones_lengths",  # optional
]

subsets: ["dev", "test", "train"]

paths:
  examples_ids_folder: "examples_ids"
  videos_folder: "videos"
  poses_folder: "skel_sequences"
  dataset_folder: "dataset"
  verif_videos_folder: "verif_videos"
  src_images_folder: "../../raw_data/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px"
  src_annotations_folder: "../../raw_data/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual"

skel_extraction_cfg:
  method: "zelinka2020"
  show_progress_bar: yes
  export_video: no
  verbose: yes

parallelize_skel_extraction: yes
cpu: 24

n_verif_videos:
  test: 10
  dev: 10
  train: 20

quaternions_verif: yes

sentences_embeddings_pretrained_model: "all-MiniLM-L6-v2"  # cf. https://www.sbert.net/docs/sentence_transformer/pretrained_models.html
```

---
**Dev Remarks**

Also note that you can directly use the `Phoenix14TPTPipeline` (in `pipeline.py`) 
and customize it as you want to make your own, more personalized, dataset 
preparation scripts.