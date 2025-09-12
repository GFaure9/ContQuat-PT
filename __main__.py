# todo: clean / comment / adapt imports
import argparse

from .training import train, test


def main():

    ap = argparse.ArgumentParser("Progressive Transformers")

    # Choose between Train and Test
    ap.add_argument("--mode", choices=["train", "test"], required=True,
                    help="train a model or test")
    # Path to Config
    ap.add_argument("--config_path", type=str, required=True,
                    help="path to YAML config file")

    # Optional path to checkpoint
    ap.add_argument("--ckpt", type=str, default=None,
                    help="path to model checkpoint")

    # Optional save_skeletal_poses argument (for test)
    ap.add_argument("--save_skeletal_poses", type=bool, default=True,
                    help="whether to store skeletal poses (optional). If True, it stores poses for `n_videos`")

    # Optional produce_videos argument (for test)
    ap.add_argument("--produce_videos", type=bool, default=True,
                    help="whether to generate or no validation videos (ground truth versus prediction) (optional)")

    # Optional n_videos argument (for test)
    ap.add_argument("--n_videos", type=str, default=5,
                    help="number of validation videos (ground truth versus prediction) to generate per subset (optional)")

    args = ap.parse_args()

    # If Train
    if args.mode == "train":
        train(cfg_file=args.config_path, ckpt=args.ckpt)
    # If Test
    elif args.mode == "test":
        test(
            cfg_file=args.config_path,
            ckpt=args.ckpt,
            save_skeletal_poses=args.save_skeletal_poses,
            produce_videos=args.produce_videos,
            n_videos=args.n_videos,
        )
    else:
        raise ValueError("Unknown mode")

if __name__ == "__main__":
    main()
