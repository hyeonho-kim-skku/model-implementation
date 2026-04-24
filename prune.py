import argparse
import os

import torch
import yaml

from pruning.structured import prune_checkpoint


device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to the yaml config file")
    parser.add_argument("--checkpoint-path", dest="checkpoint_path", type=str, help="Path to a LoRA checkpoint")
    parser.add_argument("--output-dir", dest="output_dir", type=str, default="./pruned", help="Directory to save pruned artifacts")
    parser.add_argument("--pruning-ratio", dest="pruning_ratio", type=float, default=0.2, help="Structured pruning ratio")
    parser.add_argument("--pruning-modules", dest="pruning_modules", type=str, default="proj,mlp", help="Comma-separated pruning targets: qkv,proj,mlp")
    parser.add_argument("--iterative-steps", dest="iterative_steps", type=int, default=1, help="Number of iterative pruning steps")
    parser.add_argument("--global-pruning", dest="global_pruning", action=argparse.BooleanOptionalAction, default=False, help="Use global pruning across target modules")
    parser.add_argument("--round-to", dest="round_to", type=int, default=None, help="Round pruned dimensions to a multiple")
    return parser.parse_args()


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    prune_checkpoint(
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir,
        pruning_ratio=args.pruning_ratio,
        pruning_modules=args.pruning_modules,
        iterative_steps=args.iterative_steps,
        global_pruning=args.global_pruning,
        round_to=args.round_to,
        device=device,
    )


if __name__ == "__main__":
    args = parse_args()
    if args.config:
        with open(args.config, "r") as file:
            config_dict = yaml.safe_load(file)
        parser_args = vars(args)
        parser_args.update({key: value for key, value in config_dict.items() if value is not None})
        args = argparse.Namespace(**parser_args)

    main(args)
