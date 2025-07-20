import argparse

from omegaconf import OmegaConf


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-c", "--configs", type=str, default="configs/configs_f0.yml", help="")

    return parser.parse_args()


args = get_args()
cfgs = OmegaConf.load(args.configs)
config = OmegaConf.to_container(cfgs, resolve=True)
