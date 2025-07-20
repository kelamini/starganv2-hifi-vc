import argparse

from omegaconf import OmegaConf


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-c", "--configs", type=str, default="configs/configs_asr.yml", help="")

    return parser.parse_args()


args = get_args()
cfgs = OmegaConf.load(args.configs)
config = OmegaConf.to_container(cfgs, resolve=True)
