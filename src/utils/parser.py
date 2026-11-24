import yaml
import argparse


def get_config(path: str) -> argparse.Namespace:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return argparse.Namespace(**cfg)
