# src/utils/config.py
import yaml
from types import SimpleNamespace

def load_config(cfg_path):
    with open(cfg_path, "r") as f:
        cfg_dict = yaml.safe_load(f)

    def dict_to_namespace(d):
        """递归将字典转换为 SimpleNamespace"""
        for k, v in d.items():
            if isinstance(v, dict):
                d[k] = dict_to_namespace(v)
        return SimpleNamespace(**d)

    return dict_to_namespace(cfg_dict)