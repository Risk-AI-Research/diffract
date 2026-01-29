import os

from typing import TypeVar, List
from omegaconf import OmegaConf, ListConfig, DictConfig
from pathlib import Path

from hydra import initialize, compose
from hydra.utils import instantiate

from notebooks_src.constants import MetaConstant, Paths

T = TypeVar("T")


def try_load_config(config_path: str, overrides: List[str] = []) -> tuple[bool, DictConfig | None]:
    try:
        configs_dir_str = os.path.relpath(Paths.CONFIG_DIR, Path(__file__).parent)
        with initialize(configs_dir_str, version_base="1.3"):
            config_path_str = os.path.relpath(Path(config_path), Paths.CONFIG_DIR)
            cfg = compose(config_path_str, overrides=overrides)

        OmegaConf.resolve(cfg)

        return True, cfg, None
    except Exception as e:
        return False, None, e


def resolve(config_path: str, key: str, default_value: T, cfg_overrides: List[str] = []) -> T:
    success, cfg, e = try_load_config(config_path, cfg_overrides)

    if not success:
        if default_value is None:
            raise e
        else:
            return default_value

    selected_value = OmegaConf.select(key=key, cfg=cfg, default=default_value)
    if isinstance(selected_value, ListConfig) or isinstance(selected_value, DictConfig):
        resolved_value = instantiate(selected_value)
    else:
        resolved_value = selected_value

    if type(resolved_value) in (ListConfig, DictConfig):
        resolved_value = OmegaConf.to_container(resolved_value)

    return resolved_value


class MetaConfiguredValuesProvider(MetaConstant, type):
    def __new__(mcs, name, bases, class_dict, config_path: str):
        cls = super().__new__(mcs, name, bases, class_dict)

        def __getattribute__(cls, attr):
            if attr.startswith("__"):
                return super().__getattribute__(attr)

            return resolve(config_path, attr, cls.__dict__.get(attr))

        cls.__class__.__getattribute__ = __getattribute__

        return cls


class ConfigResolveProvider:
    def __init__(self, config_path: str):
        self.config_path = config_path

    def resolve(self, attr, cfg_overrides: List[str] = []):
        return resolve(self.config_path, attr, self.__dict__.get(attr), cfg_overrides)
