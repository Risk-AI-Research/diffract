import re
from tqdm.auto import tqdm
from enum import StrEnum
from pathlib import Path
from collections import Counter
from dataclasses import dataclass, field, fields
from typing import cast
from collections.abc import Callable
from omegaconf import OmegaConf

import torch

from safetensors.torch import load_file

from notebooks_src.constants import DefaultValues


class CkptType(StrEnum):
    WEIGHTS = "weights"
    UPDATES = "updates"
    DIFF = "diff"
    F_DIFF = "f_diff"

    @classmethod
    def from_str(cls, name: str | None) -> "CkptType":
        if name is None:
            return cls.OTHER
        try:
            return cls[name.upper()]
        except KeyError:
            return cls.OTHER

    def __repr__(self):
        return f"'{self.value}'"


class AttnParamType(StrEnum):
    Q_PROJ_HEAD = "q_proj_head"
    K_PROJ_HEAD = "k_proj_head"
    V_PROJ_HEAD = "v_proj_head"
    O_PROJ_HEAD = "o_proj_head"
    FC1 = "fc1"
    FC2 = "fc2"
    GATE = "gate"

    OTHER = "other"

    @classmethod
    def from_str(cls, name: str | None) -> "AttnParamType":
        if name is None:
            return cls.OTHER
        try:
            return cls[name.upper()]
        except KeyError:
            return cls.OTHER

    def __repr__(self):
        return f"'{self.value}'"


class ArchType(StrEnum):
    OLMO = "olmo"

    OTHER = "other"

    @classmethod
    def from_str(cls, name: str | None) -> "ArchType":
        if name is None:
            return cls.OTHER
        try:
            return cls[name.upper()]
        except KeyError:
            return cls.OTHER

    def __repr__(self):
        return f"'{self.value}'"


@dataclass(frozen=True)
class CkptInfo:
    # meta
    path: str = field(default_factory=lambda: DefaultValues.MISSING_STR_PLACEHOLDER)

    # from path
    arch: ArchType = field(default=ArchType.OTHER)

    # from config.json
    d_model: int = field(default_factory=lambda: DefaultValues.MISSING_INT_PLACEHOLDER)
    n_head: int = field(default_factory=lambda: DefaultValues.MISSING_INT_PLACEHOLDER)
    n_layer: int = field(default_factory=lambda: DefaultValues.MISSING_INT_PLACEHOLDER)

    def __le__(self, other):
        return hash(self) <= hash(other)

    def __lt__(self, other):
        return hash(self) < hash(other)

    def __ge__(self, other):
        return hash(self) >= hash(other)

    def __gt__(self, other):
        return hash(self) > hash(other)

    def __eq__(self, other):
        return hash(self) == hash(other)


class CkptLoader:
    path: str
    _ckpt_info: CkptInfo | None = None

    def __init__(self, path: str):
        self.path = path

    def load_state_dict(self) -> dict[str, torch.tensor]:
        suffix = Path(self.path).suffix
        if suffix in (".ckpt", ".pt"):
            ckpt = torch.load(Path(self.path).absolute(), map_location="cpu")
            if "state_dict" in ckpt.keys():
                state_dict = ckpt["state_dict"]
            else:
                state_dict = ckpt
        elif suffix in ".safetensors":
            state_dict = load_file(self.path)
        else:
            raise ValueError(f"Cannot load checkpoint with path: {self.path}")

        return state_dict

    @property
    def ckpt_info(self) -> CkptInfo:
        if self._ckpt_info is None:
            ckpt_info_kwargs = dict()
            for field in fields(CkptInfo):
                if (resolver := getattr(self, f"_resolve_{field.name}", None)) is not None:
                    ckpt_info_kwargs[field.name] = resolver()
                elif (field_value := self._default_resolve(field.name)) is not None:
                    ckpt_info_kwargs[field.name] = cast(str, field.type)(field_value)

            self._ckpt_info = CkptInfo(**ckpt_info_kwargs)

        return self._ckpt_info

    def _default_resolve(self, field: str) -> str | None:
        val = None

        try:
            ckpt_cfg_path = Path(self.path).parent / "config.json"
            ckpt_cfg = OmegaConf.load(ckpt_cfg_path)
            val = OmegaConf.select(ckpt_cfg, field, default=None)
        except:
            ...

        if val is None:
            try:
                run_cfg_path = Path(self.path).parent / ".." / "config.yaml"
                run_cfg = OmegaConf.load(run_cfg_path)
                val = OmegaConf.select(run_cfg, field, default=None)
            except:
                ...

        return val

    def _resolve_path(self) -> str:
        return self.path

    def _resolve_arch(self) -> ArchType:
        return ArchType.OLMO
    
    def _resolve_d_model(self) -> int:
        return 2048

    def _resolve_n_head(self) -> int:
        return 16

    def _resolve_n_layer(self) -> int:
        return 16
    
    def _resolve_ckpt_type(self) -> CkptType:
        return CkptType.WEIGHTS
    
    def _resolve_diff_pattern(self) -> str:
        return DefaultValues.MISSING_STR_PLACEHOLDER


class CkptFilter:
    filters: Callable[[CkptInfo], bool]

    def __init__(self, filter_funcs=None):
        self.filters = list()

        if filter_funcs is not None:
            for filter_func in filter_funcs:
                self.add_filter(filter_func)

    def add_filter(self, filter_func: Callable[[CkptInfo], bool]) -> None:
        self.filters.append(filter_func)

    def __call__(self, ckpt_info: CkptInfo, log_bad_calls: bool = False) -> bool | tuple[bool, tuple[str]]:
        if log_bad_calls:
            bad_calls = list()

        accepted = True
        for filter_func in self.filters:
            filter_flag = filter_func(ckpt_info)
            if not filter_flag and log_bad_calls:
                bad_calls.append(filter_func)
            accepted &= filter_flag

        if log_bad_calls:
            return accepted, tuple(bad_calls)
        else:
            return accepted


class CkptProcessor:
    _filters: list[CkptFilter]
    _handlers: list[list[Callable[[CkptInfo, dict[str, torch.tensor]], dict[str, torch.tensor]]]]

    def __init__(self, handlers=None, ckpt_filters=None):
        self._filters = [None]
        self._handlers = [[]]

        if handlers is not None and ckpt_filters is not None:
            for handler, ckpt_filter in zip(handlers, ckpt_filters):
                self.add_handler(handler, ckpt_filter)

    def add_handler(
        self,
        handler: Callable[[CkptInfo, dict[str, torch.tensor]], dict[str, torch.tensor]],
        ckpt_filter: CkptFilter | None = None,
    ) -> None:
        if ckpt_filter is None:
            self._handlers[0].append(handler)
        elif ckpt_filter in self._filters:
            self._handlers[self._filters.index(ckpt_filter)].append(handler)
        else:
            self._filters.append(ckpt_filter)
            self._handlers.append([handler])

    def __call__(self, ckpt_info: CkptInfo, ckpt_state_dict: dict[str, torch.tensor]) -> dict[str, torch.tensor]:
        for ckpt_filter, handlers in zip(self._filters, self._handlers):
            if ckpt_filter is None or ckpt_filter(ckpt_info):
                for handler in handlers:
                    ckpt_state_dict = handler(ckpt_info, ckpt_state_dict)

        return ckpt_state_dict


class CkptCollector:
    paths: list[str]
    ckpt_filter: CkptFilter | None
    ckpt_processor: CkptProcessor | None
    filter_bad_calls_counter: Counter = None

    _loaded: bool = False
    _filtered: list[CkptInfo] = None
    _checkpoints: dict[CkptInfo, dict[str, torch.tensor]] = None

    def __init__(
        self, paths: list[str], ckpt_filter: CkptFilter | None = None, ckpt_processor: CkptProcessor | None = None
    ):
        self.paths = paths
        self.ckpt_filter = ckpt_filter
        self.ckpt_processor = ckpt_processor

    def collect(self, load: bool = True) -> dict[CkptInfo, dict[str, torch.tensor]]:
        if self._checkpoints is None or not self._loaded:
            self._checkpoints = dict()

            if self._filtered is None:
                self._filtered = list()
                all_filter_bad_calls = list()
                for path in tqdm(self.paths, total=len(self.paths), desc="Filtering..."):
                    loader = CkptLoader(path)
                    ckpt_info = loader.ckpt_info

                    if self.ckpt_filter is not None:
                        accepted, filter_bad_calls = self.ckpt_filter(ckpt_info, log_bad_calls=True)

                    if self.ckpt_filter is None or accepted:
                        self._filtered.append(loader)
                    else:
                        all_filter_bad_calls.extend(filter_bad_calls)
            
            for loader in tqdm(
                self._filtered, total=len(self._filtered), desc=("Loading..." if load else "Skip loading...")
            ):
                ckpt_info = loader.ckpt_info
                if load:
                    ckpt_state_dict = loader.load_state_dict()
                    if self.ckpt_processor is not None:
                        ckpt_state_dict = self.ckpt_processor(ckpt_info, ckpt_state_dict)
                else:
                    ckpt_state_dict = {"dummy.weight": torch.empty((2, 2))}
                self._checkpoints[ckpt_info] = ckpt_state_dict

            if self.filter_bad_calls_counter is None:
                self.filter_bad_calls_counter = Counter(all_filter_bad_calls)
            self._loaded |= load

        return self._checkpoints

    def get_filtered_paths(self) -> list[str]:
        return [ckpt_info.path for ckpt_info in self._checkpoints]

    def clear(self) -> None:
        del self._checkpoints
        self._checkpoints = None
