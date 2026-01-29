import re
import torch
import hashlib
import numpy as np
import pandas as pd
import typing as tp
from pathlib import Path
import os

from functools import partial
from dataclasses import fields, replace, _MISSING_TYPE
from typing import TypeAlias, Any
from collections.abc import Callable
from collections import defaultdict
from itertools import permutations
import matplotlib.colors as mcolors
import dataclasses

from notebooks_src.load_checkpoints import CkptInfo, CkptType, AttnParamType
from notebooks_src.constants import Paths

from notebooks_src.constants import DefaultValues
from einops import rearrange

import __main__


OLMO_WEIGHTS_MAPPING = {
    r".+self_attn\.q_proj\.head\d+\.weight": AttnParamType.Q_PROJ_HEAD.value,
    r".+self_attn\.k_proj\.head\d+\.weight": AttnParamType.K_PROJ_HEAD.value,
    r".+self_attn\.v_proj\.head\d+\.weight": AttnParamType.V_PROJ_HEAD.value,
    r".+self_attn\.o_proj\.head\d+\.weight": AttnParamType.O_PROJ_HEAD.value,
    r".+mlp\.up_proj\.weight": AttnParamType.FC1.value,
    r".+mlp\.down_proj\.weight": AttnParamType.FC2.value,
    r".+mlp\.gate_proj\.weight": AttnParamType.GATE.value,
}


# region hydra utils


def get_variable(name: str):
    return getattr(__main__, name)


def make_dict(keys: list[Any], values: list[Any]) -> dict:
    return dict(zip(keys, values))


def make_diff_pattern(diff_on: list[str], diff_patterns: list[tuple[Any, Any]]) -> str:
    return "-".join(sorted([f"{col}:({a},{b})" for col, (a, b) in zip(diff_on, diff_patterns)]))


def transpose_list_of_lists(list_of_lists: list[list[Any]]) -> list[list[Any]]:
    return list(zip(*list_of_lists))


def get_item(obj: Any, index: int) -> Any:
    return obj[index]


def dataclasses_replace_with_dict(obj, changes):
    return dataclasses.replace(obj, **changes)


def dataclasses_reset_fields(obj, fields_to_reset: list[str]):
    changes = dict()

    for field in fields(CkptInfo):
        if field.name in fields_to_reset:
            default_value = field.default
            if isinstance(default_value, _MISSING_TYPE):
                default_value = field.default_factory()
            changes[field.name] = default_value

    return dataclasses.replace(obj, **changes)


def remap_handler_helper(remap_func, ckpt_info, state_dict):
    return remap_func(state_dict, ckpt_info.n_head)


def remap_olmo_state_dict(state_dict, n_head: int):
    new_state_dict = dict()
    for key in state_dict.keys():
        if "self_attn.q_proj.weight" in key or "self_attn.k_proj.weight" in key or "self_attn.v_proj.weight" in key:
            heads = rearrange(state_dict[key], "(h d_head) d_model -> h d_head d_model", h=n_head)
            for h in range(n_head):
                new_state_dict[key.replace("weight", f"head{h}.weight")] = heads[h]
        elif "self_attn.o_proj.weight" in key:
            heads = rearrange(state_dict[key], "d_model (h d_head) -> h d_model d_head", h=n_head)
            for h in range(n_head):
                new_state_dict[key.replace("weight", f"head{h}.weight")] = heads[h]

    state_dict = {**new_state_dict, **state_dict}

    def sort_state_dict(state_dict):
        group_a, group_b = [], []

        for key in state_dict.keys():
            
            if "model.layers." not in key:
                group_a.append(key)
            else:
                layer_num = int(key.split("model.layers.")[-1].split(".")[0])
                group_b.append((layer_num, key))

        group_a_sorted = sorted(group_a)
        group_b_sorted = sorted(group_b, key=lambda x: x[0])
        group_b_keys = [key for _, key in group_b_sorted]

        sorted_keys = group_a_sorted + group_b_keys
        return {k: state_dict[k] for k in sorted_keys}

    state_dict = sort_state_dict(state_dict)
    
    return state_dict
