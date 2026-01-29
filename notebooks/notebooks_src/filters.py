from typing import Any
from notebooks_src.load_checkpoints import CkptInfo


def simple_field_filter(ckpt_info: CkptInfo, field: str, values: list[Any]) -> bool:
    return getattr(ckpt_info, field) in values


def implication_filter(
    ckpt_info: CkptInfo, impl_in_field: str, impl_in_values: list[Any], impl_out_field: str, impl_out_values: list[Any]
) -> bool:
    if getattr(ckpt_info, impl_in_field) in impl_in_values:
        return getattr(ckpt_info, impl_out_field) in impl_out_values
    else:
        return True
