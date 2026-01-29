from pathlib import Path
from notebooks_src.utils import (
    OLMO_WEIGHTS_MAPPING,
)
from notebooks_src.constants import Paths

import re
import json

from notebooks_src.configuring.core import ConfigResolveProvider

import sys
import rootutils

root = Path(rootutils.setup_root(".", indicator=".project-root"))
sys.path.insert(0, str(root / "src"))
sys.path.insert(0, str(root))

from diffract import Session, ParameterOverrides


def load_session(param_analysis, agg_analysis, config_path: Path = Paths.CONFIG_DIR / "load.yaml"):
    provider = ConfigResolveProvider(config_path)

    experiments_dirs = provider.resolve("experiments_dirs")
    ckpt_paths = [
        str(path.absolute())
        for exps_dir in experiments_dirs
        for path in (
            list(Path(exps_dir).rglob("*.ckpt"))
            + list(Path(exps_dir).rglob("*.safetensors"))
            + list(Path(exps_dir).rglob("*.pt"))
        )
    ]

    ckpt_collector = provider.resolve("ckpt_collector")(sorted(ckpt_paths))
    checkpoints_offloaded = ckpt_collector.collect(load=False)
    print("Filters bad calls:")
    print(json.dumps({str(k): v for k, v in ckpt_collector.filter_bad_calls_counter.items()}, indent=4))
    print("".join(["-"] * 100))
    print("Load paths:")
    print(json.dumps(ckpt_collector.get_filtered_paths(), indent=4))
    print("".join(["-"] * 100))

    session = Session(config_path=provider.resolve("diffract_config_path"))  
    
    with session:
        existing_models = session.list_models()

        # taichi produce problems during notebook launching
        session.configure_kernel("power_law_fit", fit_method="powerlaw")
        session.configure_kernel("truncated_power_law_fit", fit_method="powerlaw")
        session.configure_kernel("exponential_fit", fit_method="powerlaw")
        
    def _build_overrides(key: str, pattern: str):
        int_mapping = {
            "q_proj_head": 1,
            "k_proj_head": 2,
            "v_proj_head": 3,
            "o_proj_head": 4,
            "fc1": 5,
            "gate": 6,
            "fc2": 7,
        }
        
        other_meta = {
            'param': f"#{int_mapping[OLMO_WEIGHTS_MAPPING[pattern]]}-{OLMO_WEIGHTS_MAPPING[pattern]}",
            'layer_id': f'{int(key.split(".")[5]):02d}',
        }
        
        if ".head" in key:
            other_meta["head_id"] = f'{int(key.split(".")[-2].replace("head", "")):02d}'
            name = f"{other_meta['param']}#layer={other_meta['layer_id']}#head={other_meta['head_id']}"
        else:
            name = f"{other_meta['param']}#layer={other_meta['layer_id']}"
            
        overrides = ParameterOverrides(
            name = name,
            ptype = "attn",
            other_meta = other_meta,
        )
            
        return overrides
    
    for path in sorted(ckpt_collector.get_filtered_paths()):
        if "model=olmo-step10000-unsharded-hf" in path:
            model_id = "olmo_step10k"
        elif "model=olmo-step1140000-unsharded-hf" in path:
            model_id = "olmo_step100k"
        else:
            raise ValueError

        if model_id in existing_models:
            continue
            
        collected = provider.resolve("ckpt_collector")((path,)).collect(load=True)
        ckpt_state_dict = next(iter(collected.values()))
        
        parameter_overrides = {
            key: _build_overrides(key, pattern)
            for key in ckpt_state_dict
            for pattern in OLMO_WEIGHTS_MAPPING.keys()
            if re.match(pattern, key)
        }

        try:
            session.add(
                ckpt_state_dict,
                model_id=model_id,
                parameter_overrides=parameter_overrides,
            )
        except Exception as e:
            print(e)
            
    session.compute(*param_analysis, parameter_types=["attn"])

    agg = provider.resolve("add_model_agg")
    with session:
        for model_ids in agg:
            path1 = model_ids[0]
            path2 = model_ids[1]
            session.compute(*agg_analysis, model_ids=[path1, path2], parameter_types=["attn"])

    return session
