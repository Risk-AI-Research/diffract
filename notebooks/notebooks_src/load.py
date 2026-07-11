import json
import re
import sys
from pathlib import Path

import rootutils

root = Path(rootutils.setup_root(".", indicator=".project-root"))
sys.path.insert(0, str(root / "src"))
sys.path.insert(0, str(root))

from diffract import ParameterOverrides, ParameterType, Session

from notebooks_src.configuring.core import ConfigResolveProvider
from notebooks_src.constants import Paths
from notebooks_src.utils import OLMO_WEIGHTS_MAPPING

# Numeric prefixes fix the display order of parameter groups in the plots.
PARAM_ORDER = {
    "q_proj_head": 1,
    "k_proj_head": 2,
    "v_proj_head": 3,
    "o_proj_head": 4,
    "fc1": 5,
    "gate": 6,
    "fc2": 7,
}

# The plot configs reference these model ids explicitly.
MODEL_IDS = {
    "model=olmo-step10000-unsharded-hf": "olmo_step10k",
    "model=olmo-step1140000-unsharded-hf": "olmo_step100k",
}

LAYER_PATTERN = re.compile(r"\.layers\.(\d+)\.")


def _build_overrides(key: str, pattern: str) -> ParameterOverrides:
    param = OLMO_WEIGHTS_MAPPING[pattern]

    layer_match = LAYER_PATTERN.search(key)
    if layer_match is None:
        msg = f"Cannot extract layer index from parameter key {key!r}"
        raise ValueError(msg)

    other_meta = {
        "param": f"#{PARAM_ORDER[param]}-{param}",
        "layer_id": f"{int(layer_match.group(1)):02d}",
    }

    if ".head" in key:
        other_meta["head_id"] = f"{int(key.split('.')[-2].replace('head', '')):02d}"
        name = f"{other_meta['param']}#layer={other_meta['layer_id']}#head={other_meta['head_id']}"
    else:
        name = f"{other_meta['param']}#layer={other_meta['layer_id']}"

    return ParameterOverrides(
        name=name,
        ptype="attn",
        other_meta=other_meta,
    )


def _model_id_for(path: str, model_mapping: dict[str, str]) -> str:
    if path in model_mapping:
        return model_mapping[path]
    for folder, model_id in MODEL_IDS.items():
        if folder in path:
            return model_id
    msg = (
        f"Cannot derive a model id for checkpoint {path!r}; add it to "
        "MODEL_IDS or to model_mapping in configs/load.yaml"
    )
    raise ValueError(msg)


def load_session(
    param_analysis: list[str],
    agg_analysis: list[str],
    config_path: Path = Paths.CONFIG_DIR / "load.yaml",
) -> Session:
    """Load the checkpoints into a session and compute the analyses.

    Re-entrant: models that are already stored are not re-added, and
    kernels skip fields that are already computed. Heavy-tailed fits use
    fit_method="auto" (their default): the accelerated taichi
    implementation when available, the powerlaw library otherwise.
    """
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
        # HF shards are merged into model.safetensors by the notebook;
        # only whole-model files are loadable checkpoints.
        if not re.search(r"-\d{5}-of-\d{5}", path.name)
    ]

    ckpt_collector = provider.resolve("ckpt_collector")(sorted(ckpt_paths))
    ckpt_collector.collect(load=False)
    print("Filters bad calls:")
    print(json.dumps({str(k): v for k, v in ckpt_collector.filter_bad_calls_counter.items()}, indent=4))
    print("".join(["-"] * 100))
    print("Load paths:")
    print(json.dumps(ckpt_collector.get_filtered_paths(), indent=4))
    print("".join(["-"] * 100))

    session = Session(config_path=provider.resolve("diffract_config_path"))
    existing_models = session.models.list()
    model_mapping = provider.resolve("model_mapping")

    for path in sorted(ckpt_collector.get_filtered_paths()):
        model_id = _model_id_for(path, model_mapping)
        if model_id in existing_models:
            print(f"Model {model_id!r} is already stored, skipping {path}")
            continue

        collected = provider.resolve("ckpt_collector")((path,)).collect(load=True)
        ckpt_state_dict = next(iter(collected.values()))

        parameter_overrides = {
            key: _build_overrides(key, pattern)
            for key in ckpt_state_dict
            for pattern in OLMO_WEIGHTS_MAPPING
            if re.match(pattern, key)
        }

        session.models.add(
            ckpt_state_dict,
            model_id=model_id,
            parameter_overrides=parameter_overrides,
        )
        # A full checkpoint is several GB; free it before loading the next.
        del collected, ckpt_state_dict, parameter_overrides

    attn = ParameterType.from_string("attn")
    session.filter(param_types=[attn]).compute.apply(*param_analysis)

    for model_pair in provider.resolve("add_model_agg"):
        scoped = session.filter(model_ids=list(model_pair), param_types=[attn])
        scoped.compute.apply(*agg_analysis)

    return session
