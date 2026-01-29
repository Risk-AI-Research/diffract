import numpy as np

from diffract.core.compute.config import KernelConfig
from diffract.core.compute.field_signature import (
    collect_field_catalog,
    collect_field_signatures,
)
from diffract.core.compute.execution.enums import (
    KernelApplyLevel,
    KernelExecutionProtocol,
)
from diffract.core.compute.metadata import KernelInfo
from diffract.core.compute.registry import KernelRegistry
from diffract.core.storage.ram_manager import RAMStorageManager


def test_collect_field_signatures_reads_metadata() -> None:
    storage = RAMStorageManager()
    storage.set_field("uid1", "scalar_field", 1.5)
    storage.set_field("uid2", "vector_field", np.array([1, 2, 3], dtype=np.float32))

    sigs = collect_field_signatures(storage)

    assert "scalar_field" in sigs
    assert sigs["scalar_field"].kind == "scalar"
    assert "vector_field" in sigs
    assert sigs["vector_field"].kind in ("vector", "matrix", "ndarray")


def test_collect_field_catalog_adds_registry_only_fields() -> None:
    storage = RAMStorageManager()
    storage.set_field("uid1", "scalar_field", 2.0)

    registry = KernelRegistry()
    registry.register_kernel(
        name="new_kernel",
        require_fields=(),
        produce_fields=("missing_field",),
        implementation=lambda: 0,
        apply_level=KernelApplyLevel.PARAMETER,
        execution_protocol=KernelExecutionProtocol.SEQUENTIAL,
        restrictions=None,
        config=KernelConfig(),
        info=KernelInfo(summary="dummy"),
    )

    catalog = collect_field_catalog(storage, registry)

    assert "scalar_field" in catalog
    assert catalog["scalar_field"].available is True
    assert "missing_field" in catalog
    assert catalog["missing_field"].available is False
    assert catalog["missing_field"].apply_level == KernelApplyLevel.PARAMETER
