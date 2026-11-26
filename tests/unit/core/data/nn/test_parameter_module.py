from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from diffract.core.data.identity import DuplicateParameterError
from diffract.core.data.metadata.errors import UniqueConstraintError
from diffract.core.data.metadata.sqlite_index import SQLiteMetadataIndex
from diffract.core.data.nn.aggregates.metadata import AggregateMetadata
from diffract.core.data.nn.aggregates.proxy import AggregateProxy
from diffract.core.data.nn.aggregates.repository import AggregateRepository
from diffract.core.data.nn.params.metadata import ParameterMetadata
from diffract.core.data.nn.params.proxy import ParameterDataProxy
from diffract.core.data.nn.params.repository import ParameterRepository
from diffract.core.data.nn.params.schema import ParameterType
from diffract.core.storage.catalog_manager import CatalogStorageManager
from diffract.core.storage.ram_manager import RAMStorageManager

if TYPE_CHECKING:
    from diffract.core.data.nn.params.interface import IParameterView


pytestmark = pytest.mark.unit


class FakeStorage:
    """Fake storage for testing - stores field values only, no metadata."""

    def __init__(self) -> None:
        # field_name -> uid -> value
        self._store: dict[str, dict[str, Any]] = {}

    def __enter__(self) -> "FakeStorage":
        return self

    def __exit__(self, _exc_type: object, _exc_val: object, _exc_tb: object) -> None:
        return None

    def set_field(
        self, obj_uid: str, field_name: str, value: Any, *, table: str = "default"
    ) -> None:
        self._store.setdefault(field_name, {})[obj_uid] = value

    def get_field(
        self, obj_uid: str, field_name: str, *, table: str = "default"
    ) -> Any:
        try:
            return self._store[field_name][obj_uid]
        except KeyError as err:
            raise KeyError(obj_uid, field_name) from err

    def has_field(
        self, obj_uid: str, field_name: str, *, table: str = "default"
    ) -> bool:
        return obj_uid in self._store.get(field_name, {})

    def erase_field(
        self, obj_uid: str, field_name: str, *, table: str = "default"
    ) -> None:
        self._store.get(field_name, {}).pop(obj_uid, None)

    def erase_obj(self, obj_uid: str, *, table: str = "default") -> None:
        for field_data in self._store.values():
            field_data.pop(obj_uid, None)

    def erase_field_for_all(self, field_name: str, *, table: str = "default") -> None:
        self._store.pop(field_name, None)

    def list_fields(
        self, obj_uid: str | None = None, *, table: str = "default"
    ) -> list[str]:
        res: list[str] = []
        for f, data in self._store.items():
            if obj_uid is None or obj_uid in data:
                res.append(f)
        return res

    def list_objs(self, *, table: str = "default") -> list[str]:
        uids: set[str] = set()
        for data in self._store.values():
            uids.update(data.keys())
        return sorted(uids)

    def list_objs_has_field(
        self, field_name: str, *, table: str = "default"
    ) -> list[str]:
        return sorted(self._store.get(field_name, {}).keys())

    def get_field_metadata(
        self, obj_uid: str, field_name: str, *, table: str = "default"
    ) -> dict[str, Any] | None:
        return None


class FakeCache:
    def __init__(self) -> None:
        self._cache: dict[tuple[str, str], Any] = {}

    def __enter__(self) -> "FakeCache":
        return self

    def __exit__(self, _exc_type: object, _exc_val: object, _exc_tb: object) -> None:
        return None

    def set_field(self, obj_uid: str, field_name: str, value: Any) -> None:
        self._cache[(obj_uid, field_name)] = value

    def get_field(self, obj_uid: str, field_name: str) -> Any:
        return self._cache[(obj_uid, field_name)]

    def has_field(self, obj_uid: str, field_name: str) -> bool:
        return (obj_uid, field_name) in self._cache

    def erase_field(self, obj_uid: str, field_name: str) -> None:
        self._cache.pop((obj_uid, field_name), None)

    def erase_field_for_all(self, field_name: str) -> None:
        keys_to_remove = [k for k in self._cache if k[1] == field_name]
        for k in keys_to_remove:
            del self._cache[k]


@pytest.fixture
def managers():
    index = SQLiteMetadataIndex(":memory:")
    try:
        ParameterRepository.define_schema(index)
        yield FakeStorage(), FakeCache(), index
    finally:
        index.close()


def make_proxy(
    name: str,
    ptype: ParameterType,
    model_id: str,
    storage: FakeStorage,
    cache: FakeCache,
    metadata_index: SQLiteMetadataIndex,
) -> ParameterDataProxy:
    """Create a proxy and store its metadata in the index."""
    meta = ParameterMetadata(name=name, ptype=ptype, model_id=model_id)
    repository = ParameterRepository(storage, metadata_index, cache)
    return ParameterDataProxy.create_and_store(meta=meta, repository=repository)


def make_view(c: ParameterRepository) -> "IParameterView":
    return c.create_view()


# ---------------------- Tests: ParameterType ----------------------


def test_parameter_type_dynamic_from_string():
    # Existing member
    assert ParameterType.from_string("dense") == ParameterType.DENSE
    # New dynamic member
    custom = ParameterType.from_string("conv")
    assert isinstance(custom, ParameterType)
    assert (custom & ParameterType.DENSE) == 0


# ---------------------- Tests: ParameterMetadata ----------------------


def test_parameter_metadata_uid_and_fields():
    m = ParameterMetadata(name="w", ptype=ParameterType.DENSE, model_id="m1")
    assert isinstance(m.uid, str)
    assert m.name == "w"
    assert m.ptype == ParameterType.DENSE
    assert m.model_id == "m1"
    assert isinstance(m.other_meta, dict)


def test_parameter_metadata_to_dict():
    m = ParameterMetadata(name="w", ptype=ParameterType.DENSE, model_id="m1")
    d = m.to_dict()
    assert d["name"] == "w"
    assert d["model_id"] == "m1"
    assert d["ptype"] == "DENSE"
    assert "uid" in d


# ---------------------- Tests: ParameterDataProxy ----------------------


def test_proxy_create_and_store(managers):
    storage, cache, metadata_index = managers
    repository = ParameterRepository(storage, metadata_index, cache)
    ParameterRepository.define_schema(metadata_index)

    meta = ParameterMetadata(name="w", ptype=ParameterType.DENSE, model_id="m1")
    ParameterDataProxy.create_and_store(
        meta=meta,
        repository=repository,
    )

    # Verify metadata was stored in index
    stored = metadata_index.get("parameters", meta.uid)
    assert stored is not None
    assert stored["name"] == "w"
    assert repository.instances.get(stored["instance_uid"]).name == "m1"
    assert stored["ptype"] == "DENSE"

    proxy2 = repository.get_proxy(meta.uid)
    uncached = ParameterDataProxy(meta=proxy2.meta, _repository=repository)
    assert hash(proxy2) == hash(meta.uid)
    assert {proxy2: "stored"}[uncached] == "stored"
    assert proxy2.meta.uid == meta.uid
    assert proxy2.meta.name == "w"


def test_aggregate_proxy_hash_uses_persistent_uid(managers):
    storage, cache, index = managers
    aggregates = AggregateRepository(storage, index, cache)
    meta = AggregateMetadata(field_name="score")
    proxy = AggregateProxy(meta=meta, _repository=aggregates)
    assert hash(proxy) == hash(meta.uid)


def test_direct_kernel_reads_require_each_layer_to_opt_in(managers):
    class WrappedCatalog(CatalogStorageManager):
        pass

    class WrappedRepository(ParameterRepository):
        pass

    class DirectRepository(WrappedRepository):
        supports_direct_kernel_reads = True

    _, cache, index = managers
    catalog = CatalogStorageManager(RAMStorageManager(), index)
    repository = ParameterRepository(catalog, index)
    meta = ParameterMetadata(name="w", ptype=ParameterType.DENSE, model_id="model")
    proxy = ParameterDataProxy.create_and_store(meta, repository)
    proxy.set_field("weights", np.array([2.0]))
    key = (proxy.meta.uid, "weights")

    view = DirectRepository(catalog, index).create_view()
    pointers = view.resolve_kernel_inputs(("weights",))
    assert pointers is not None
    assert view.kernel_inputs_are_resident()
    np.testing.assert_array_equal(catalog.read_pointer(pointers[key]), [2.0])

    for wrapped in (
        WrappedRepository(catalog, index),
        ParameterRepository(WrappedCatalog(catalog.backend, index), index),
        ParameterRepository(catalog, index, cache),
    ):
        view = wrapped.create_view()
        assert view.resolve_kernel_inputs(("weights",)) is None
        assert not view.kernel_inputs_are_resident()
        np.testing.assert_array_equal(view[proxy.meta.uid].get_field("weights"), [2.0])


def test_wrapped_ram_copies_kernel_inputs_until_it_opts_in(managers):
    class WrappedRAM(RAMStorageManager):
        pass

    class BorrowingRAM(WrappedRAM):
        borrows_kernel_inputs = True

    _, _, index = managers
    for backend, resident in ((WrappedRAM(), False), (BorrowingRAM(), True)):
        catalog = CatalogStorageManager(backend, index)
        repository = ParameterRepository(catalog, index)
        meta = ParameterMetadata(
            name="w", ptype=ParameterType.DENSE, model_id=backend.__class__.__name__
        )
        proxy = ParameterDataProxy.create_and_store(meta, repository)
        proxy.set_field("weights", np.array([2.0]))
        view = repository.create_view()
        assert view.kernel_inputs_are_resident() is resident
        sample = proxy.get_field("weights", kernel_input=True)
        sample[0] = 77.0
        np.testing.assert_array_equal(
            proxy.get_field("weights"), [77.0 if resident else 2.0]
        )


def test_proxy_fields_and_cache_prefetch(managers):
    storage, cache, metadata_index = managers
    ParameterRepository.define_schema(metadata_index)
    p = make_proxy("w", ParameterType.DENSE, "m1", storage, cache, metadata_index)

    # Initially field absent
    assert p.has_field("weights") is False

    # Set in storage and get with prefetch
    storage.set_field(p.meta.uid, "weights", np.array([1, 2, 3]))
    val = p.get_field("weights", auto_prefetch=True)
    assert isinstance(val, np.ndarray)
    assert cache.has_field(p.meta.uid, "weights") is True

    # Erase via proxy
    p.erase_field("weights")
    assert storage.has_field(p.meta.uid, "weights") is False


# ---------------------- Tests: ParameterRepository ----------------------


def test_sequence_protocol_and_order(managers):
    storage, cache, metadata_index = managers
    ParameterRepository.define_schema(metadata_index)
    c = ParameterRepository(storage, metadata_index, cache)

    make_proxy("z", ParameterType.DENSE, "m1", storage, cache, metadata_index)
    make_proxy("a", ParameterType.DENSE, "m2", storage, cache, metadata_index)
    make_proxy("a", ParameterType.UNKNOWN, "m0", storage, cache, metadata_index)

    assert len(c) == 3
    names = [p.meta.name for p in c]
    assert sorted(names) == ["a", "a", "z"]


def test_duplicate_handling(managers):
    storage, cache, metadata_index = managers
    ParameterRepository.define_schema(metadata_index)
    c = ParameterRepository(storage, metadata_index, cache)

    make_proxy("a", ParameterType.DENSE, "m0", storage, cache, metadata_index)
    make_proxy("a", ParameterType.UNKNOWN, "m1", storage, cache, metadata_index)

    by_names = [p.meta.model_id for p in c if p.meta.name == "a"]
    assert set(by_names) == {"m0", "m1"}


def test_filter_by_name(managers):
    storage, cache, metadata_index = managers
    ParameterRepository.define_schema(metadata_index)
    c = ParameterRepository(storage, metadata_index, cache)

    make_proxy("a", ParameterType.DENSE, "m0", storage, cache, metadata_index)
    make_proxy("b", ParameterType.UNKNOWN, "m1", storage, cache, metadata_index)
    make_proxy("c", ParameterType.DENSE, "m1", storage, cache, metadata_index)

    cf = make_view(c).filter_by_name("a", "c")
    assert sorted([p.meta.name for p in cf]) == ["a", "c"]


def test_filter_by_name_regexp_prefix(managers):
    storage, cache, metadata_index = managers
    ParameterRepository.define_schema(metadata_index)
    c = ParameterRepository(storage, metadata_index, cache)

    make_proxy("some_model", ParameterType.DENSE, "m0", storage, cache, metadata_index)
    make_proxy(
        "some_model_2", ParameterType.DENSE, "m0", storage, cache, metadata_index
    )
    make_proxy("other", ParameterType.DENSE, "m0", storage, cache, metadata_index)

    # exact by default
    exact = make_view(c).filter_by_name("some_model")
    assert [p.meta.name for p in exact] == ["some_model"]

    # regex via "re:" prefix (fullmatch semantics)
    re_matched = make_view(c).filter_by_name("re:some_model.*")
    assert sorted([p.meta.name for p in re_matched]) == ["some_model", "some_model_2"]


def test_filter_by_ptype(managers):
    storage, cache, metadata_index = managers
    ParameterRepository.define_schema(metadata_index)
    c = ParameterRepository(storage, metadata_index, cache)

    make_proxy("a", ParameterType.DENSE, "m0", storage, cache, metadata_index)
    make_proxy("b", ParameterType.UNKNOWN, "m1", storage, cache, metadata_index)
    make_proxy("c", ParameterType.DENSE, "m1", storage, cache, metadata_index)

    cd = make_view(c).filter_by_ptype("DENSE")
    assert sorted([p.meta.name for p in cd]) == ["a", "c"]


def test_filter_by_model_id(managers):
    storage, cache, metadata_index = managers
    ParameterRepository.define_schema(metadata_index)
    c = ParameterRepository(storage, metadata_index, cache)

    make_proxy("a", ParameterType.DENSE, "m0", storage, cache, metadata_index)
    make_proxy("b", ParameterType.UNKNOWN, "m1", storage, cache, metadata_index)
    make_proxy("c", ParameterType.DENSE, "m1", storage, cache, metadata_index)

    cm = make_view(c).filter_by_model_id("m1")
    assert sorted([p.meta.name for p in cm]) == ["b", "c"]


def test_filter_by_model_id_regexp_prefix(managers):
    storage, cache, metadata_index = managers
    ParameterRepository.define_schema(metadata_index)
    c = ParameterRepository(storage, metadata_index, cache)

    make_proxy("a", ParameterType.DENSE, "some_model", storage, cache, metadata_index)
    make_proxy("b", ParameterType.DENSE, "some_model_2", storage, cache, metadata_index)
    make_proxy("c", ParameterType.DENSE, "other", storage, cache, metadata_index)

    # exact by default
    exact = make_view(c).filter_by_model_id("some_model")
    assert [p.meta.model_id for p in exact] == ["some_model"]

    # regex via "re:" prefix (fullmatch semantics)
    re_matched = make_view(c).filter_by_model_id("re:some_model.*")
    assert sorted([p.meta.model_id for p in re_matched]) == [
        "some_model",
        "some_model_2",
    ]


def test_filter_by_field(managers):
    storage, cache, metadata_index = managers
    ParameterRepository.define_schema(metadata_index)
    c = ParameterRepository(storage, metadata_index, cache)

    p1 = make_proxy("a", ParameterType.DENSE, "m0", storage, cache, metadata_index)
    make_proxy("b", ParameterType.DENSE, "m0", storage, cache, metadata_index)

    storage.set_field(p1.meta.uid, "weights", [1, 2, 3])

    cw = make_view(c).filter_by_fields("weights")
    assert sorted([p.meta.name for p in cw]) == ["a"]


def test_prefetch(managers):
    storage, cache, metadata_index = managers
    ParameterRepository.define_schema(metadata_index)
    c = ParameterRepository(storage, metadata_index, cache)

    p1 = make_proxy("a", ParameterType.DENSE, "m0", storage, cache, metadata_index)
    p2 = make_proxy("b", ParameterType.DENSE, "m0", storage, cache, metadata_index)

    storage.set_field(p1.meta.uid, "weights", [1, 2, 3])

    # Filter to only p1 (which has weights)
    view = make_view(c).filter_by_fields("weights")
    ok = view.prefetch_fields(fields=["weights"], verify_prefetch=True)
    assert ok is True
    assert cache.has_field(p1.meta.uid, "weights") is True
    assert cache.has_field(p2.meta.uid, "weights") is False

    # Now try with full view (p2 doesn't have weights)
    ok = make_view(c).prefetch_fields(fields=["weights"], verify_prefetch=True)
    assert ok is False


def test_erase_fields(managers):
    storage, cache, metadata_index = managers
    ParameterRepository.define_schema(metadata_index)
    c = ParameterRepository(storage, metadata_index, cache)

    p1 = make_proxy("a", ParameterType.DENSE, "m0", storage, cache, metadata_index)
    p2 = make_proxy("b", ParameterType.DENSE, "m0", storage, cache, metadata_index)

    storage.set_field(p1.meta.uid, "weights", [1])
    storage.set_field(p1.meta.uid, "bias", [0])
    storage.set_field(p2.meta.uid, "weights", [3])
    storage.set_field(p2.meta.uid, "bias", [2])

    make_view(c).erase_fields("bias")
    assert storage.has_field(p1.meta.uid, "bias") is False
    assert storage.has_field(p1.meta.uid, "weights") is True


def test_hand_built_parameter_coordinate_is_unique_and_case_sensitive(managers):
    storage, cache, index = managers
    repo = ParameterRepository.initialize(storage, index, cache)
    uid = repo.instances.get_or_create("Net")
    index.insert_rows(
        "parameters", [{"uid": "p0", "name": "Weight", "instance_uid": uid}]
    )
    other = ParameterMetadata(name="weight", ptype=ParameterType.DENSE, model_id="Net")
    ParameterDataProxy.create_and_store(other, repo)
    same = ParameterMetadata(name="Weight", ptype=ParameterType.DENSE, model_id="Net")
    with pytest.raises(DuplicateParameterError) as error:
        ParameterDataProxy.create_and_store(same, repo)
    assert (error.value.instance_uid, error.value.step, error.value.name) == (
        uid,
        0,
        "Weight",
    )
    assert isinstance(error.value.__cause__, UniqueConstraintError)
    assert error.value.__cause__.columns == ("instance_uid", "step", "name")
    assert sorted(p.meta.name for p in repo) == ["Weight", "weight"]
