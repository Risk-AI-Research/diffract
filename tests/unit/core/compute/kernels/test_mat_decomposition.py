"""Unit tests for SVD and ESD kernels.

The ESD and singular-value accessors are fed hand-built arrays directly; they
encode the ascending-sort contract that `svd` guarantees for its consumers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from hypothesis import (
    given,
    strategies as st,
)
from hypothesis.extra import numpy as hnp
from scipy import linalg

if TYPE_CHECKING:
    from numpy.typing import NDArray

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("rows", "cols"),
    [(64, 32), (32, 64), (48, 48), (7, 3)],
    ids=["tall", "wide", "square", "tiny"],
)
def test_svd_returns_economy_factors(ram_container, rows: int, cols: int) -> None:
    from diffract.core.compute.kernels.mat_decomposition import svd

    rng = np.random.default_rng(0)
    mat = rng.standard_normal((rows, cols))
    k = min(rows, cols)

    lsvs, svals, rsvs = svd(mat, allow_cuda=False)

    assert lsvs.shape == (rows, k)
    assert svals.shape == (k,)
    assert rsvs.shape == (cols, k)
    assert np.all(np.diff(svals) >= 0)

    reconstructed = lsvs @ np.diag(svals) @ rsvs.T
    np.testing.assert_allclose(reconstructed, mat, atol=1e-8)


def test_svd_singular_values_match_reference(ram_container) -> None:
    from diffract.core.compute.kernels.mat_decomposition import svd

    rng = np.random.default_rng(1)
    mat = rng.standard_normal((64, 32))

    _, svals, _ = svd(mat, allow_cuda=False)
    reference = np.sort(np.linalg.svd(mat, compute_uv=False))

    np.testing.assert_allclose(svals, reference, rtol=1e-12)


def test_host_memory_probe_on_supported_platforms() -> None:
    import sys

    from diffract.core.compute.execution.host_memory import available_host_bytes

    if sys.platform not in {"darwin", "linux", "win32"}:
        pytest.skip("The host memory probe does not cover this platform")
    available = available_host_bytes()
    assert available is not None
    assert available >= 0


def test_cpu_svd_rejects_insufficient_workspace_before_calling_lapack(
    ram_container, monkeypatch
) -> None:
    from diffract import Session
    from diffract.core.compute.extensions import svd_workspace
    from diffract.session import SessionError

    matrix = np.eye(64, dtype=np.float32)
    estimate = svd_workspace.estimate_svd_workspace_bytes(matrix, vectors=True)
    assert estimate > matrix.nbytes
    monkeypatch.setattr(svd_workspace, "available_host_bytes", lambda: estimate)

    def unexpected_svd(*args, **kwargs):
        pytest.fail("LAPACK was called despite the memory budget")

    monkeypatch.setattr(linalg, "svd", unexpected_svd)
    with Session(container=ram_container) as session:
        session.models.add({"w": matrix}, model_id="net")
        session.compute.configure_kernel("weights_svd", allow_cuda=False)
        with pytest.raises(
            SessionError, match=r"CPU SVD for \(64, 64\).*working memory"
        ):
            session.compute.apply("weights_svals")
        (status,) = session.results.status("weights_svals")
        assert status.status == "never computed"


def test_local_svd_loads_each_parameter_when_it_is_executed(
    tmp_path, monkeypatch
) -> None:
    from diffract import Session
    from diffract.core.storage.catalog_manager import CatalogStorageManager

    monkeypatch.chdir(tmp_path)
    with Session(profile="local") as seed:
        seed.models.add({"a": np.eye(4), "b": np.eye(5)}, model_id="net")

    calls = []
    read_pointer = CatalogStorageManager.read_pointer
    svd = linalg.svd

    def log_read(self, pointer):
        if pointer.storage_key.endswith("/weights"):
            calls.append("read")
        return read_pointer(self, pointer)

    def log_svd(*args, **kwargs):
        calls.append("svd")
        return svd(*args, **kwargs)

    monkeypatch.setattr(CatalogStorageManager, "read_pointer", log_read)
    monkeypatch.setattr(linalg, "svd", log_svd)
    with Session(profile="local") as session:
        session.compute.configure_kernel("weights_svd", allow_cuda=False)
        session.compute.apply("weights_svals")
        assert calls == ["read", "svd", "read", "svd"]
        values = session.results.export_metrics("weights_svals", export_format="dict")
        assert len(values) == 2
        for row in values.values():
            np.testing.assert_array_equal(row["fields"]["weights_svals"], 1)


def test_random_spectrum_writes_vectors_only_when_requested(ram_container) -> None:
    from diffract import Session

    weights = np.random.default_rng(3).standard_normal((18, 7), dtype=np.float32)
    with Session(container=ram_container) as session:
        session.models.add({"layer": weights}, model_id="net")
        summary = session.compute.apply("weights_rand_svals")
        assert summary.parameter_fields == ("weights_rand", "weights_rand_svals")
        fields = session.results.export_metrics(
            "weights_rand", "weights_rand_svals", export_format="dict"
        )
        (row,) = fields.values()
        shuffled = row["fields"]["weights_rand"]
        spectrum = row["fields"]["weights_rand_svals"]
        expected = np.sort(np.linalg.svd(shuffled, full_matrices=False)[1])
        np.testing.assert_allclose(spectrum, expected, rtol=1e-6)

        summary = session.compute.apply("weights_rand_lsvs")
        assert summary.parameter_fields == ("weights_rand_lsvs", "weights_rand_rsvs")
        fields = session.results.export_metrics(
            "weights_rand_lsvs", "weights_rand_rsvs", export_format="dict"
        )
        (row,) = fields.values()
        left, right = (row["fields"][f"weights_rand_{side}svs"] for side in ("l", "r"))
        np.testing.assert_allclose(
            (left * spectrum) @ right.T, shuffled, rtol=1e-5, atol=1e-6
        )


@pytest.mark.gpu
def test_svd_cuda_matches_cpu(ram_container) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from diffract.core.compute.kernels.mat_decomposition import svd

    rng = np.random.default_rng(2)
    mat = rng.standard_normal((64, 32)).astype(np.float32)

    lsvs_gpu, svals_gpu, rsvs_gpu = svd(mat, allow_cuda=True)
    lsvs_cpu, svals_cpu, rsvs_cpu = svd(mat, allow_cuda=False)

    np.testing.assert_allclose(svals_gpu, svals_cpu, rtol=1e-4)
    assert lsvs_gpu.shape == lsvs_cpu.shape
    assert rsvs_gpu.shape == rsvs_cpu.shape


def test_esd_is_squared_svals_over_greater_dim() -> None:
    from diffract.core.compute.kernels.mat_decomposition import esd

    result = esd(np.array([2.0, 3.0]), greater_dim=4)

    np.testing.assert_allclose(result, [1.0, 2.25])


@given(
    svals=hnp.arrays(
        dtype=np.float64,
        shape=st.integers(min_value=1, max_value=50),
        elements=st.floats(
            min_value=1e-3, max_value=1e3, allow_nan=False, allow_infinity=False
        ),
    ),
    scale=st.floats(min_value=0.1, max_value=10.0),
)
def test_esd_is_scale_squared_equivariant(
    svals: NDArray[np.float64], scale: float
) -> None:
    from diffract.core.compute.kernels.mat_decomposition import esd

    np.testing.assert_allclose(
        esd(scale * svals, greater_dim=8),
        scale**2 * esd(svals, greater_dim=8),
        rtol=1e-9,
    )


def test_singular_value_accessors_read_ascending_ends() -> None:
    from diffract.core.compute.kernels.mat_decomposition import max_sval, min_sval

    ascending = np.array([1.0, 2.0, 3.0, 5.0])

    assert max_sval(ascending) == pytest.approx(5.0)
    assert min_sval(ascending) == pytest.approx(1.0)


def test_esd_accessors_read_ascending_ends() -> None:
    from diffract.core.compute.kernels.mat_decomposition import esd_max, esd_min

    ascending = np.array([0.1, 0.5, 2.0])

    assert esd_max(ascending) == pytest.approx(2.0)
    assert esd_min(ascending) == pytest.approx(0.1)


def test_svd_survives_rank_collapsed_matrix() -> None:
    from diffract.core.compute.kernels.mat_decomposition import svd

    # A dead / zero-initialised layer collapses to a zero spectrum.
    _, svals, _ = svd(np.zeros((4, 3)), allow_cuda=False)

    assert np.all(svals == 0.0)


def test_svd_propagates_non_finite_weights() -> None:
    from diffract.core.compute.kernels.mat_decomposition import svd

    # A diverged checkpoint passes extraction unvalidated; a non-finite weight
    # poisons the whole spectrum to nan (all-or-nothing) rather than crashing.
    _, svals, _ = svd(np.array([[np.inf, 2.0], [3.0, 4.0]]), allow_cuda=False)

    assert np.all(np.isnan(svals))


def test_svd_propagates_nan_weights() -> None:
    from diffract.core.compute.kernels.mat_decomposition import svd

    # A nan-corrupted checkpoint makes the LAPACK driver fail to converge; svd
    # must mirror the inf path and return all-nan factors so the nan contract
    # holds, rather than raising LinAlgError with a misleading message.
    lsvs, svals, rsvs = svd(np.array([[np.nan, 2.0], [3.0, 4.0]]), allow_cuda=False)

    assert np.all(np.isnan(svals))
    assert np.all(np.isnan(lsvs))
    assert np.all(np.isnan(rsvs))


def test_spectrum_kernels_propagate_nan_corruption() -> None:
    from diffract.core.compute.kernels.mat_decomposition import (
        esd,
        esd_max,
        esd_min,
        max_sval,
        min_sval,
    )

    # An inf-corrupted checkpoint yields an all-nan spectrum; every consumer
    # must carry the nan through rather than fabricate a finite number.
    nan_spectrum = np.full(3, np.nan)

    assert np.all(np.isnan(esd(nan_spectrum, greater_dim=4)))
    assert np.isnan(max_sval(nan_spectrum))
    assert np.isnan(min_sval(nan_spectrum))
    assert np.isnan(esd_max(nan_spectrum))
    assert np.isnan(esd_min(nan_spectrum))


def test_spectrum_kernels_survive_zero_spectrum() -> None:
    from diffract.core.compute.kernels.mat_decomposition import (
        esd,
        esd_max,
        max_sval,
        min_sval,
    )

    zeros = np.zeros(3)

    assert np.all(esd(zeros, greater_dim=4) == 0.0)
    assert max_sval(zeros) == 0.0
    assert min_sval(zeros) == 0.0
    assert esd_max(zeros) == 0.0
