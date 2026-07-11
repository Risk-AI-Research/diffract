"""Unit tests for SVD and ESD kernels."""

from __future__ import annotations

import numpy as np
import pytest

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
