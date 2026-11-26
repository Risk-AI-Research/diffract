"""Check stateless analysis, summaries, and ordered comparison roles."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from diffract import Session, analyze, compare
from diffract.containers import compute_singleton_container
from diffract.core.compute.registry import KernelRegistry
from diffract.session import ScopeValidationError, SessionError, ShapeMismatch

pytestmark = pytest.mark.unit


def test_analyze_requires_named_inputs_before_computing():
    with pytest.raises(SessionError, match=r"head\.weight \(absent from input\)"):
        analyze({"other.weight": np.eye(2)}, required_names=("head.weight",))


def test_analyze_summary_is_derived_from_returned_rows():
    first = analyze({"one": np.eye(2), "two": np.eye(2) * 2})
    second = analyze({"another": np.eye(2)}, fields=("frob_norm",))
    assert {row["param_name"] for row in first.details} == {"one", "two"}
    assert {row["param_name"] for row in second.details} == {"another"}
    selected = [
        row["value_scalar"] for row in first.details if row["field"] == "frob_norm"
    ]
    assert first.summary["frob_norm"] == {
        "count": len(selected),
        "finite_numeric_count": len(selected),
        "mean": pytest.approx(np.mean(selected)),
    }
    (esd,) = [
        row
        for row in first.details
        if row["field"] == "esd" and row["param_name"] == "two"
    ]
    assert esd["value_array"] == [2.0, 2.0]


@pytest.mark.parametrize(
    ("dtype", "scale"),
    [
        (np.float32, 1e-25),
        (np.float32, -1e20),
        (np.float32, 1e20),
        (np.float64, 1e-300),
        (np.float64, 1e-16),
        (np.float64, 1e300),
    ],
)
def test_analyze_norm_and_ranks_across_scales(dtype, scale):
    golden = json.loads(
        (Path(__file__).parent / "_golden/spectral_ranks.json").read_text()
    )
    inputs = {
        "dense": np.array(golden["matrix"]),
        "identity": np.eye(2),
        "unequal": np.diag([0.0, 3.0, 4.0]),
        "rank_one": np.array([[3.0, -4.0, 0.0], [6.0, -8.0, 0.0]]),
        "zero": np.zeros((2, 3)),
    }
    p, q = 3 / 7, 4 / 7
    size = abs(scale)
    expected = {
        "dense": (
            size * golden["frob_norm"],
            golden["stable_rank"],
            golden["effective_rank"],
        ),
        "identity": (math.sqrt(2) * size, 2.0, 2.0),
        "unequal": (5 * size, 25 / 16, math.exp(-p * math.log(p) - q * math.log(q))),
        "rank_one": (math.sqrt(125) * size, 1.0, 1.0),
        "zero": (0.0, 0.0, 1.0),
    }
    fields = ("frob_norm", "stable_rank", "effective_rank")
    with compute_singleton_container.kernel_registry.override(KernelRegistry()):
        with Session(profile="ram") as session:
            session.compute.configure_kernel("weights_svd", allow_cuda=False)
        result = analyze(
            {k: v.astype(dtype) * scale for k, v in inputs.items()}, fields=fields
        )
    values = {(r["param_name"], r["field"]): r["value_scalar"] for r in result.details}
    for name, truth in expected.items():
        measured = [values[name, f] for f in fields]
        np.testing.assert_allclose(
            measured, truth, rtol=32 * np.finfo(dtype).eps, atol=0
        )


@pytest.mark.parametrize("scale", [np.nextafter(0.0, 1.0), np.finfo(float).max])
def test_analyze_ranks_at_float64_limits(scale):
    with compute_singleton_container.kernel_registry.override(KernelRegistry()):
        with Session(profile="ram") as session:
            session.compute.configure_kernel("weights_svd", allow_cuda=False)
        result = analyze(
            {"w": scale * np.eye(2)},
            fields=("frob_norm", "stable_rank", "effective_rank"),
        )
    measured = {r["field"]: r["value_scalar"] for r in result.details}
    assert measured["frob_norm"] == math.hypot(scale, scale)
    assert measured["stable_rank"] == 2.0
    assert measured["effective_rank"] == pytest.approx(2.0, rel=1e-15)


def test_reopened_store_refreshes_versioned_ranks_and_reuses_spectrum(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    fields = ("frob_norm", "stable_rank", "effective_rank", "weights_svals")

    def snapshot(session):
        rows = session.results.export(*fields, shape="long", export_format="list")
        return {r["field"]: (r["run_id"], r["value_scalar"]) for r in rows}

    with (
        compute_singleton_container.kernel_registry.override(KernelRegistry()),
        Session(profile="local") as seed,
    ):
        seed.compute.configure_kernel("weights_svd", allow_cuda=False)

        @seed.compute.kernel(version="1")
        def frob_norm(weights_svals):
            return float(np.sqrt(np.sum(weights_svals**2)))

        @seed.compute.kernel(version="1")
        def stable_rank(frob_norm, l2_norm):
            return (frob_norm / (l2_norm + 1e-16)) ** 2

        @seed.compute.kernel(version="1")
        def effective_rank(weights_svals):
            p = weights_svals / (weights_svals.sum() + 1e-16)
            return float(np.exp(-np.sum(p * np.log(p + 1e-16))))

        seed.models.add({"w": 1e-25 * np.eye(2, dtype=np.float32)}, "example")
        seed.compute.apply(*fields)
        prior = snapshot(seed)
    with (
        compute_singleton_container.kernel_registry.override(KernelRegistry()),
        Session(profile="local") as current,
    ):
        current.compute.configure_kernel("weights_svd", allow_cuda=False)
        assert snapshot(current) == prior
        assert {
            r.field for r in current.results.status(*fields) if r.status == "stale"
        } == set(fields[:3])
        current.compute.apply(*fields)
        updated = snapshot(current)
        assert updated["weights_svals"] == prior["weights_svals"]
        for field in fields[:3]:
            assert updated[field][0] != prior[field][0]
        scale = float(np.float32(1e-25))
        norm = math.hypot(scale, scale)
        np.testing.assert_allclose(
            [updated[f][1] for f in fields[:3]], [norm, 2.0, 2.0], rtol=1e-14, atol=0.0
        )
        current.compute.apply(*fields)
        assert snapshot(current) == updated


def test_compare_retains_only_requested_order_and_real_join_keys():
    source = {"w": np.diag([1.0, 2.0])}
    target = {"w": np.array([[1.0, 2.0], [0.0, 3.0]])}
    rows = compare(source, target)
    assert len(rows) == 12
    assert {(row["model_source"], row["model_target"]) for row in rows} == {
        ("source", "target")
    }
    assert {row["member_role"] for row in rows} == {"source", "target"}
    assert len({row["aggregate_uid"] for row in rows}) == 6
    assert all(row["run_id"] for row in rows)


def test_compare_reports_extraction_join_and_field_specific_shapes():
    matrix = np.diag([1.0, 2.0])
    source = {"ok": matrix, "resize": matrix, "old": matrix, "lost": matrix}
    target = {
        "ok": matrix * 3,
        "resize": np.pad(matrix, ((0, 0), (0, 1))),
        "new": matrix,
        "lost": matrix.astype(int),
    }
    fields = (
        "relative_weight_change",
        "relative_spectrum_change",
        "avg_l_agreement",
        "avg_r_agreement",
    )
    rows, report = compare(source, target, fields=fields, return_coverage=True)
    assert report.matched == ("ok", "resize")
    assert report.source_only == ("lost", "old")
    assert report.target_only == ("new",)
    assert (report.source_count, report.target_count) == (4, 3)
    assert report.source_extraction.counts["extracted"] == 4
    assert {(i.name, i.status) for i in report.target_extraction.items} == {
        ("ok", "extracted"),
        ("resize", "extracted"),
        ("new", "extracted"),
        ("lost", "unsupported"),
    }
    mismatch = (ShapeMismatch("resize", (2, 2), (2, 3)),)
    assert report.incompatible == dict(
        zip(fields, [mismatch, (), (), mismatch], strict=True)
    )
    assert report.measurement_counts == dict(zip(fields, [1, 2, 2, 1], strict=True))
    expected = {
        (fields[0], "ok"): 2,
        (fields[1], "ok"): 2,
        (fields[1], "resize"): 0,
        (fields[2], "ok"): 1,
        (fields[2], "resize"): 1,
        (fields[3], "ok"): 1,
    }
    for role in ("source", "target"):
        values = {
            (r["field"], r["param_name"]): r["value_scalar"]
            for r in rows
            if r["member_role"] == role
        }
        assert values == pytest.approx(expected)
    assert len({r["aggregate_uid"] for r in rows}) == len(expected)
    assert len(rows) == 2 * len(expected)
    with pytest.raises(
        ScopeValidationError,
        match=r"relative_weight_change.*resize.*source.*\(2, 2\).*target.*\(2, 3\)",
    ):
        compare(source, target, fields=fields)


def test_disjoint_comparison_returns_empty_rows_and_both_denominators():
    rows, report = compare(
        {"one": np.eye(2)},
        {"two": np.eye(2), "three": np.eye(2)},
        fields=("relative_weight_change",),
        return_coverage=True,
        export_format="dict",
    )
    assert rows["aggregate_uid"] == []
    assert (report.matched, report.source_only, report.target_only) == (
        (),
        ("one",),
        ("three", "two"),
    )
    assert report.measurement_counts == {"relative_weight_change": 0}


def test_changed_row_count_and_spectrum_length_restrict_different_fields():
    fields = ("relative_spectrum_change", "avg_l_agreement", "avg_r_agreement")
    rows, report = compare(
        {"tall": np.eye(3, 2), "wide": np.eye(2, 3)},
        {"tall": np.eye(4, 2), "wide": np.eye(3, 3)},
        fields=fields,
        return_coverage=True,
    )
    assert report.measured == {
        fields[0]: ("tall",),
        fields[1]: (),
        fields[2]: ("tall",),
    }
    assert {(r["field"], r["param_name"]) for r in rows} == {
        (fields[0], "tall"),
        (fields[2], "tall"),
    }
    assert report.incompatible[fields[2]] == (ShapeMismatch("wide", (2, 3), (3, 3)),)


def test_missing_required_name_on_an_empty_endpoint():
    with pytest.raises(ScopeValidationError, match=r"required|Required") as caught:
        compare({"w": np.eye(2)}, {"other": np.eye(2, dtype=int)}, required_names="w")
    assert "'w'" in str(caught.value)
    assert "target" in str(caught.value)


@pytest.mark.parametrize("reverse", [False, True])
def test_empty_extraction_retains_both_comparison_outcomes(reverse):
    valid = {"w": np.eye(2)}
    invalid = {"w": np.eye(2, dtype=int), "bias": np.zeros(2)}
    endpoints = (invalid, valid) if reverse else (valid, invalid)
    rows, report = compare(
        *endpoints, fields=("relative_weight_change",), return_coverage=True
    )
    assert rows == []
    assert report.matched == ()
    assert report.measurement_counts == {"relative_weight_change": 0}
    observed = (report.source_extraction, report.target_extraction)
    supported, rejected = observed[::-1] if reverse else observed
    assert [(i.name, i.status) for i in supported.items] == [("w", "extracted")]
    assert {(i.name, i.status) for i in rejected.items} == {
        ("w", "unsupported"),
        ("bias", "excluded"),
    }
    assert (report.source_only, report.target_only) == (
        ((), ("w",)) if reverse else (("w",), ())
    )


@pytest.mark.parametrize(("reverse", "role"), [(False, "target"), (True, "source")])
def test_required_comparison_name_identifies_its_endpoint(reverse, role):
    left, right = {"w": np.eye(2)}, {"other": np.eye(2), "w": np.eye(2, dtype=int)}
    endpoints = (right, left) if reverse else (left, right)
    with pytest.raises(ScopeValidationError, match=rf"{role}.*{role}.*'w'.*handler"):
        compare(*endpoints, fields=("relative_weight_change",), required_names="w")


def test_compare_separates_scale_change_from_rotation_of_leading_components():
    weights = np.diag(np.arange(1.0, 65.0))
    rotation = np.eye(64)
    angle = np.pi / 3
    rotation[-2:, -2:] = [
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)],
    ]

    def source_rows(target):
        return {
            row["field"]: row
            for row in compare({"w": weights}, {"w": target})
            if row["member_role"] == "source"
        }

    scaled = source_rows(2 * weights)
    assert scaled["relative_weight_change"]["value_scalar"] == pytest.approx(1)
    assert scaled["relative_spectrum_change"]["value_scalar"] == pytest.approx(1)
    np.testing.assert_allclose(scaled["l_agreement"]["value_array"], 1)

    rotated_weights = rotation @ weights
    rotated = source_rows(rotated_weights)
    assert rotated["relative_weight_change"]["value_scalar"] == pytest.approx(
        np.linalg.norm(rotated_weights - weights) / np.linalg.norm(weights)
    )
    assert rotated["relative_spectrum_change"]["value_scalar"] == pytest.approx(
        0, abs=1e-12
    )
    left = np.asarray(rotated["l_agreement"]["value_array"])
    np.testing.assert_allclose(left[:-2], 1)
    np.testing.assert_allclose(left[-2:], 0.5)
    np.testing.assert_allclose(rotated["r_agreement"]["value_array"], 1)
