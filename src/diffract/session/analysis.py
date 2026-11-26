"""Stateless entry points for weight analysis and checkpoint comparison."""

from __future__ import annotations

from dataclasses import dataclass
from math import fsum, isfinite
from typing import TYPE_CHECKING, Any, Literal, overload

from diffract.core.export.formatters.registry import get_long_formatter

from .errors import EmptyExtractionError, ScopeValidationError
from .resolver import FieldSelector
from .session import Session
from .summaries import PairCoverage

if TYPE_CHECKING:
    from collections.abc import Iterable

    from diffract.core.data.nn.extractors.report import ExtractionReport

_ANALYZE_FIELDS = (
    "frob_norm",
    "l2_norm",
    "lower_dim",
    "stable_rank",
    "effective_rank",
    "esd",
)
_COMPARE_FIELDS = (
    "relative_weight_change",
    "relative_spectrum_change",
    "l_agreement",
    "r_agreement",
    "max_l_agreement",
    "max_r_agreement",
)


@dataclass(frozen=True)
class AnalysisResult:
    """Long-form parameter values and their per-field numeric summaries."""

    details: Any
    summary: dict[str, dict[str, int | float | None]]


def _check_fields(session: Session, fields: tuple[str, ...], level: str) -> None:
    if not fields:
        raise ValueError("At least one field must be specified")
    for field in fields:
        actual = session.compute.describe_field(field)["level"]
        if actual != level:
            raise ValueError(
                f"Field {field!r} has level {actual}; this call requires {level}."
            )


def _summarize(
    rows: list[dict[str, Any]], fields: tuple[str, ...]
) -> dict[str, dict[str, int | float | None]]:
    numeric: dict[str, list[float]] = {field: [] for field in fields}
    counts = dict.fromkeys(fields, 0)
    for row in rows:
        field = row["field"]
        counts[field] += 1
        value = row["value_scalar"]
        if value is None:
            value = row["value_integer"]
        if value is not None and isfinite(value):
            numeric[field].append(value)
    return {
        field: {
            "count": counts[field],
            "finite_numeric_count": len(values),
            "mean": fsum(value / len(values) for value in values) if values else None,
        }
        for field, values in numeric.items()
    }


@overload
def analyze(
    model: Any,
    *,
    fields: tuple[str, ...] = ...,
    export_format: str = ...,
    required_names: Iterable[str] = ...,
    return_coverage: Literal[False] = ...,
) -> AnalysisResult: ...


@overload
def analyze(
    model: Any,
    *,
    fields: tuple[str, ...] = ...,
    export_format: str = ...,
    required_names: Iterable[str] = ...,
    return_coverage: Literal[True],
) -> tuple[AnalysisResult, ExtractionReport]: ...


def analyze(
    model: Any,
    *,
    fields: tuple[str, ...] = _ANALYZE_FIELDS,
    export_format: str = "list",
    required_names: Iterable[str] = (),
    return_coverage: bool = False,
) -> AnalysisResult | tuple[AnalysisResult, ExtractionReport]:
    """Analyze weight matrices in a fresh in-memory session.

    Args:
        model: A supported model object or mapping of names to weight matrices.
        fields: Registered parameter fields. Defaults to two norms, the
            spectrum size, stable and effective rank, and the full ESD.
        export_format: Long result format, such as "list", "pandas", or "polars".
        required_names: Exact source parameter names that must be extracted.
        return_coverage: Return the analysis and its extraction report as a pair.

    Returns:
        An AnalysisResult with long result rows and numeric summaries, or a
        pair of the AnalysisResult and ExtractionReport when return_coverage is true.
    """
    format_rows = get_long_formatter(export_format)
    with Session(profile="ram") as session:
        _check_fields(session, fields, "PARAMETER")
        coverage = session.models.add(
            model, model_id="model", required_names=required_names
        )
        session.compute.apply(*fields)
        rows = session.results.export(
            *fields, shape="long", sources="metrics", export_format="list"
        )
    result = AnalysisResult(format_rows(rows), _summarize(rows, fields))
    return (result, coverage) if return_coverage else result


@overload
def compare(
    source: Any,
    target: Any,
    *,
    fields: tuple[str, ...] = _COMPARE_FIELDS,
    export_format: str = "list",
    return_coverage: Literal[False] = False,
    required_names: Iterable[str] = (),
) -> Any: ...


@overload
def compare(
    source: Any,
    target: Any,
    *,
    fields: tuple[str, ...] = _COMPARE_FIELDS,
    export_format: str = "list",
    return_coverage: Literal[True],
    required_names: Iterable[str] = (),
) -> tuple[Any, PairCoverage]: ...


@overload
def compare(
    source: Any,
    target: Any,
    *,
    fields: tuple[str, ...] = _COMPARE_FIELDS,
    export_format: str = "list",
    return_coverage: bool,
    required_names: Iterable[str] = (),
) -> Any | tuple[Any, PairCoverage]: ...


def compare(
    source: Any,
    target: Any,
    *,
    fields: tuple[str, ...] = _COMPARE_FIELDS,
    export_format: str = "list",
    return_coverage: bool = False,
    required_names: Iterable[str] = (),
) -> Any:
    """Compare same-name weight matrices in two models in source-to-target order.

    The full set of default fields requires matched matrices of the same shape.

    Args:
        source: The source model or named weight matrices.
        target: The target model or named weight matrices.
        fields: Registered cross-model fields. Defaults to source-relative
            weight and singular-value changes and the per-component diagonal
            and best-match profiles for both singular-vector bases.
        export_format: Long result format, such as "list", "pandas", or "polars".
        return_coverage: Return the long rows and a PairCoverage report. Each
            field is evaluated on names with compatible matrix dimensions.
        required_names: Stored parameter names required at both endpoints with
            compatible dimensions for every requested field.

    Returns:
        Long aggregate rows with the source and target roles labeled, or
        a pair of the formatted rows and their report when coverage is requested.
    """
    format_rows = get_long_formatter(export_format)
    with Session(profile="ram") as session:
        _check_fields(session, fields, "CROSS_MODEL")
        required = (
            (required_names,)
            if isinstance(required_names, str)
            else tuple(required_names)
        )
        extractions = {}
        empty_endpoint = False
        for endpoint, model in (("source", source), ("target", target)):
            try:
                extractions[endpoint] = session.models.add(model, model_id=endpoint)
            except EmptyExtractionError as exc:
                if required:
                    raise ScopeValidationError(
                        f"Required names {required!r} missing from {endpoint}: {exc}"
                    ) from exc
                if not return_coverage:
                    raise
                extractions[endpoint] = exc.coverage
                empty_endpoint = True
        if empty_endpoint:

            def stored_names(extraction: ExtractionReport) -> tuple[str, ...]:
                return tuple(
                    sorted(
                        i.stored_name
                        for i in extraction.items
                        if i.stored_name is not None
                    )
                )

            left, right = extractions["source"], extractions["target"]
            return format_rows([]), PairCoverage(
                source=("source", 0),
                target=("target", 0),
                source_only=stored_names(left),
                target_only=stored_names(right),
                matched=(),
                source_extraction=left,
                target_extraction=right,
                incompatible=dict.fromkeys(fields, ()),
                measured=dict.fromkeys(fields, ()),
            )
        summary = session.compute.apply(
            *fields,
            pairs=[("source", "target")],
            return_coverage=return_coverage,
            required_names=required,
        )
        report = summary.pair_coverage[0] if return_coverage else None
        selected = tuple(f for f in fields if report is None or report.measured[f])
        rows = (
            session.results.export(
                *(
                    FieldSelector(field, roles=("source", "target"))
                    for field in selected
                ),
                shape="long",
                sources="aggregates",
                export_format=export_format,
            )
            if selected
            else format_rows([])
        )
        return (rows, report) if return_coverage else rows
