from diffract.core.data.nn.aggregates.view import AggregateView
from diffract.core.data.nn.params import ParameterType
from diffract.core.data.nn.params.interface import IParameterView


def filter_parameter_view(
    view: IParameterView,
    *,
    parameter_ids: list[str] | None = None,
    parameter_names: list[str] | None = None,
    parameter_types: list[ParameterType] | None = None,
    model_ids: list[str] | None = None,
) -> IParameterView:
    """Apply the given filters to a parameter view.

    Each filter is only applied when a non-empty value is provided. Scalar
    values are wrapped into single-element lists before filtering.

    Args:
        view: The parameter view to filter.
        parameter_ids: Parameter IDs to select, or None to skip ID filtering.
        parameter_names: Parameter names to filter by, or None to skip.
        parameter_types: Parameter types to filter by, or None to skip.
        model_ids: Model IDs to filter by, or None to skip.

    Returns:
        The filtered parameter view.
    """
    if parameter_names:
        if isinstance(parameter_names, str):
            parameter_names = [parameter_names]
        view = view.filter_by_name(*parameter_names)

    if parameter_types:
        if isinstance(parameter_types, (str, ParameterType)):
            parameter_types = [parameter_types]
        view = view.filter_by_ptype(*parameter_types)

    if model_ids:
        if isinstance(model_ids, str):
            model_ids = [model_ids]
        view = view.filter_by_model_id(*model_ids)

    if parameter_ids:
        if isinstance(parameter_ids, (str, int)):
            parameter_ids = [parameter_ids]
        view = view[parameter_ids]

    return view


def filter_aggregate_view(
    view: AggregateView,
    *,
    parameter_names: list[str] | None = None,
    model_ids: list[str] | None = None,
) -> AggregateView:
    """Apply the given filters to an aggregate view.

    Each filter is only applied when a non-empty value is provided. Scalar
    values are wrapped into single-element lists before filtering.

    Args:
        view: The aggregate view to filter.
        parameter_names: Context parameter names to filter by, or None to skip.
        model_ids: Context model IDs to filter by, or None to skip.

    Returns:
        The filtered aggregate view.
    """
    if parameter_names:
        if isinstance(parameter_names, str):
            parameter_names = [parameter_names]
        view = view.filter_by_context_params(*parameter_names)

    if model_ids:
        if isinstance(model_ids, str):
            model_ids = [model_ids]
        view = view.filter_by_context_models(*model_ids)

    return view
