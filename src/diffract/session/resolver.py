"""Single identity resolver for user-facing field addresses.

Maps every accepted address form — a plain field name, a legacy contextual
grammar string (``metric@models[m1,m2]@params[p1]``), a viz ``FieldRef``,
or a :class:`FieldSelector` — to structured selector components. This
module is the only interpreter of the legacy grammar; :func:`render_label`
renders labels from structure, never parses them back.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from diffract.core.data.identity import training_step
from diffract.session.errors import ModelNotFoundError

if TYPE_CHECKING:
    from diffract.core.data.nn.instances.repository import InstanceRepository
    from diffract.core.data.nn.params.interface import IParameterView
    from diffract.core.data.nn.params.schema import StepSelection

__all__ = [
    "FieldAddress",
    "FieldSelector",
    "render_label",
    "resolve",
    "resolve_training_coordinates",
]

_COORDINATE_PARTS = 2
_SEPARATOR = "@"
_MODELS_KEY = "models"
_PARAMS_KEY = "params"
_MODELS_PART = re.compile(rf"^{_MODELS_KEY}\[(.+)\]$")
_PARAMS_PART = re.compile(rf"^{_PARAMS_KEY}\[(.+)\]$")


@dataclass(frozen=True)
class FieldSelector:
    """Structured address of a stored value.

    Args:
        field: Base field name.
        models: Model names constraining the address, or None when
            unconstrained.
        params: Parameter names constraining the address, or None when
            unconstrained.
        steps: Training steps in source, target order when roles are given.
            Otherwise, step order is ignored.
        roles: Source and target model names for an ordered pair.
    """

    field: str
    models: tuple[str, ...] | None = None
    params: tuple[str, ...] | None = None
    steps: tuple[int, ...] | None = None
    roles: tuple[str, ...] | None = None

    @property
    def is_contextual(self) -> bool:
        """Whether the address constrains models, parameters, steps, or roles."""
        return any(
            item is not None
            for item in (self.models, self.params, self.steps, self.roles)
        )

    @classmethod
    def from_context(cls, record: dict[str, Any]) -> FieldSelector:
        """Build a selector from a decoded context dictionary."""

        def slot(name: str) -> tuple | None:
            items = record.get(name)
            return tuple(items) if items is not None else None

        return cls(
            record["field"],
            slot("models"),
            slot("params"),
            slot("steps"),
            slot("roles"),
        )

    def matches(self, candidate: FieldSelector) -> bool:
        """Match set constraints independently of order and pair slots in order."""
        if self.field != candidate.field or (
            self.roles is not None and self.roles != candidate.roles
        ):
            return False
        for requested, actual in (
            (self.models, candidate.models),
            (self.params, candidate.params),
        ):
            if requested is not None and set(requested) != set(actual or ()):
                return False
        if self.steps is None:
            return True
        if self.roles is not None:
            return self.steps == candidate.steps
        return set(self.steps) == set(candidate.steps or ())


def aggregate_selector(record: dict[str, Any]) -> FieldSelector:
    """Build a selector from exported membership and ordered operand slots."""
    directed = "model_source" in record
    return FieldSelector(
        field=record["field"],
        models=tuple(record["context_models"]) or None,
        params=tuple(record["context_params"]) or None,
        roles=(record["model_source"], record["model_target"]) if directed else None,
        steps=(record["step_source"], record["step_target"])
        if directed
        else tuple(sorted({member["step"] for member in record.get("members", ())}))
        or None,
    )


class SupportsField(Protocol):
    """A visualization or API reference carrying a field address."""

    @property
    def field(self) -> str | FieldSelector: ...


type FieldAddress = str | FieldSelector | SupportsField


@dataclass(frozen=True)
class ModelSelection:
    """Model names and checkpoint coordinates selected inside a view."""

    names: tuple[str, ...]
    instances: tuple[str, ...]
    coordinates: tuple[tuple[str, int], ...]


def resolve_models(
    view: IParameterView,
    instances: InstanceRepository,
    targets: Sequence[str | tuple[str, int]],
    *,
    allow_empty_instances: bool,
) -> ModelSelection:
    """Resolve exact model names and explicit checkpoints inside a view."""
    by_instance: dict[str, list[int]] = {}
    for uid, step in view.coordinates():
        by_instance.setdefault(uid, []).append(step)
    names: dict[str, str] = {}
    selected: dict[tuple[str, int], None] = {}
    for target in targets:
        if isinstance(target, str):
            name, step = target, None
        elif (
            isinstance(target, tuple)
            and len(target) == _COORDINATE_PARTS
            and isinstance(target[0], str)
        ):
            name, step = target[0], training_step(target[1])
        else:
            raise TypeError("Expected a model name or a (model name, step) tuple.")
        instance = instances.get_by_name(name)
        if instance is None:
            raise ModelNotFoundError(f"Model '{name}' not found")
        uid = instance.instance_uid
        visible = by_instance.get(uid, [])
        if step is not None and step not in visible:
            raise ModelNotFoundError(
                f"Model '{name}' at step {step} not found in this scope"
            )
        if step is None and not visible and not allow_empty_instances:
            raise ModelNotFoundError(f"Model '{name}' not found in this scope")
        names[name] = uid
        for number in visible if step is None else (step,):
            selected[uid, number] = None
    return ModelSelection(tuple(names), tuple(names.values()), tuple(selected))


def resolve_training_coordinates(
    view: IParameterView, steps: StepSelection
) -> list[tuple[str, int]]:
    """Resolve integers and per-model final aliases within a parameter view."""
    selectors = (
        steps
        if isinstance(steps, Sequence) and not isinstance(steps, str)
        else (steps,)
    )
    explicit = {training_step(step) for step in selectors if step != "final"}
    latest: dict[str, int] = {}
    if "final" in selectors:
        for instance_uid, step in view.coordinates(whole_store=True):
            latest[instance_uid] = max(step, latest.get(instance_uid, step))
    return [
        (instance, step)
        for instance, step in view.coordinates()
        if step in explicit or step == latest.get(instance)
    ]


def resolve(address: FieldAddress) -> FieldSelector:
    """Resolve a user-facing field address into a structured selector.

    Args:
        address: A field name, legacy input address, structured selector,
            or reference carrying one of these in its ``field`` attribute.

    Returns:
        The structured selector, with component order as written.

    Raises:
        TypeError: If the address is none of the accepted kinds.
    """
    if isinstance(address, FieldSelector):
        return address
    if isinstance(address, str):
        return _resolve_string(address)
    ref_field = getattr(address, "field", None)
    if isinstance(ref_field, (str, FieldSelector)):
        return resolve(ref_field)
    msg = (
        f"Cannot resolve a field address of type {type(address).__name__}; "
        "expected a field name string, a FieldRef, or a FieldSelector"
    )
    raise TypeError(msg)


def render_label(
    selector: FieldSelector,
    *,
    members: Sequence[tuple[str, int, str, str | None]] = (),
) -> str:
    """Render member coordinates or a requested context for display."""
    parts: list[str] = []
    grouped: dict[tuple[str, int, str | None], list[str]] = {}
    for name, step, parameter, label in members:
        grouped.setdefault((name, step, label), []).append(parameter)

    def coordinate(name: str, step: int, label: str | None) -> str:
        caption = f", {label!r}" if label is not None else ""
        return f"{name} (step {step}{caption})"

    if selector.roles:
        for index, (role, name) in enumerate(
            zip(("source", "target"), selector.roles, strict=True)
        ):
            if members:
                model, step, parameter, label = members[index]
                caption = f"{coordinate(model, step, label)} / {parameter}"
            elif selector.steps:
                caption = coordinate(name, selector.steps[index], None)
            else:
                caption = name
            parts.append(f"{role}={caption}")
    elif grouped:
        for (name, step, label), parameters in sorted(grouped.items()):
            parts.append(
                f"{coordinate(name, step, label)} / {', '.join(sorted(parameters))}"
            )
    else:
        if selector.models:
            parts.append(f"models={', '.join(sorted(selector.models))}")
        if selector.steps:
            parts.append(f"steps={', '.join(map(str, sorted(selector.steps)))}")
    if selector.params and not members:
        parts.append(f"parameters={', '.join(sorted(selector.params))}")
    return f"{selector.field} [{', '.join(parts)}]" if parts else selector.field


def _resolve_string(address: str) -> FieldSelector:
    """Parse a raw string address; the single interpreter of the grammar.

    A string without the separator is a plain field name; suffix parts
    that are not canonical ``models[...]`` / ``params[...]`` groups
    contribute no context.
    """
    if _SEPARATOR not in address:
        return FieldSelector(field=address)

    parts = address.split(_SEPARATOR)
    models: tuple[str, ...] | None = None
    params: tuple[str, ...] | None = None

    for part in parts[1:]:
        if match := _MODELS_PART.match(part):
            models = tuple(match.group(1).split(","))
        elif match := _PARAMS_PART.match(part):
            params = tuple(match.group(1).split(","))

    return FieldSelector(field=parts[0], models=models, params=params)
