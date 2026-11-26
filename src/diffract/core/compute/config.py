"""Kernel configuration management and validation."""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .exceptions import InvalidConfiguration

if TYPE_CHECKING:
    from collections.abc import Callable

CONFIG_HASH_SCHEME = 1


@dataclass(frozen=True, slots=True)
class KernelConfigIdentity:
    """A versioned configuration fingerprint and its effective inputs."""

    kernel_name: str
    kernel_version: str
    config_json: str
    config_hash: str
    hash_scheme: int
    requires_kernel_trace: bool = False


@dataclass(frozen=True, slots=True)
class KernelResult:
    """Kernel output with optional arithmetic facts reported for this invocation."""

    value: Any
    input_dtype: str | None = None
    working_dtype: str | None = None
    output_dtype: str | None = None
    backend: str | None = None
    device: str | None = None
    solver: str | None = None
    driver: str | None = None

    def arithmetic(self) -> dict[str, str | None]:
        """Return the execution facts without copying the numerical payload."""
        return {
            key: getattr(self, key)
            for key in (
                "input_dtype",
                "working_dtype",
                "output_dtype",
                "backend",
                "device",
                "solver",
                "driver",
            )
        }


@dataclass(frozen=True, slots=True)
class OutputComparison:
    """A kernel's equivalence predicate and optional array comparison budget.

    The byte budget covers the previous and new array payloads together.
    Without an array budget, comparisons are limited to real scalars.
    """

    equivalent: Callable[[object, object], bool]
    max_array_bytes: int = 0

    def __post_init__(self) -> None:
        """Validate the comparison predicate and byte budget."""
        if not callable(self.equivalent):
            raise InvalidConfiguration("Output equivalence must be callable")
        if type(self.max_array_bytes) is not int or self.max_array_bytes < 0:
            raise InvalidConfiguration(
                "The array comparison budget must be non-negative"
            )


def _validate_text(value: str, path: str) -> None:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as err:
        raise InvalidConfiguration(f"{path} must contain valid Unicode text") from err


def _validate_config_value(value: Any, path: str, active: set[int]) -> None:
    value_type = type(value)
    if value is None or value_type in {bool, int}:
        return
    if value_type is str:
        _validate_text(value, path)
        return
    if value_type is float:
        if not math.isfinite(value):
            raise InvalidConfiguration(f"{path} must be finite")
        return
    if value_type not in {list, dict}:
        raise InvalidConfiguration(
            f"{path} has unsupported type {value_type.__name__}; use a JSON value"
        )
    if id(value) in active:
        raise InvalidConfiguration(f"{path} contains a circular reference")
    active.add(id(value))
    try:
        if value_type is list:
            for index, child in enumerate(value):
                _validate_config_value(child, f"{path}[{index}]", active)
        else:
            for key, child in value.items():
                if type(key) is not str:
                    raise InvalidConfiguration(f"{path} keys must be strings")
                _validate_text(key, f"{path} key")
                _validate_config_value(child, f"{path}[{key!r}]", active)
    finally:
        active.remove(id(value))


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


class KernelConfig:
    """Configuration container with defaults and safe updates.

    Manages kernel configuration parameters with validation against
    predefined defaults and safe merging of configuration updates.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize configuration with default values.

        Args:
            **kwargs: Default configuration parameters.
        """
        self._defaults: dict[str, Any] = dict(kwargs)
        self._values: dict[str, Any] = dict(kwargs)

    def copy_for_registration(self) -> KernelConfig:
        """Copy configuration without sharing mutable values with its defaults."""
        copied = deepcopy(self)
        copied._defaults = deepcopy(self._defaults)  # noqa: SLF001 - same class
        return copied

    def identity(
        self, name: str, version: str, *, execution: dict[str, str] | None = None
    ) -> KernelConfigIdentity:
        """Return effective settings and their execution identity.

        Hash the kernel name, version, non-default values and execution settings.
        """
        try:
            _validate_text(name, "kernel name")
            _validate_text(version, "kernel version")
            values = self._values.copy()
            _validate_config_value(values, "configuration", set())
            _validate_config_value(self._defaults, "defaults", set())
            effective = _canonical_json(values)
            overrides = {
                key: value
                for key, value in json.loads(effective).items()
                if _canonical_json(value) != _canonical_json(self._defaults[key])
            }
            parts = [CONFIG_HASH_SCHEME, name, version, overrides]
            if execution is not None:
                parts.append(execution)
            identity = _canonical_json(parts)
        except RecursionError as err:
            raise InvalidConfiguration(
                "Kernel configuration is nested too deeply"
            ) from err
        except ValueError as err:
            raise InvalidConfiguration(
                f"Cannot encode kernel configuration: {err}"
            ) from err
        return KernelConfigIdentity(
            kernel_name=name,
            kernel_version=version,
            config_json=effective,
            config_hash=hashlib.sha256(identity.encode("ascii")).hexdigest(),
            hash_scheme=CONFIG_HASH_SCHEME,
            requires_kernel_trace=execution is not None,
        )

    def as_dict(self) -> dict[str, Any]:
        """Return current configuration as a dictionary.

        Returns:
            Dictionary containing current configuration values.
        """
        return dict(self._values)

    def update(self, other: KernelConfig) -> None:
        """Update current configuration with another KernelConfig.

        Args:
            other: KernelConfig instance containing updates.

        Raises:
            InvalidConfiguration: If unknown configuration keys are present.
        """
        upd = other.as_dict()
        invalid = set(upd.keys()) - set(self._defaults.keys())
        if invalid:
            msg = f"Invalid kernel configuration parameters: {sorted(invalid)}"
            raise InvalidConfiguration(msg)
        self._values.update(upd)
