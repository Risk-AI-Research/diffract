"""Kernel configuration management and validation."""

from __future__ import annotations

from typing import Any

from .exceptions import InvalidConfiguration


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
