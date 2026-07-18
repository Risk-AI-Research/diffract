"""Main application dependency injection container.

This module provides the MainContainer class that orchestrates all subsystem
containers for the diffract application, including storage, cache, compute,
model parameters, and export functionality.
"""

from __future__ import annotations

import configparser
import contextlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from dependency_injector import containers, providers

from .core.cache.containers import CacheContainer
from .core.compute.containers import (
    ComputeContainer,
    ComputeSingletonContainer,
    ComputeSingletonContainerWiringConfig,
)
from .core.data.metadata.containers import MetadataContainer
from .core.data.metadata.sqlite_index import IN_MEMORY_DATABASE
from .core.data.nn.containers import ModelParametersContainer
from .core.export.containers import ExportContainer
from .core.parallel import ParallelSingletonContainer
from .core.storage.containers import StorageContainer
from .core.utils.exceptions import format_exception_message

logger = logging.getLogger(__name__)

# Built-in configuration profiles for common use cases.
#
# Values are relative paths to INI configs shipped with the repo/package.
PROFILES: dict[str, str] = {
    "ram": "configs/fast_speed_without_disk.ini",
    "local": "configs/sqlite.ini",
    "hybrid": "configs/hybrid.ini",
}

# Backend sentinels that occupy a "path" key without naming a file. They are
# carried verbatim so a backend can recognize them by equality.
_NON_FILESYSTEM_PATH_VALUES: frozenset[str] = frozenset({IN_MEMORY_DATABASE})


def list_profiles() -> list[str]:
    """Return available profile names."""
    return list(PROFILES.keys())


def _resolve_packaged_or_repo_path(relative_path: str) -> Path:
    """Resolve a config path within package or repository layout."""
    candidates = [
        # Packaged layout: <site-packages>/diffract/<relative_path>
        Path(__file__).resolve().parent / relative_path,
        # Repo layout: <root>/<relative_path>
        Path(__file__).resolve().parents[2] / relative_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    msg = (
        f"Config file not found for relative path {relative_path!r}. "
        f"Tried: {', '.join(str(c) for c in candidates)}"
    )
    raise FileNotFoundError(msg)


def get_profile_config(profile: str) -> dict[str, Any]:
    """Get configuration dictionary for a profile."""
    if profile not in PROFILES:
        available = ", ".join(PROFILES.keys())
        msg = f"Unknown profile '{profile}'. Available: {available}"
        raise ValueError(msg)
    config_path = _resolve_packaged_or_repo_path(PROFILES[profile])
    return _parse_ini_config(config_path)


_FALLBACK_LOGGING_CONFIG: dict[str, Any] = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {"format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"}
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "standard",
            "stream": "ext://sys.stdout",
        }
    },
    "root": {"level": "INFO", "handlers": ["console"]},
}


def _expand_env_vars(value: str) -> str:
    """Expand ${VAR} and $VAR patterns with environment variables."""

    def replacer(match: re.Match[str]) -> str:
        var_name = match.group(1) or match.group(2)
        return os.environ.get(var_name, match.group(0))

    return re.sub(r"\$\{([^}]+)\}|\$([A-Za-z_][A-Za-z0-9_]*)", replacer, value)


def _coerce_ini_value(value: str) -> Any:
    """Convert an INI string value into a reasonable Python type.

    INI files store values as strings; for service initialization configs we
    accept basic scalar types and JSON literals for lists/dicts.
    Supports ${VAR} syntax for environment variable expansion.
    """
    raw = _expand_env_vars(value.strip())
    lowered = raw.lower()

    if lowered in {"none", "null"}:
        return None
    if lowered in {"true", "false"}:
        return lowered == "true"

    # Try JSON first for explicit literals like lists/dicts/quoted strings.
    if raw.startswith(("{", "[", '"')) or raw in {"0", "1"}:
        with contextlib.suppress(json.JSONDecodeError):
            return json.loads(raw)

    # Numeric scalars (allow underscores).
    int_candidate = raw.replace("_", "")
    with contextlib.suppress(ValueError):
        return int(int_candidate)

    try:
        return float(int_candidate)
    except ValueError:
        return raw


def _parse_ini_config(path: Path) -> dict[str, Any]:
    """Parse an INI file into a nested dictionary.

    Section names can use dot-notation to represent nesting, e.g.:
      [storage.hdf5]
      path=/tmp/main.h5
    """
    # Disable interpolation because logging formatters use "%(name)s" syntax.
    parser = configparser.ConfigParser(interpolation=None)
    # Preserve case for keys like "maxBytes" used by logging.config.dictConfig.
    parser.optionxform = str
    parser.read(path)

    data: dict[str, Any] = {}
    for section in parser.sections():
        cursor: dict[str, Any] = data
        for part in section.split("."):
            cursor = cursor.setdefault(part, {})
        for key, val in parser.items(section):
            cursor[key] = _coerce_ini_value(val)
    return data


def _default_config_path() -> Path | None:
    """Return the default launch config path if present."""
    with contextlib.suppress(FileNotFoundError):
        return _resolve_packaged_or_repo_path(PROFILES["ram"])
    return None


class MainContainer(containers.DeclarativeContainer):
    """Main application container that orchestrates all subsystem containers.

    This container manages the dependency injection for the entire application,
    providing centralized configuration and wiring of all subsystem components
    including storage, caching, compute kernels, model parameters, and export.
    """

    # Configuration
    config = providers.Configuration()

    # Callable, not Resource: logging is configured once at build and must stay
    # out of the init_resources()/shutdown_resources() lifecycle, which would
    # otherwise re-run dictConfig on every context entry and reset user handlers.
    logging_config = providers.Callable(
        "logging.config.dictConfig",
        config=config.logging,
    )

    # Subsystem containers
    storage = providers.Container(
        StorageContainer,
        config=config.storage,
    )

    cache = providers.Container(
        CacheContainer,
        config=config.cache,
    )

    metadata = providers.Container(
        MetadataContainer,
        config=config.metadata,
    )

    compute_singleton = providers.Container(ComputeSingletonContainer)

    parallel_singleton = providers.Container(
        ParallelSingletonContainer,
        config=config.parallel,
    )

    nn = providers.Container(
        ModelParametersContainer,
        storage_manager=storage.storage_manager_resource,
        cache_manager=cache.cache_manager_resource,
        metadata_index=metadata.metadata_index_resource,
        parallel=parallel_singleton.thread_pool_context,
        config=config.nn,
    )

    compute = providers.Container(
        ComputeContainer,
        config=config.compute,
        compute_singleton=compute_singleton,
        parallel=parallel_singleton.thread_pool_context,
        process_pool=parallel_singleton.process_pool_context,
        aggregate_repository=nn.aggregate_repository,
    )

    export = providers.Container(
        ExportContainer,
        config=config.export,
    )


class WiringConfiguration:
    """Complete application wiring configuration.

    Manages the dependency injection wiring for all containers in the application,
    ensuring proper module and package registration for dependency resolution.
    """

    @classmethod
    def get_wiring_configs(cls) -> list[dict[str, Any]]:
        """Get all wiring configurations for the application.

        Returns:
            List of dictionaries containing container, modules, and packages
            configuration for each subsystem that requires wiring.
        """
        return [
            {
                "container": ComputeSingletonContainer,
                "modules": ComputeSingletonContainerWiringConfig.modules,
                "packages": ComputeSingletonContainerWiringConfig.packages,
            },
        ]

    @classmethod
    def wire(cls, container: MainContainer) -> None:
        """Wire the entire application with all subsystem containers.

        Args:
            container: The main application container to wire.

        Raises:
            Exception: If wiring fails for any container or module.
        """
        try:
            # Wire main container
            container.wire(
                modules=[
                    __name__,
                    "diffract.session",
                    "diffract.session.namespaces",
                    "diffract.session.namespaces.models",
                ],
            )

            # Wire subsystem containers
            for config in cls.get_wiring_configs():
                container_cls = config["container"]
                parts = re.findall(r"[A-Z][^A-Z]*", container_cls.__name__)
                parts = [p.lower() for p in parts]
                parts.remove("container")
                container_name = "_".join(parts)
                container_instance = getattr(container, container_name)

                if config.get("modules"):
                    container_instance.wire(modules=config["modules"])

                if config.get("packages"):
                    container_instance.wire(packages=config["packages"])

            logger.debug("Successfully wired all application containers")

        except Exception:
            logger.exception("Failed to wire application containers")
            raise


compute_singleton_container = ComputeSingletonContainer()


def create_main_container(
    config_path: str | Path | None = None,
    profile: str | None = None,
) -> MainContainer:
    """Create and configure the main application container.

    Args:
        config_path: Path to configuration file (YAML/JSON/INI). Overrides
            profile settings if both are provided.
        profile: Built-in profile name ("ram", "local", "hybrid").

    Returns:
        Configured MainContainer instance with all subsystems wired
        and default kernels registered.

    Raises:
        ValueError: If profile name is not recognized.
        Exception: If container creation or wiring fails.
    """
    container = MainContainer(
        compute_singleton=compute_singleton_container,
    )

    repo_root = Path(__file__).resolve().parents[2]
    cwd = Path.cwd()

    config_loaded = False

    if config_path is not None:
        config_path = Path(config_path)
        if config_path.exists():
            suffix = config_path.suffix.lower()
            match suffix:
                case ".yaml" | ".yml":
                    container.config.from_yaml(str(config_path))
                case ".json":
                    container.config.from_json(str(config_path))
                case ".ini":
                    container.config.from_dict(_parse_ini_config(config_path))
                case _:
                    msg = f"Unsupported config file extension: '{suffix}'"
                    raise ValueError(msg)
            logger.info("Using config file: %s", config_path)
            config_loaded = True
        else:
            logger.warning("Configuration file not found: %s", config_path)

    if not config_loaded and profile is not None:
        profile_config = get_profile_config(profile)
        container.config.from_dict(profile_config)
        logger.info("Using profile: %s", profile)
        config_loaded = True

    if not config_loaded:
        default_path = _default_config_path()
        if default_path is not None:
            container.config.from_dict(_parse_ini_config(default_path))
            logger.debug("Loaded default configuration from %s", default_path)

    # Resolve relative file paths to absolute paths.
    base_path = cwd if profile is not None else repo_root
    resolved = container.config()
    for key in ("path", "filename"):
        stack: list[dict[str, Any]] = [resolved]
        while stack:
            cur = stack.pop()
            for k, v in list(cur.items()):
                if isinstance(v, dict):
                    stack.append(v)
                    continue
                if k != key or not isinstance(v, str):
                    continue
                value = v.strip().strip('"')
                if value in _NON_FILESYSTEM_PATH_VALUES:
                    continue
                if value.startswith(("ext://", "/", "~")):
                    continue
                cur[k] = str(base_path / value)
    container.config.from_dict(resolved)

    # Configs without a [logging] section get the fallback silently.
    if not resolved.get("logging"):
        container.config.logging.override(_FALLBACK_LOGGING_CONFIG)

    # logging.config.dictConfig cannot create missing log directories itself.
    for handler in resolved.get("logging", {}).get("handlers", {}).values():
        filename = handler.get("filename") if isinstance(handler, dict) else None
        if isinstance(filename, str):
            Path(filename.strip().strip('"')).parent.mkdir(parents=True, exist_ok=True)

    try:
        container.logging_config()
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "Failed to configure logging; using fallback config: %s",
            format_exception_message(e),
        )
        logger.debug("Logging configuration failure details", exc_info=True)
        container.config.logging.override(_FALLBACK_LOGGING_CONFIG)
        container.logging_config()

    # Wire all containers
    WiringConfiguration.wire(container)

    container.compute_singleton.register_default_kernels()

    return container
