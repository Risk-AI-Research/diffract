"""Dependency injection container for compute subsystem.

Provides providers for the kernel registry, default kernel registration,
and the kernel executor resource. Configuration for the executor is taken
from the container configuration under the "executor" key.

Example:
    >>> container = ComputeContainer()
    >>> with container.kernel_executor() as executor:
    ...     executor.execute("field_x", collection)
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import ClassVar

from dependency_injector import containers, providers

from diffract.core.parallel import ParallelContext
from diffract.core.utils.build import build_with_defaults

from .decorator import register_default_kernels
from .execution import KernelExecutor
from .registry import KernelRegistry


class ComputeSingletonContainer(containers.DeclarativeContainer):
    """Singleton providers for compute subsystem."""

    kernel_registry = providers.Singleton(KernelRegistry)

    register_default_kernels = providers.Callable(
        register_default_kernels, kernel_registry
    )


def _config_to_dict(cfg: dict | None) -> dict:
    """Convert config to dict, returning empty dict if None."""
    return cfg if cfg is not None else {}


class ComputeContainer(containers.DeclarativeContainer):
    """Container for compute-related dependencies.

    Provides singleton kernel registry, default kernel registration callable,
    and resource-managed kernel executor with configuration support.
    """

    config = providers.Configuration()

    compute_singleton = providers.Container(ComputeSingletonContainer)
    parallel = providers.Dependency(instance_of=ParallelContext)
    process_pool = providers.Dependency(instance_of=ProcessPoolExecutor)
    aggregate_repository = providers.Dependency()

    kernel_registry = providers.Singleton(compute_singleton.kernel_registry)

    _executor_config = providers.Callable(_config_to_dict, config.executor)

    kernel_executor = providers.Resource(
        build_with_defaults,
        KernelExecutor,
        _executor_config,
        registry=kernel_registry,
        process_pool=process_pool,
        parallel=parallel,
        aggregate_repository=aggregate_repository,
    )


class ComputeSingletonContainerWiringConfig:
    """Wiring configuration for compute module.

    Specifies the container class, modules to wire, and packages for
    dependency injection setup in the compute subsystem.
    """

    container: ClassVar[type[ComputeSingletonContainer]] = ComputeSingletonContainer
    modules: ClassVar[tuple[str, ...]] = ("diffract.core.compute.decorator",)
    packages: ClassVar[tuple[str, ...]] = ()
