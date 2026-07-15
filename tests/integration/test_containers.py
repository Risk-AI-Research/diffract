"""Improved integration tests for main application containers."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import pytest

from diffract.containers import (
    MainContainer,
    WiringConfiguration,
    create_main_container,
)
from diffract.core.cache.interface import ICacheManager
from diffract.core.compute.exceptions import InconsistentWiring
from diffract.core.storage.interface import IStorageManager

pytestmark = pytest.mark.integration


logger = logging.getLogger(__name__)


class TestMainContainer:
    """Integration tests for MainContainer with deeper checks."""

    def test_container_creation_without_config(self) -> None:
        """Container initializes and exposes subsystems with defaults."""
        container = MainContainer()

        assert container is not None
        assert hasattr(container, "config")
        assert hasattr(container, "storage")
        assert hasattr(container, "cache")
        assert hasattr(container, "compute")
        assert hasattr(container, "nn")

        # Subcontainers should be instantiable
        assert container.storage() is not None
        assert container.cache() is not None
        assert container.compute() is not None
        assert container.nn() is not None

    def test_dependency_injection_between_containers(
        self, temp_config_file: Path
    ) -> None:
        """ModelParametersContainer receives actual callable dependencies."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "test_storage.h5"

            container = MainContainer()
            container.config.from_yaml(str(temp_config_file))
            # Override storage path to a safe temp location
            container.config.storage.hdf5.path.override(str(storage_path))

            mp_container = container.nn()

            storage_provider = mp_container.storage_manager
            cache_provider = mp_container.cache_manager

            assert storage_provider is not None
            assert cache_provider is not None

            storage = storage_provider()
            cache = cache_provider()

            # Instances should comply with protocols
            assert isinstance(storage, IStorageManager)
            assert isinstance(cache, ICacheManager)

    def test_wiring(self, temp_config_file: Path) -> None:
        container = MainContainer()
        container.config.from_yaml(str(temp_config_file))

        failed = False
        try:
            container.compute_singleton.register_default_kernels()
        except InconsistentWiring:
            failed = True

        assert failed

        WiringConfiguration.wire(container)

        container.compute_singleton.register_default_kernels()
        registry = container.compute_singleton.kernel_registry()

        assert len(registry.list_kernels()) > 0

    def test_container_creation_with_config(self, temp_config_file: Path) -> None:
        """Configuration file is loaded and applied to providers."""
        container = create_main_container(temp_config_file)

        assert container is not None
        # Access configuration values through providers
        assert container.config.storage.backend() == "hdf5"
        assert container.config.cache.backend() == "simple"
        # Compute defaults should resolve from config file
        assert container.config.compute.max_workers() == 2

    def test_container_creation_with_ini_config(
        self, temp_ini_config_file: Path
    ) -> None:
        """INI configuration file is loaded and applied to providers."""
        container = create_main_container(temp_ini_config_file)

        assert container is not None
        assert container.config.storage.backend() == "hdf5"
        assert container.config.storage.hdf5.path().endswith("test_storage.h5")
        assert container.config.cache.backend() == "simple"
        assert container.config.cache.simple.max_memory_mb() == 64
        assert container.config.compute.executor.max_workers() == 2
        assert container.config.nn.extractor.skip_not_implemented_types() is True

    def test_container_creation_with_nonexistent_config(self) -> None:
        """Non-existent config should not break container creation."""
        nonexistent_path = Path("/definitely/not/exist/config.yaml")
        container = create_main_container(nonexistent_path)
        assert container is not None
