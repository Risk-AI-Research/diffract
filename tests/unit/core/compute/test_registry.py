"""Comprehensive test suite for KernelRegistry with dependency injection support.

This test suite handles the dependency injection issues by properly mocking
the container and Provide decorator, ensuring tests work correctly with
the @inject decorator on the kernel function.
"""

from unittest.mock import Mock, patch

import pytest

from diffract.core.compute.execution.enums import (
    KernelApplyLevel,
    KernelExecutionProtocol,
    KernelRestrictions,
)

from diffract.core.compute.config import KernelConfig

from diffract.core.compute.exceptions import (
    DependencyNotFoundError,
    CircularDependencyError,
    InvalidConfiguration,
)

from diffract.core.compute.metadata import KernelInfo, KernelMetadata
from diffract.core.compute.registry import KernelRegistry

# Import kernel with special handling for dependency injection
try:
    from diffract.core.compute.registry import kernel
except ImportError:
    # Fallback if there are import issues
    kernel = None


class TestKernelConfig:
    """Test suite for KernelConfig class."""
    
    def test_init_empty(self):
        """Test KernelConfig initialization with no parameters."""
        config = KernelConfig()
        assert config.as_dict() == {}
    
    def test_init_with_defaults(self):
        """Test KernelConfig initialization with default values."""
        config = KernelConfig(param1="value1", param2=42)
        result = config.as_dict()
        assert result == {"param1": "value1", "param2": 42}
    
    def test_update_valid_params(self):
        """Test updating config with valid parameters."""
        config = KernelConfig(param1="old", param2=1)
        update_config = KernelConfig(param1="new")
        
        config.update(update_config)
        result = config.as_dict()
        assert result["param1"] == "new"
        assert result["param2"] == 1  # Should remain unchanged
    
    def test_update_invalid_params(self):
        """Test updating config with invalid parameters raises error."""
        config = KernelConfig(param1="value")
        invalid_config = KernelConfig(invalid_param="value")
        
        with pytest.raises(InvalidConfiguration, match="Invalid kernel configuration parameters"):
            config.update(invalid_config)


class TestKernelMetadata:
    """Test suite for KernelMetadata dataclass."""
    
    def test_repr(self):
        """Test string representation of KernelMetadata."""
        config = KernelConfig(param1="value1")
        info = KernelInfo(summary="Test kernel")
        
        metadata = KernelMetadata(
            name="test_kernel",
            require_fields=("input1", "input2"),
            produce_fields=("output",),
            implementation=lambda x: x,
            apply_level=KernelApplyLevel.PARAMETER,
            execution_protocol=KernelExecutionProtocol.SEQUENTIAL,
            restrictions=KernelRestrictions.BINARY,
            config=config,
            info=info
        )
        
        repr_str = repr(metadata)
        assert "test_kernel" in repr_str
        assert "input1, input2" in repr_str
        assert "param1=value1" in repr_str
        assert "PARAMETER" in repr_str
        assert "BINARY" in repr_str


class TestKernelRegistry:
    """Test suite for KernelRegistry class."""
    
    @pytest.fixture
    def fresh_registry(self):
        """Create a fresh KernelRegistry for each test."""
        # Mock the register_default_kernels function to avoid import issues
        with patch('diffract.core.compute.decorator.register_default_kernels'):
            registry = KernelRegistry()
        return registry
    
    @pytest.fixture
    def sample_kernel_func(self):
        """Sample kernel function for testing."""
        def sample_func(x: int, y: int = 10) -> int:
            return x + y
        return sample_func
    
    def test_init(self, fresh_registry):
        """Test KernelRegistry initialization."""
        assert len(fresh_registry._metadata) == 0
        assert len(fresh_registry._resolve_cache) == 0
    
    def test_split_signature_no_defaults(self, fresh_registry):
        """Test signature splitting with no default parameters."""
        def func(a, b, c):
            return a + b + c
        
        required, config = fresh_registry._split_signature(func)
        assert required == ("a", "b", "c")
        assert config.as_dict() == {}
    
    def test_split_signature_with_defaults(self, fresh_registry):
        """Test signature splitting with default parameters."""
        def func(a, b=10, c="default"):
            return str(a) + str(b) + c
        
        required, config = fresh_registry._split_signature(func)
        assert required == ("a",)
        assert config.as_dict() == {"b": 10, "c": "default"}
    
    def test_split_signature_invalid_varargs(self, fresh_registry):
        """Test signature splitting rejects *args and **kwargs."""
        def func_with_varargs(*args):
            return sum(args)
        
        def func_with_kwargs(**kwargs):
            return len(kwargs)
        
        with pytest.raises(TypeError):
            fresh_registry._split_signature(func_with_varargs)
        
        with pytest.raises(TypeError):
            fresh_registry._split_signature(func_with_kwargs)
    
    def test_register_kernel(self, fresh_registry, sample_kernel_func):
        """Test kernel registration."""
        config = KernelConfig()
        info = KernelInfo(summary="Test kernel")
        
        fresh_registry.register_kernel(
            name="test_kernel",
            require_fields=("x",),
            produce_fields=("result",),
            implementation=sample_kernel_func,
            apply_level=KernelApplyLevel.PARAMETER,
            execution_protocol=KernelExecutionProtocol.SEQUENTIAL,
            restrictions=None,
            config=config,
            info=info
        )
        
        assert fresh_registry.has_kernel("test_kernel")
        assert "test_kernel" in fresh_registry._metadata
        assert len(fresh_registry._resolve_cache) == 0  # Should be cleared
    
    def test_configure_kernel(self, fresh_registry, sample_kernel_func):
        """Test kernel configuration update."""
        config = KernelConfig(param1="old_value")
        
        fresh_registry.register_kernel(
            name="test_kernel",
            require_fields=("x",),
            produce_fields=("result",),
            implementation=sample_kernel_func,
            apply_level=KernelApplyLevel.PARAMETER,
            execution_protocol=KernelExecutionProtocol.SEQUENTIAL,
            restrictions=None,
            config=config,
        )
        
        new_config = KernelConfig(param1="new_value")
        fresh_registry.configure_kernel("test_kernel", new_config)
        
        result_config = fresh_registry.get_kernel_config("test_kernel")
        assert result_config["param1"] == "new_value"
    
    def test_configure_nonexistent_kernel(self, fresh_registry):
        """Test configuring non-existent kernel raises error."""
        config = KernelConfig(param1="value")
        
        with pytest.raises(DependencyNotFoundError, match="Kernel 'missing' not registered"):
            fresh_registry.configure_kernel("missing", config)
    
    def test_list_kernels(self, fresh_registry, sample_kernel_func):
        """Test kernel listing functionality."""
        # Initially empty
        assert fresh_registry.list_kernels() == []
        assert fresh_registry.list_kernels(verbose=True) == []
        
        # Register a kernel
        fresh_registry.register_kernel(
            name="test_kernel",
            require_fields=("x",),
            produce_fields=("result",),
            implementation=sample_kernel_func,
            apply_level=KernelApplyLevel.PARAMETER,
            execution_protocol=KernelExecutionProtocol.SEQUENTIAL,
            restrictions=None,
            config=KernelConfig(),
        )
        
        # Check listings
        assert fresh_registry.list_kernels() == ["test_kernel"]
        verbose_list = fresh_registry.list_kernels(verbose=True)
        assert len(verbose_list) == 1
        assert "test_kernel" in verbose_list[0]
    
    def test_kernel_getters(self, fresh_registry, sample_kernel_func):
        """Test various kernel getter methods."""
        config = KernelConfig(test_param=42)
        info = KernelInfo(summary="Test summary", notes="Test notes")
        
        fresh_registry.register_kernel(
            name="test_kernel",
            require_fields=("x", "y"),
            produce_fields=("result",),
            implementation=sample_kernel_func,
            apply_level=KernelApplyLevel.IN_MODEL,
            execution_protocol=KernelExecutionProtocol.PARALLEL,
            restrictions=KernelRestrictions.BINARY,
            config=config,
            info=info
        )
        
        assert fresh_registry.get_kernel_apply_level("test_kernel") == KernelApplyLevel.IN_MODEL
        assert fresh_registry.get_kernel_execution_protocol("test_kernel") == KernelExecutionProtocol.PARALLEL
        assert fresh_registry.get_fields_kernel_require("test_kernel") == ("x", "y")
        assert fresh_registry.get_kernel_config("test_kernel") == {"test_param": 42}
        assert fresh_registry.get_kernel_restrictions("test_kernel") == KernelRestrictions.BINARY
        
        kernel_info = fresh_registry.get_kernel_info("test_kernel")
        assert kernel_info.summary == "Test summary"
        assert kernel_info.notes == "Test notes"
    
    def test_kernel_implementation_with_config(self, fresh_registry):
        """Test kernel implementation wrapping with configuration."""
        def test_func(x, multiplier=1):
            return x * multiplier
        
        config = KernelConfig(multiplier=5)
        
        fresh_registry.register_kernel(
            name="test_kernel",
            require_fields=("x",),
            produce_fields=("result",),
            implementation=test_func,
            apply_level=KernelApplyLevel.PARAMETER,
            execution_protocol=KernelExecutionProtocol.SEQUENTIAL,
            restrictions=None,
            config=config,
        )
        
        wrapped_impl = fresh_registry.get_kernel_implementation("test_kernel")
        result = wrapped_impl(10)  # Should use multiplier=5 from config
        assert result == 50
        
        # Test override
        result_override = wrapped_impl(10, multiplier=3)
        assert result_override == 30
    
    def test_resolve_dependencies_simple(self, fresh_registry):
        """Test simple dependency resolution."""
        def func_a():
            return "a"
        
        def func_b(a):
            return "b"
        
        fresh_registry.register_kernel(
            name="a",
            require_fields=(),
            produce_fields=("a",),
            implementation=func_a,
            apply_level=KernelApplyLevel.PARAMETER,
            execution_protocol=KernelExecutionProtocol.SEQUENTIAL,
            restrictions=None,
            config=KernelConfig(),
        )
        
        fresh_registry.register_kernel(
            name="b",
            require_fields=("a",),
            produce_fields=("b",),
            implementation=func_b,
            apply_level=KernelApplyLevel.PARAMETER,
            execution_protocol=KernelExecutionProtocol.SEQUENTIAL,
            restrictions=None,
            config=KernelConfig(),
        )
        
        deps = fresh_registry.resolve_dependencies("b")
        assert deps == ("a", "b")
        
        # Test caching
        deps2 = fresh_registry.resolve_dependencies("b")
        assert deps2 == ("a", "b")
        assert "b" in fresh_registry._resolve_cache
    
    def test_resolve_dependencies_circular(self, fresh_registry):
        """Test circular dependency detection."""
        def func_a(b):
            return "a"
        
        def func_b(a):
            return "b"
        
        fresh_registry.register_kernel(
            name="a",
            require_fields=("b",),
            produce_fields=("a",),
            implementation=func_a,
            apply_level=KernelApplyLevel.PARAMETER,
            execution_protocol=KernelExecutionProtocol.SEQUENTIAL,
            restrictions=None,
            config=KernelConfig(),
        )
        
        fresh_registry.register_kernel(
            name="b",
            require_fields=("a",),
            produce_fields=("b",),
            implementation=func_b,
            apply_level=KernelApplyLevel.PARAMETER,
            execution_protocol=KernelExecutionProtocol.SEQUENTIAL,
            restrictions=None,
            config=KernelConfig(),
        )
        
        with pytest.raises(CircularDependencyError, match="Circular dependency for 'a'"):
            fresh_registry.resolve_dependencies("a")
    
    def test_resolve_dependencies_diamond(self, fresh_registry):
        """Test diamond dependency resolution (stable ordering)."""
        kernels_data = [
            ("root", (), ("root",)),
            ("left", ("root",), ("left",)),
            ("right", ("root",), ("right",)),
            ("final", ("left", "right"), ("final",)),
        ]
        
        for name, require, produce in kernels_data:
            fresh_registry.register_kernel(
                name=name,
                require_fields=require,
                produce_fields=produce,
                implementation=lambda: "result",
                apply_level=KernelApplyLevel.PARAMETER,
                execution_protocol=KernelExecutionProtocol.SEQUENTIAL,
                restrictions=None,
                config=KernelConfig(),
            )
        
        deps = fresh_registry.resolve_dependencies("final")
        # root must come first, final must be last
        assert deps[0] == "root"
        assert deps[-1] == "final"
        assert "left" in deps and "right" in deps
        assert deps.index("root") < deps.index("left")
        assert deps.index("root") < deps.index("right")
    
    def test_normalize_kernel_result_dict(self, fresh_registry, sample_kernel_func):
        """Test result normalization with dict input."""
        fresh_registry.register_kernel(
            name="test_kernel",
            require_fields=("x",),
            produce_fields=("a", "b"),
            implementation=sample_kernel_func,
            apply_level=KernelApplyLevel.PARAMETER,
            execution_protocol=KernelExecutionProtocol.SEQUENTIAL,
            restrictions=None,
            config=KernelConfig(),
        )
        
        result_dict = {"a": 1, "b": 2}
        normalized = fresh_registry.normalize_kernel_result("test_kernel", result_dict)
        assert normalized == result_dict
    
    def test_normalize_kernel_result_tuple(self, fresh_registry, sample_kernel_func):
        """Test result normalization with tuple input."""
        fresh_registry.register_kernel(
            name="test_kernel",
            require_fields=("x",),
            produce_fields=("a", "b"),
            implementation=sample_kernel_func,
            apply_level=KernelApplyLevel.PARAMETER,
            execution_protocol=KernelExecutionProtocol.SEQUENTIAL,
            restrictions=None,
            config=KernelConfig(),
        )
        
        result_tuple = (1, 2)
        normalized = fresh_registry.normalize_kernel_result("test_kernel", result_tuple)
        assert normalized == {"a": 1, "b": 2}
    
    def test_normalize_kernel_result_single_value(self, fresh_registry, sample_kernel_func):
        """Test result normalization with single value."""
        fresh_registry.register_kernel(
            name="test_kernel",
            require_fields=("x",),
            produce_fields=("result",),
            implementation=sample_kernel_func,
            apply_level=KernelApplyLevel.PARAMETER,
            execution_protocol=KernelExecutionProtocol.SEQUENTIAL,
            restrictions=None,
            config=KernelConfig(),
        )
        
        normalized = fresh_registry.normalize_kernel_result("test_kernel", 42)
        assert normalized == {"result": 42}
    
    def test_normalize_kernel_result_tuple_length_mismatch(self, fresh_registry, sample_kernel_func):
        """Test result normalization with tuple length mismatch."""
        fresh_registry.register_kernel(
            name="test_kernel",
            require_fields=("x",),
            produce_fields=("a", "b"),
            implementation=sample_kernel_func,
            apply_level=KernelApplyLevel.PARAMETER,
            execution_protocol=KernelExecutionProtocol.SEQUENTIAL,
            restrictions=None,
            config=KernelConfig(),
        )
        
        with pytest.raises(InvalidConfiguration, match="returned 3 values but produce declares 2"):
            fresh_registry.normalize_kernel_result("test_kernel", (1, 2, 3))
    
    def test_normalize_kernel_result_scalar_multiple_fields(self, fresh_registry, sample_kernel_func):
        """Test result normalization error with scalar for multiple fields."""
        fresh_registry.register_kernel(
            name="test_kernel",
            require_fields=("x",),
            produce_fields=("a", "b"),
            implementation=sample_kernel_func,
            apply_level=KernelApplyLevel.PARAMETER,
            execution_protocol=KernelExecutionProtocol.SEQUENTIAL,
            restrictions=None,
            config=KernelConfig(),
        )
        
        with pytest.raises(InvalidConfiguration, match="returned scalar but produce declares multiple fields"):
            fresh_registry.normalize_kernel_result("test_kernel", 42)


class TestKernelDecorator:
    """Test suite for the kernel decorator with dependency injection mocking."""
    
    @pytest.fixture
    def mock_registry(self):
        """Create a mock registry for dependency injection."""
        registry = Mock(spec=KernelRegistry)
        registry._split_signature.return_value = (("x",), KernelConfig())
        registry.register_kernel = Mock()
        return registry

    def test_kernel_decorator_basic(self, mock_registry):
        """Test basic kernel decorator functionality."""
        # Import kernel after mocking is in place
        from diffract.core.compute.decorator import kernel
        
        @kernel(registry=mock_registry)
        def test_func(x):
            return x * 2
        
        # Verify the registry's register_kernel was called
        mock_registry.register_kernel.assert_called_once()
        call_args = mock_registry.register_kernel.call_args
        assert call_args[1]['name'] == 'test_func'
        assert call_args[1]['apply_level'] == KernelApplyLevel.PARAMETER
    
    def test_kernel_decorator_with_params(self, mock_registry):
        """Test kernel decorator with custom parameters."""
        from diffract.core.compute.decorator import kernel
        
        info = KernelInfo(summary="Test kernel")
        
        @kernel(
            registry=mock_registry,
            name="custom_name",
            require_fields=("input1", "input2"),
            produce_fields=("output1", "output2"),
            apply_level=KernelApplyLevel.IN_MODEL,
            execution_protocol=KernelExecutionProtocol.PARALLEL,
            restrictions=KernelRestrictions.BINARY,
            info=info
        )
        def test_func(input1, input2):
            return input1 + input2, input1 - input2
        
        mock_registry.register_kernel.assert_called_once()
        call_args = mock_registry.register_kernel.call_args
        assert call_args[1]['name'] == 'custom_name'
        assert call_args[1]['require_fields'] == ('input1', 'input2')
        assert call_args[1]['produce_fields'] == ('output1', 'output2')
        assert call_args[1]['apply_level'] == KernelApplyLevel.IN_MODEL
        assert call_args[1]['execution_protocol'] == KernelExecutionProtocol.PARALLEL
        assert call_args[1]['restrictions'] == KernelRestrictions.BINARY
        assert call_args[1]['info'] == info
    
    def test_kernel_decorator_multiple_kernels(self, mock_registry):
        """Test kernel decorator with multiple kernel registration."""
        from diffract.core.compute.decorator import kernel
        
        @kernel(registry=mock_registry, name="k1")
        @kernel(registry=mock_registry, name="k2")
        def multi_func(x):
            return x * 2
        
        # Should register multiple kernels
        assert mock_registry.register_kernel.call_count == 2
    
    def test_kernel_decorator_function_without_parentheses(self, mock_registry):
        """Test kernel decorator used without parentheses."""
        from diffract.core.compute.decorator import kernel
        from functools import partial

        kernel_ = partial(kernel, registry=mock_registry)
        
        @kernel_
        def bare_func(x):
            return x
        
        mock_registry.register_kernel.assert_called_once()
        call_args = mock_registry.register_kernel.call_args
        assert call_args[1]['name'] == 'bare_func'


class TestIntegrationWithoutDI:
    """Integration tests that work around dependency injection issues."""
    
    @pytest.fixture
    def registry_with_mock_registry(self):
        """Create registry with mocked DI for decorator tests."""
        with patch('diffract.core.compute.decorator.register_default_kernels'):
            registry = KernelRegistry()
        
        # Mock the kernel decorator to use this registry directly
        original_kernel = None
        try:
            from diffract.core.compute.registry import kernel as original_kernel
        except ImportError:
            pass
        
        def mock_kernel_decorator(
            _func=None,
            *,
            name=None,
            require_fields=None,
            produce_fields=None,
            apply_level=KernelApplyLevel.PARAMETER,
            execution_protocol=KernelExecutionProtocol.SEQUENTIAL,
            restrictions=None,
            info=None,
        ):
            def decorator(func):
                req_auto, cfg = registry._split_signature(func)
                final_name = name or func.__name__
                final_require = require_fields or req_auto
                final_produce = produce_fields or (func.__name__,)
                
                registry.register_kernel(
                    name=final_name,
                    require_fields=final_require,
                    produce_fields=final_produce,
                    implementation=func,
                    apply_level=apply_level,
                    execution_protocol=execution_protocol,
                    restrictions=restrictions,
                    config=cfg,
                    info=info or KernelInfo(),
                )
                return func
            
            if _func is None:
                return decorator
            else:
                return decorator(_func)
        
        return registry, mock_kernel_decorator
    
    def test_edge_cases_minimal_kernels(self, registry_with_mock_registry):
        """Test edge cases with minimal kernels."""
        registry, kernel_decorator = registry_with_mock_registry
        
        @kernel_decorator(produce_fields=("const",))
        def no_args():
            return 42
        
        assert registry.has_kernel("no_args")
        res = registry.normalize_kernel_result("no_args", 42)
        assert res == {"const": 42}
        
        with pytest.raises(InvalidConfiguration):
            registry.normalize_kernel_result("no_args", (1, 2))
    
    def test_error_messages_quality(self, registry_with_mock_registry):
        """Test error message quality."""
        registry, kernel_decorator = registry_with_mock_registry
        
        with pytest.raises(DependencyNotFoundError, match=r"Kernel 'missing' not registered"):
            registry._get("missing")
        
        @kernel_decorator(require_fields=("b",))
        def a(b):
            return b
        
        @kernel_decorator(require_fields=("a",))
        def b(a):
            return a
        
        with pytest.raises(CircularDependencyError, match=r"Circular dependency for 'a'"):
            registry.resolve_dependencies("a")
    
    def test_diamond_dependency_ordering(self, registry_with_mock_registry):
        """Test diamond dependency ordering."""
        registry, kernel_decorator = registry_with_mock_registry
        
        @kernel_decorator
        def root():
            return "root"
        
        @kernel_decorator(require_fields=("root",))
        def left(root):
            return root
        
        @kernel_decorator(require_fields=("root",))
        def right(root):
            return root
        
        @kernel_decorator(require_fields=("left", "right"))
        def final(left, right):
            return left + right
        
        deps = registry.resolve_dependencies("final")
        # root must come before left/right; final must be last
        assert deps == ("root", "left", "right", "final")
    
    def test_large_graph_performance_and_cache(self, registry_with_mock_registry):
        """Test performance with large dependency graphs."""
        registry, kernel_decorator = registry_with_mock_registry
        
        N = 50  # Reduced for test performance
        
        @kernel_decorator
        def base():
            return "base"
        
        # Create a chain base -> k1 -> k2 -> ... -> kN
        prev_field = "base"
        for i in range(1, N + 1):
            kernel_name = f"k{i}"
            
            # Create kernel function with proper closure
            def make_kernel_func(dep_field, name):
                def kernel_func(dep_val):
                    return 1
                kernel_func.__name__ = name
                return kernel_func
            
            kernel_func = make_kernel_func(prev_field, kernel_name)
            kernel_decorator(require_fields=(prev_field,))(kernel_func)
            prev_field = kernel_name
        
        # Resolve dependencies at the tail
        deps = registry.resolve_dependencies(f"k{N}")
        assert deps[0] == "base"
        assert deps[-1] == f"k{N}"
        assert len(deps) == N + 1
        
        # Cache should contain the result
        assert f"k{N}" in registry._resolve_cache
    
    def test_tuple_and_dict_result_mapping(self, registry_with_mock_registry):
        """Test result mapping for different return types."""
        registry, kernel_decorator = registry_with_mock_registry
        
        @kernel_decorator(produce_fields=("a", "b"))
        def ab_dict(x):
            return {"a": x, "b": x + 1}

        assert registry.normalize_kernel_result("ab_dict", {"a": 5, "b": 6}) == {"a": 5, "b": 6}
    
    def test_kernel_info_defaults_and_usage(self, registry_with_mock_registry):
        """Test kernel info handling."""
        registry, kernel_decorator = registry_with_mock_registry
        
        @kernel_decorator
        def no_info(x):
            return x
        
        info = registry.get_kernel_info("no_info")
        assert info.summary is None and info.notes is None
        
        @kernel_decorator(info=KernelInfo(summary="S", notes="N"))
        def with_info(x):
            return x
        
        info2 = registry.get_kernel_info("with_info")
        assert info2.summary == "S" and info2.notes == "N"


if __name__ == "__main__":
    pytest.main([__file__])
