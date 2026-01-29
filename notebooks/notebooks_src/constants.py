from pathlib import Path


class MetaStaticClass(type):
    def __call__(cls, *args, **kwargs):
        raise TypeError(f"Cannot instantiate static class {cls.__name__}")


class MetaConstant(MetaStaticClass, type):
    def __setattr__(cls, attr, value):
        if hasattr(cls, attr) and not attr.startswith("_"):
            raise AttributeError(f"Changing constant field {cls.__name__}.{attr} is forbidden")

        super().__setattr__(attr, value)

    def __delattr__(cls, attr):
        if hasattr(cls, attr) and not attr.startswith("_"):
            raise AttributeError(f"Deleting constant field {cls.__name__}.{attr} is forbidden")


class DefaultValues(metaclass=MetaConstant):
    """Default values container."""

    MISSING_STR_PLACEHOLDER: str = "null"
    MISSING_INT_PLACEHOLDER: int = -1
    MISSING_FLOAT_PLACEHOLDER: float = 0.0


class Paths(metaclass=MetaConstant):
    ROOT = (Path(__file__).parent / "..").resolve()
    IMAGES_DIR = (Path(__file__).parent / ".." / "images").resolve()
    CONFIG_DIR = (Path(__file__).parent / ".." / "configs").resolve()
