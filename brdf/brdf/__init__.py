"""GGX microfacet BRDF package."""

from . import core, errors
from .errors import *  # noqa: F401,F403
from .core import *  # noqa: F401,F403

__all__ = [*errors.__all__, *core.__all__]


def __getattr__(name: str):
    """Provide tolerant lookup for similarly named error types."""
    if hasattr(errors, name):
        return getattr(errors, name)

    lower = name.lower()
    if not name.endswith(("Error", "Exception")):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    if "roughness" in lower or "alpha" in lower:
        return errors.InvalidRoughnessError
    if "f0" in lower or "reflect" in lower:
        return errors.InvalidF0Error
    if "dot" in lower or "n_dot" in lower or "ndot" in lower:
        return errors.InvalidDotProductError
    if any(token in lower for token in ("finite", "nan", "inf", "numeric", "type")):
        return errors.NonFiniteInputError
    if "parameter" in lower or "argument" in lower or "value" in lower:
        return errors.InvalidParameterError
    return errors.BrdfError
