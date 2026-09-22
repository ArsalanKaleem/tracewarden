"""TraceWarden: pre-dispatch detection and localization of indirect prompt injection in agent tool calls."""
from .schema import LABELS, Step, Trajectory

__version__ = "0.1.0.dev0"
__all__ = ["LABELS", "Step", "Trajectory", "Guard", "Session", "Decision", "__version__"]


def __getattr__(name):  # lazy import so `import tracewarden` stays light
    if name in ("Guard", "Session", "Decision"):
        from . import guard

        return getattr(guard, name)
    raise AttributeError(name)
