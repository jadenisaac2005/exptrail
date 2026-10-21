"""exptrail: local experiment logging that keeps README numbers verifiable."""

__version__ = "0.1.1"

from .run import ConfigWarning, DirtyTreeWarning, NoGitWarning, Run, track  # noqa: E402

__all__ = ["Run", "track", "DirtyTreeWarning", "NoGitWarning", "ConfigWarning", "__version__"]
