"""exptrail: local experiment logging that keeps README numbers verifiable."""

__version__ = "0.1.0"

from .run import DirtyTreeWarning, NoGitWarning, Run, track  # noqa: E402

__all__ = ["Run", "track", "DirtyTreeWarning", "NoGitWarning", "__version__"]
