from importlib.metadata import PackageNotFoundError, version

from gimkit.guides import guide
from gimkit.models import from_openai, from_vllm, from_vllm_offline


try:
    __version__ = version("gimkit")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"


__all__ = [
    "from_openai",
    "from_vllm",
    "from_vllm_offline",
    "guide",
]
