from importlib.metadata import PackageNotFoundError, version

from gimkit.contexts import Query, Response
from gimkit.guides import guide
from gimkit.models import GenerationResult, from_openai, from_vllm, from_vllm_offline


try:
    __version__ = version("gimkit")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"


__all__ = [
    "GenerationResult",
    "Query",
    "Response",
    "from_openai",
    "from_vllm",
    "from_vllm_offline",
    "guide",
]
