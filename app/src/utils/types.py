from distutils.util import strtobool
from typing import Dict, Optional, Tuple, Type


def build_types_map(*types: Tuple[Type]) -> Dict[str, Type]:
    return {t.__name__: t for t in types}


def str2bool(string: str, default: Optional[bool] = None) -> bool:
    return default if not string and default else bool(strtobool(string))
