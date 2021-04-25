from typing import Dict, Tuple, Type


def build_types_map(types: Tuple[Type]) -> Dict[str, Type]:
    return {t.__name__: t for t in types}
