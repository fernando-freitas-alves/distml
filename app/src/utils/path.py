import os
import pickle
from typing import Any, Dict


def mkdir(path: str) -> None:
    abspath = os.path.abspath(path)
    root, ext = os.path.splitext(abspath)
    if not ext:
        dirname = root
    else:
        dirname = os.path.dirname(root)

    os.makedirs(dirname, exist_ok=True)


def save_object(obj: Any, filepath: str) -> None:
    mkdir(filepath)
    with open(filepath, "wb") as output:
        pickle.dump(obj, output)


def join_paths(folderpath: str, filenames: Dict[str, str]) -> Dict[str, str]:
    filepaths = {}

    for var_name, filename in filenames.items():
        filepaths[var_name] = os.path.join(folderpath, filename)

    return filepaths
