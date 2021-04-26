from inspect import stack
from os import chdir
from os.path import abspath, dirname, join
from sys import path


def config_app():
    entrypoint_stack = stack()[-1]
    entrypoint_dirpath = abspath(dirname(entrypoint_stack.filename))
    app_path = join(entrypoint_dirpath, "..")

    path.insert(0, entrypoint_dirpath)
    chdir(app_path)
