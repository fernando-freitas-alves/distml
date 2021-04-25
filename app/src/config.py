from inspect import stack
from os.path import abspath, dirname
from sys import path

entrypoint_stack = stack()[-1]
entrypoint_dirpath = abspath(dirname(entrypoint_stack.filename))

path.insert(0, entrypoint_dirpath)
