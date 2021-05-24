#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
"""
Dev Tools Module.

Examples:
    >>> from copy import deepcopy
    >>> import environs
    >>>
    >>> deepcopy(environs.Env()) # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    RecursionError: maximum recursion depth exceeded
"""
__all__ = (
    'timeit',
    'rich_inspect',
    'rich_print',

    '__version__',
    'console',
    'cp',
    'debug',
    'ic',
    'icc',
    'print_exception',
    'PYTHON_VERSIONS',

    'RcDev',
    'pretty_install',
    'traceback_install',
)
import timeit as timeit
from os import environ
from pathlib import Path
import colorama
import rich.pretty
import rich.traceback
from click.exceptions import Exit
from devtools import Debug
from icecream import IceCreamDebugger
from jinja2 import Template
from rich import inspect as rich_inspect
from rich import print as rich_print
from rich.console import Console
from semver import VersionInfo
from typer import Typer

__version__ = '0.0.34'
app = Typer(context_settings=dict(help_option_names=['-h', '--help'], color=True))
console = Console(color_system='256')
cp = console.print
debug = Debug(highlight=True)
fmic = IceCreamDebugger(prefix=str()).format
fmicc = IceCreamDebugger(prefix=str(), includeContext=True).format
ic = IceCreamDebugger(prefix=str())
icc = IceCreamDebugger(prefix=str(), includeContext=True)
print_exception = console.print_exception
PYTHON_VERSIONS = (VersionInfo(3, 9), VersionInfo(3, 10),)


class RcDev:
    path = Path(__file__).parent.parent
    python = f'{PYTHON_VERSIONS[0].major}.{PYTHON_VERSIONS[0].minor}'

    def __init__(self, path=path, pypi=None, script=None):
        path = Path(path); self.path = path if path.is_dir() else path.parent
        self.pypi = pypi if pypi else path.name
        self.script = script if script else pypi
        requirements = self.path / 'requirements.txt'
        self.requires = sorted(set((self.path / 'requirements.txt').read_text().splitlines()))
        requirements.write_text('\n'.join(self.requires))

    def pyproject(self): Template((Path(__file__).parent / 'pyproject.toml.j2').read_text(), autoescape=True).stream(
        **vars(self) | vars(type(self))).dump(str(self.path / 'pyproject.toml'))


def pretty_install(cons=console, expand=False): return rich.pretty.install(cons, expand_all=expand)


@app.command()
def pyproject(path: str = str(RcDev.path), pypi: str = RcDev.path.name, script: str = RcDev.path.name):
    """pyproject.toml"""
    RcDev(path=path, pypi=pypi, script=script).pyproject()


def traceback_install(cons=console, extra=5, locs=True): return rich.traceback.install(
    console=cons, extra_lines=extra, show_locals=locs)


colorama.init()
environ['PYTHONWARNINGS'] = 'ignore'
pretty_install(expand=True)

if __name__ == '__main__':
    try:
        Exit(app())
    except KeyboardInterrupt:
        print('Aborted!')
        Exit()
