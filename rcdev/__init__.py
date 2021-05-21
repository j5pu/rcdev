#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
"""RC Cmd Package."""
from .main import *
__all__ = main.__all__

app = Typer(context_settings=dict(help_option_names=['-h', '--help'], color=True))


@app.command()
def pyproject(path: str = str(RcDev.path), pypi: str = RcDev.path.name, script: str = RcDev.path.name):
    """pyproject.toml"""
    RcDev(path=path, pypi=pypi, script=script).pyproject()


if __name__ == '__main__':
    try:
        TyperExit(app())
    except KeyboardInterrupt:
        print('Aborted!')
        TyperExit()
