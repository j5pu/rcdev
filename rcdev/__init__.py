#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
"""RC Dev Package."""
from click.exceptions import Exit

from .tools import *
__all__ = tools.__all__

if __name__ == '__main__':
    try:
        Exit(app())
    except KeyboardInterrupt:
        print('Aborted!')
        Exit()
