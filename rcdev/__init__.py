#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
"""RC Cmd Package."""
from .main import *
__all__ = main.__all__

if __name__ == '__main__':
    try:
        TyperExit(app())
    except KeyboardInterrupt:
        print('Aborted!')
        TyperExit()
