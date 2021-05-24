from pathlib import Path

from typing import ClassVar
from typing import Union

from devtools import Debug
from icecream import IceCreamDebugger
from rich import inspect as rich_inspect
from rich import print as rich_print
from rich.console import Console
from typer import Typer

__all__: tuple[str, ...]
__version__: str

console: Console
cp: console.print
debug: Debug
fmic: IceCreamDebugger().format
fmicc: IceCreamDebugger().format
ic: IceCreamDebugger
icc: IceCreamDebugger
print_exception: console.print_exception
def pretty_install(cons: Console = ..., expand: bool = ...): ...
def traceback_install(cons: Console = ..., extra: int = ..., locs: bool = ...): ...

class RcDev:
    path: Union[Path, str] = ...
    pypi: str = ...
    script: str = ...
    python: ClassVar[str] = ...
    requires: list[str, ...] = ...
    def __init__(self, path: Union[Path, str] = ..., pypi: str = ..., script: str = ...) -> None: ...
    def pyproject(self) -> None: ...
app: Typer = ...
@app.command
def pyproject(path: str = ..., pypi: str = ..., script: str = ...) -> None: ...
