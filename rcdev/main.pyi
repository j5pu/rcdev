from logging import getLogger
from logging import NOTSET
from socket import gaierror
from typing import Final
from typing import Union

import ansible as ansible
import astpretty as astpretty
import asttokens as asttokens
import bson as bson
import box as box
import click as click
import click_completion as click_completion
import colorama as colorama
import colorlog as colorlog
import decouple as decouple
import devtools as devtools
import distro as distro
import dns as dns
import dotenv as dotenv
import dpath as dpath
import environs as environs
import envtoml as envtoml
import executing as executing
import flit as flit
import funcy as funcy
import git as git
import icecream as icecream
import inflect as inflect
import intervaltree as intervaltree
import iocextract as iocextract
import ipaddress as ipaddress
import jsonpickle as jsonpickle
import jsonpickle.ext.numpy as pickle_np
import jupyter as jupyter
import libnmap as libnmap
import loguru as loguru
import mackup as mackup
import mashumaro as mashumaro
import more_itertools as more_itertools
import motor as motor
import notifiers as notifiers
import numpy as np
import pandas as pd
import paramiko as paramiko
import pickledb as pickledb
import psutil as psutil
import pydantic as pydantic
import pymetasploit3 as pymetasploit3
import pymongo as pymongo
import pypattyrn as pypattyrn
import pytest as pytest
import yaml as pyyaml
import requests as requests
import retry as retry
import returns as returns
import rich as rich
import rich.pretty as rich_pretty
import rich.traceback as rich_traceback
import semver as semver
import setuptools as setuptools
import shell_proc as shell_proc
import shellingham as shellingham
import structlog as structlog
import sty as sty
import taskflow as taskflow
import tenacity as tenacity
import thefuck as thefuck
import tinydb as tinydb
import toolz as toolz
import toolz.itertoolz as toolz_iter
import toolz.functoolz as toolz_func
import toolz.dicttoolz as toolz_dict
import tqdm as tqdm
import typer as typer
import urllib3 as urllib3
import verboselogs as verboselogs
import xmltodict as xmltodict
from astpretty import pformat as astformat
from astpretty import pprint as astprint
from asttokens import ASTTokens as ASTTokens
from box import Box as Box
from bson import Binary as Binary
from bson import CodecOptions as CodecOptions
from bson import ObjectId as ObjectId
from bson.binary import USER_DEFINED_SUBTYPE as USER_DEFINED_SUBTYPE
from bson.codec_options import TypeDecoder as TypeDecoder
from bson.codec_options import TypeRegistry as TypeRegistry
from colorlog import ColoredFormatter as ColoredFormatter
from colorlog import LevelFormatter as LevelFormatter
from decorator import decorator as decorator
from decouple import config as decouple_config
from devtools import Debug as Debug
from distro import LinuxDistribution as LinuxDistribution
from dotenv import load_dotenv as load_dotenv
from dpath.util import delete as dpathdelete
from dpath.util import get as dpathget
from dpath.util import new as dpathnew
from dpath.util import search as dpathsearch
from dpath.util import set as dpathset
from dpath.util import values as dpathvalues
from environs import Env as Environs
from environs import ErrorMapping as EnvErrorMapping
from executing import Executing as Executing
from furl import furl as furl
from git import Commit as Commit
from git import GitCmdObjectDB as GitCmdObjectDB
from git import GitConfigParser as GitConfigParser
from git import Remote as Remote
from git import Repo as GitRepo
from git.exc import GitCommandError as GitCommandError
from git.exc import InvalidGitRepositoryError as InvalidGitRepositoryError
from git.exc import NoSuchPathError as NoSuchPathError
from git.refs import SymbolicReference as GitSymbolicReference
from git.util import IterableList as GitIterableList
from icecream import IceCreamDebugger as IceCreamDebugger
from intervaltree import Interval as Interval
from intervaltree import IntervalTree as IntervalTree
from iocextract import extract_ips as extract_ips
from ipaddress import AddressValueError as AddressValueError
from ipaddress import ip_address as ip_address
from ipaddress import IPv4Address as IPv4Address
from ipaddress import IPv6Address as IPv6Address
from ipaddress import NetmaskValueError as NetmaskValueError
from jinja2 import Template as Template
from jsonpickle import decode as jsondecode
from jsonpickle import encode as jsonencode
from jsonpickle.pickler import Pickler as Pickler
from jsonpickle.unpickler import Unpickler as Unpickler
from jsonpickle.util import importable_name as importable_name
from libnmap.parser import NmapParser as LibNmapParser
from loguru import logger as loguru_logger
from mashumaro import DataClassDictMixin as DataClassDictMixin
from mashumaro import DataClassJSONMixin as DataClassJSONMixin
from mashumaro import DataClassMessagePackMixin as DataClassMessagePackMixin
from mashumaro import DataClassYAMLMixin as DataClassYAMLMixin
from mashumaro import field_options as field_options
from mashumaro import MissingField as MissingField
from more_itertools import collapse as collapse
from more_itertools import consume as consume
from more_itertools import first_true as first_true
from more_itertools import map_reduce as map_reduce
from more_itertools import side_effect as side_effect
from motor.motor_asyncio import AsyncIOMotorClient as AsyncIOMotorClient
from motor.motor_asyncio import AsyncIOMotorCollection as AsyncIOMotorCollection
from motor.motor_asyncio import AsyncIOMotorCursor as AsyncIOMotorCursor
from motor.motor_asyncio import AsyncIOMotorDatabase as AsyncIOMotorDatabase
from nested_lookup import nested_lookup as nested_lookup
from nested_lookup.nested_lookup import _nested_lookup
from paramiko import AuthenticationException as AuthenticationException
from paramiko import AutoAddPolicy as AutoAddPolicy
from paramiko import BadAuthenticationType as BadAuthenticationType
from paramiko import BadHostKeyException as BadHostKeyException
from paramiko import PasswordRequiredException as PasswordRequiredException
from paramiko import SSHClient as SSHClient
from paramiko import SSHConfig as SSHConfig
from paramiko import SSHException as SSHException
from paramiko.ssh_exception import NoValidConnectionsError as NoValidConnectionsError
from psutil import AccessDenied as AccessDenied
from psutil import LINUX as LINUX
from psutil import MACOS as MACOS
from psutil import NoSuchProcess as NoSuchProcess
from psutil import Process as PsProcess
from psutil import process_iter as process_iter
from pymongo import MongoClient as PyMongoClient
from pymongo import ReturnDocument as ReturnDocument
from pymongo.cursor import Cursor as PyMongoCursor
from pymongo.database import Collection as PyMongoCollection
from pymongo.database import Database as PyMongoDB
from pymongo.errors import AutoReconnect as AutoReconnect
from pymongo.errors import ConfigurationError as ConfigurationError
from pymongo.errors import ConnectionFailure as ConnectionFailure
from pymongo.errors import ServerSelectionTimeoutError as ServerSelectionTimeoutError
from pypattyrn.behavioral.null import Null as Null
from pypattyrn.behavioral.observer import Observable as Observable
from pypattyrn.behavioral.observer import Observer as Observer
from pypattyrn.creational.builder import Builder as Builder
from pypattyrn.creational.builder import Director as Director
from pypattyrn.structural.adapter import Adapter as Adapter
from pypattyrn.structural.composite import Composite as Composite
from pypattyrn.structural.decorator import DecoratorSimple as DecoratorSimple
from pypattyrn.structural.decorator import DecoratorComplex as DecoratorComplex
from pypattyrn.structural.decorator import CallWrapper as CallWrapper
from returns.future import FutureResultE as ReturnsFutureResultE
from returns.future import future_safe as returns_future_safe
from returns.io import impure as returns_impure
from returns.io import IO as ReturnsIO
from returns.io import IOFailure as ReturnsIOFailure
from returns.io import IOSuccess as ReturnsIOSuccess
from returns.maybe import Maybe as ReturnsMaybe
from returns.maybe import maybe as returns_maybe
from returns.pipeline import flow as returns_flow
from returns.pointfree import bind as returns_bind
from returns.result import Result as ReturnsResult
from returns.result import safe as returns_safe
from rich import inspect as rich_inspect
from rich import print as rich_print
from rich.columns import Columns as RichColumns
from rich.console import Console as RichConsole
from rich.logging import RichHandler as RichHandler
from rich.progress import Progress as RichProgress
from rich.progress import track as rich_track
from rich.prompt import Confirm as RichConfirm
from rich.prompt import Prompt as RichPrompt
from rich.highlighter import RegexHighlighter as RichRegexHighlighter
from rich.style import Style as RichStyle
from rich.table import Table as RichTable
from rich.text import Text as RichText
from rich.theme import Theme as RichTheme
from rich.tree import Tree as RichTree
from ruamel.yaml import YAML as yaml
from semver import VersionInfo as VersionInfo
from setuptools import Distribution as SetUpToolsDistribution
from setuptools import find_packages as find_packages
from setuptools.command.develop import develop as SetUpToolsDevelop
from setuptools.command.install import install as SetUpToolsInstall
from shell_proc import Shell as Shell
from structlog import configure as struct_configure
from structlog import configure_once as struct_configure_once
from structlog import get_config as struct_get_config
from structlog import get_context as struct_get_context
from structlog import is_configured as struct_is_configured
from structlog import make_filtering_bound_logger as make_filtering_bound_logger
from structlog import PrintLogger as PrintLogger
from structlog import PrintLoggerFactory as PrintLoggerFactory
from structlog import reset_defaults as struct_reset_defaults
from structlog.contextvars import bind_contextvars as struct_bind_contextvars
from structlog.contextvars import clear_contextvars as struct_clear_contextvars
from structlog.contextvars import merge_contextvars as struct_merge_contextvars
from structlog.contextvars import unbind_contextvars as struct_unbind_contextvars
from structlog.dev import ConsoleRenderer as ConsoleRenderer
from structlog.dev import set_exc_info as set_exc_info
from structlog.processors import ExceptionPrettyPrinter as ExceptionPrettyPrinter
from structlog.processors import format_exc_info as format_exc_info
from structlog.processors import JSONRenderer as JSONRenderer
from structlog.processors import KeyValueRenderer as KeyValueRenderer
from structlog.processors import StackInfoRenderer as StackInfoRenderer
from structlog.processors import TimeStamper as TimeStamper
from structlog.stdlib import add_log_level as add_log_level
from structlog.stdlib import add_log_level_number as add_log_level_number
from structlog.stdlib import add_logger_name as add_logger_name
from structlog.stdlib import AsyncBoundLogger as AsyncBoundLogger
from structlog.stdlib import BoundLogger as BoundLogger
from structlog.stdlib import filter_by_level as filter_by_level
from structlog.stdlib import get_logger as get_logger
from structlog.stdlib import LoggerFactory as LoggerFactory
from structlog.stdlib import PositionalArgumentsFormatter as PositionalArgumentsFormatter
from structlog.stdlib import ProcessorFormatter as ProcessorFormatter
from structlog.stdlib import render_to_log_kwargs as render_to_log_kwargs
from structlog.testing import capture_logs as capture_logs
from structlog.testing import CapturedCall as CapturedCall
from structlog.testing import CapturingLogger as CapturingLogger
from structlog.testing import LogCapture as LogCapture
from structlog.testing import ReturnLogger as ReturnLogger
from structlog.threadlocal import as_immutable as struct_as_immutable
from structlog.threadlocal import bind_threadlocal as bind_threadlocal
from structlog.threadlocal import merge_threadlocal as merge_threadlocal
from structlog.threadlocal import wrap_dict as struct_wrap_dict
from structlog.types import BindableLogger as BindableLogger
from structlog.types import Context as StructContext
from structlog.types import EventDict as StructEventDict
from structlog.types import ExcInfo as StructExcInfo
from structlog.types import FilteringBoundLogger as FilteringBoundLogger
from structlog.types import Processor as StructProcessor
from structlog.types import WrappedLogger as WrappedLogger
from thefuck.utils import Cache as LazyCache
from thefuck.utils import memoize as memoize
from toolz.itertoolz import diff as toolz_diff
from toolz.itertoolz import frequencies as toolz_frequencies
from toolz.itertoolz import get as toolz_get
from toolz.itertoolz import isdistinct as toolz_isdistinct
from toolz.itertoolz import mapcat as toolz_mapcat
from toolz.itertoolz import remove as toolz_false
from toolz.itertoolz import unique as toolz_unique
from toolz.functoolz import apply as toolz_apply
from toolz.functoolz import complement as toolz_complement
from toolz.functoolz import compose as toolz_compose
from toolz.functoolz import compose_left as toolz_compose_left
from toolz.functoolz import curry as toolz_curry
from toolz.functoolz import do as toolz_do
from toolz.functoolz import juxt as toolz_juxt
from toolz.functoolz import pipe as toolz_pipe
from toolz.functoolz import thread_first as toolz_thread_first
from toolz.functoolz import thread_last as toolz_thread_last
from toolz.dicttoolz import assoc as toolz_assoc
from toolz.dicttoolz import assoc_in as toolz_assoc_in
from toolz.dicttoolz import dissoc as toolz_dissoc
from toolz.dicttoolz import get_in as toolz_get_in
from toolz.dicttoolz import itemfilter as toolz_itemfilter
from toolz.dicttoolz import itemmap as toolz_itemmap
from toolz.dicttoolz import keyfilter as toolz_keyfilter
from toolz.dicttoolz import keymap as toolz_keymap
from toolz.dicttoolz import merge as toolz_merge
from toolz.dicttoolz import merge_with as toolz_merge_with
from toolz.dicttoolz import update_in as toolz_update_in
from toolz.dicttoolz import valfilter as toolz_valfilter
from toolz.dicttoolz import valmap as toolz_valmap
from tqdm.asyncio import tqdm as asynctqdm
from urllib3 import disable_warnings as urllib3_disable_warnings
from varname import varname as var_name
from verboselogs import NOTICE as NOTICE
from verboselogs import SPAM as SPAM
from verboselogs import SUCCESS as SUCCESS
from verboselogs import VERBOSE as VERBOSE
from verboselogs import VerboseLogger as VerboseLogger
from verboselogs import VerboseLogger as VerboseLogger

import boltons as boltons
import cachetools as cachetools
import dataclass_factory as dataclass_factory
import frozendict as frozendict
import geoip2 as geoip2
import glom as glom
import heartrate as heartrate
import janus as janus
import jsonpath_ng as jsonpath_ng
import matplotlib as matplotlib
import orjson as orjson
import pure_eval as pure_eval
import seaborn as seaborn
import snoop as snoop
import sorcery as sorcery
import stack_data as stack_data
import ujson as ujson

__version__: str
__all__: tuple[str, ...]

# Protected
nested_lookup_protected = _nested_lookup


# Aliases
GitCommandWrapperType = GitRepo.GitCommandWrapperType


# Constants
console: RichConsole
cp: console.print
debug: Debug
DISTRO: LinuxDistribution
fmic: IceCreamDebugger().format
fmicc: IceCreamDebugger().format
ic: IceCreamDebugger
icc: IceCreamDebugger
IPvAddress = Union[IPv4Address, IPv6Address]
KALI: bool
MONGO_EXCEPTIONS: Final[(gaierror, ConnectionFailure, AutoReconnect, ServerSelectionTimeoutError, ConfigurationError, )]
plural = inflect.engine().plural
print_exception: console.print_exception
PYTHON_VERSIONS: tuple[VersionInfo, VersionInfo]
UBUNTU: bool

# Install
def pretty_install(cons: RichConsole = ..., expand: bool = ...): ...
def traceback_install(cons: RichConsole = ..., extra: int = ..., locs: bool = ...): ...

# Init
colorama.init()
getLogger(paramiko.__name__).setLevel(NOTSET)
pickle_np.register_handlers()
pretty_install(expand=True)
struct_configure(logger_factory=LoggerFactory())
urllib3_disable_warnings()

__all__: tuple[str, ...]
__version__: str
