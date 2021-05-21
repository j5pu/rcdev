BUMP := patch  # <major|minor|patch>
all: install
.PHONY: venv requirements publish all
SHELL := $(shell command -v bash)
DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
PACKAGE := $(shell basename $(DIR))
VENV := $(DIR)venv
ACTIVATE := $(VENV)/bin/activate

venv:
	@test -d $(VENV) || @python3.9 -m venv $(VENV)

requirements: venv
	@source $(ACTIVATE); python3.9 -m pip install --upgrade -r $(DIR)requirements_dev.txt; deactivate

publish: requirements
	@source $(ACTIVATE); gall.sh; bump2version $(BUMP); flit publish; rm -r $(DIR)dist/; deactivate

install: publish
	@python3.9 -m pip install --upgrade $(PACKAGE)
