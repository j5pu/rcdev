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
	@source $(ACTIVATE); $(VENV)/bin/python3.9 -m pip install --upgrade -q -r $(DIR)requirements_dev.txt; deactivate

publish: requirements
	@source $(ACTIVATE); bump2version $(BUMP); gpush.sh; flit publish; rm -rf $(DIR)dist/; deactivate

install: publish
	@deactivate >/dev/null 2>&1; /usr/local/bin/python3.9 -m pip install --upgrade -q $(PACKAGE); \
/usr/local/bin/python3.9 -m pip install --upgrade -q $(PACKAGE)
