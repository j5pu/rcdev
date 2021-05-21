BUMP := patch  # <major|minor|patch>
all: install
.PHONY: all venv requirements publish install
SHELL := $(shell command -v bash)
DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
PACKAGE := $(shell basename $(DIR))
VENV := $(DIR)/venv

venv:
	@test -d $(VENV) || @/usr/local/bin/python3.9 -m venv $(VENV)

requirements: venv
	@$(VENV)/bin/python3.9 -m pip install --upgrade -r $(DIR)requirements_dev.txt

publish: requirements
	@gall.sh; bump2version $(BUMP); flit publish; rm -r $(DIR)dist/

install: publish
	@/usr/local/bin/python3.9 -m pip install --upgrade $(PACKAGE)
