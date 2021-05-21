BUMP := patch  # <major|minor|patch>
all: install
.PHONY: all publish install
SHELL := $(shell command -v bash)
DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
PACKAGE := $(shell basename $(DIR))

publish:
	@gall.sh; bump2version patch; flit publish; rm -r $(DIR)/dist/

install: publish
	@/usr/local/bin/python3.9 -m pip install --upgrade $(PACKAGE)
