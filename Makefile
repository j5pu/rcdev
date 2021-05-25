BUMP := patch  # <major|minor|patch>
all: rclib
.PHONY: rcdev rctest rccmd rclib all
SHELL := $(shell command -v bash)
DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
PACKAGE := $(shell basename $(DIR))
VENV := $(DIR)venv
ACTIVATE := $(VENV)/bin/activate

rcdev:
	@test -d $(VENV) || @python3.9 -m venv $(VENV)
	@source $(ACTIVATE); $(VENV)/bin/python3.9 -m pip install --upgrade -q -r $(DIR)requirements.txt; deactivate
	@source $(ACTIVATE); $(VENV)/bin/python3.9 -W ignore -m rcdev.tools; deactivate
	@source $(ACTIVATE); gall.sh; bump2version $(BUMP); gall.sh; flit publish; rm -rf $(DIR)dist/; deactivate
	@deactivate >/dev/null 2>&1; /usr/local/bin/python3.9 -m pip install --upgrade -q $(PACKAGE); \
/usr/local/bin/python3.9 -m pip install --upgrade -q $(PACKAGE)

rcvars: rcdev
	@cd $${HOME}/rcvars; make

rctest: rcvars
	@cd $${HOME}/rctest; make

rcutils: rctest
	@cd $${HOME}/rcvars; make

rccmd: rctest
	@cd $${HOME}/rccmd; make

rclib: rctest
	@cd $${HOME}/rclib; make
