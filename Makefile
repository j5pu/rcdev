BUMP := patch  # <major|minor|patch>
all: rcdev
.PHONY: rcdev rctest rccmd rclib all
SHELL := $(shell command -v bash)
DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
PACKAGE := $(shell basename $(DIR))
VENV := $(DIR)venv
ACTIVATE := $(VENV)/bin/activate

#RCCMD = $${HOME}/rccmd
#RCLIB = $${HOME}/rclib
#RCTEST = $${HOME}/rctest

rcdev:
	@test -d $(VENV) || @python3.9 -m venv $(VENV)
	@source $(ACTIVATE); $(VENV)/bin/python3.9 -m pip install --upgrade -q -r $(DIR)requirements.txt; deactivate
	@source $(ACTIVATE); $(VENV)/bin/python3.9 -W ignore -m rcdev.main; deactivate
	@source $(ACTIVATE); bump2version $(BUMP); gpush.sh; flit publish; rm -rf $(DIR)dist/; deactivate
	@deactivate >/dev/null 2>&1; /usr/local/bin/python3.9 -m pip install --upgrade -q $(PACKAGE); \
/usr/local/bin/python3.9 -m pip install --upgrade -q $(PACKAGE)

rctest: rcdev
	@cd $${HOME}/rctest; make

rccmd: rctest
	@cd $${HOME}/rccmd; make

rclib: rctest
	@cd $${HOME}/rclib; make
