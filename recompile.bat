@echo off

REM Command 1
pip uninstall pytrino --yes

REM Command 2
python setup.py build

REM Command 3
python setup.py install