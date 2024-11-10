@echo off
set file=%1

:: Run ruff to format file
ruff format %file%

:: Run docformatter to format the file in place
docformatter --in-place %file%

