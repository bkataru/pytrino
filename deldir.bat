@echo off

set "directory1=.\build"
set "directory2=.\dist"

REM Check if the directories exist before deleting
if exist "%directory1%" (
    REM Delete directory1 and its contents
    rmdir /s /q "%directory1%"
    echo Directory '%directory1%' deleted.
) else (
    echo Directory '%directory1%' does not exist.
)

if exist "%directory2%" (
    REM Delete directory2 and its contents
    rmdir /s /q "%directory2%"
    echo Directory '%directory2%' deleted.
) else (
    echo Directory '%directory2%' does not exist.
)