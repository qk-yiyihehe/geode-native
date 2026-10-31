@echo off
setlocal
cd /d "%~dp0"
if not exist "build\geode-finder.exe" (
    echo Build first: pwsh -File build.ps1
    exit /b 1
)
if exist "finder.json" (
    "build\geode-finder.exe" --config finder.json %*
) else (
    "build\geode-finder.exe" %*
)
exit /b %errorlevel%
