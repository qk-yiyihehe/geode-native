[CmdletBinding()]
param([switch]$Test)
$ErrorActionPreference = 'Stop'
$projectDir = $PSScriptRoot
$buildDir = Join-Path $projectDir 'build'
if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) { throw 'CMake is required on PATH.' }
$configure = @('-S', $projectDir, '-B', $buildDir, '-DCMAKE_BUILD_TYPE=Release')
# 已有构建沿用缓存的生成器，避免Ninja与MinGW Makefiles互相冲突。
if (-not (Test-Path -LiteralPath (Join-Path $buildDir 'CMakeCache.txt'))) {
    if (-not (Get-Command ninja -ErrorAction SilentlyContinue)) { throw 'Ninja is required for a new build.' }
    $configure += @('-G', 'Ninja')
}
$testing = if ($Test) { 'ON' } else { 'OFF' }
$configure += "-DBUILD_TESTING=$testing"
& cmake @configure
if ($LASTEXITCODE -ne 0) { throw 'CMake configuration failed.' }
& cmake --build $buildDir --parallel
if ($LASTEXITCODE -ne 0) { throw 'Native build failed.' }
if ($Test) {
    & ctest --test-dir $buildDir --output-on-failure
    if ($LASTEXITCODE -ne 0) { throw 'Native tests failed.' }
}
Write-Host (Join-Path $buildDir 'geode-finder.exe')
