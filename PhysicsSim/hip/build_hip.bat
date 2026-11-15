@echo off
rem Builds PhysicsHip.dll, the AMD HIP backend of the physics.
rem   build_hip.bat [output dir] [offload archs, e.g. "gfx1100 gfx1030"] [Visual Studio install dir]
rem Needs the AMD HIP SDK (ROCm for Windows) and Visual Studio (for the MSVC headers and linker).
rem Exit code 0 = built, 1 = failed, 2 = HIP SDK not found (not an error: the app then runs CPU-only).
setlocal EnableDelayedExpansion

set "OUT=%~1"
if "%OUT%"=="" set "OUT=%~dp0..\..\bin\Release\"
if not "%OUT:~-1%"=="\" set "OUT=%OUT%\"
set "ARCHS=%~2"
rem RDNA2 / RDNA3 / RDNA4 GPUs
if "%ARCHS%"=="" set "ARCHS=gfx1030 gfx1100 gfx1101 gfx1102 gfx1200 gfx1201"
set "VSROOT=%~3"

rem ---- HIP SDK ----
set "HIP=%HIP_PATH%"
if "%HIP%"=="" for /d %%d in ("%ProgramFiles%\AMD\ROCm\*") do set "HIP=%%d"
if "%HIP%"=="" ( echo HIP SDK not found: skipping the GPU backend. & exit /b 2 )
if "%HIP:~-1%"=="\" set "HIP=%HIP:~0,-1%"
if not exist "%HIP%\bin\clang++.exe" ( echo HIP SDK not found in "%HIP%": skipping the GPU backend. & exit /b 2 )

rem ---- Visual Studio ----
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if "%VSROOT%"=="" if exist "%VSWHERE%" for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -prerelease -products * -property installationPath`) do set "VSROOT=%%i"
if "%VSROOT%"=="" ( echo Visual Studio not found. & exit /b 1 )
if "%VSROOT:~-1%"=="\" set "VSROOT=%VSROOT:~0,-1%"

rem The STL of the newest MSVC toolset (14.5x) refuses clang older than 22 and the HIP SDK ships clang 21, so use
rem an older toolset when there is one: 14.4x / 14.3x (VS 2022), then 14.2x (VS 2019). Start from a clean
rem environment so that the toolset the caller may have initialised does not leak in.
for /f "tokens=1 delims==" %%v in ('set VSCMD_ 2^>nul') do set "%%v="
set "INCLUDE=" & set "LIB=" & set "LIBPATH=" & set "VCINSTALLDIR="
set "VCVER="
for %%p in (14.4 14.3 14.2) do if not defined VCVER for /d %%d in ("%VSROOT%\VC\Tools\MSVC\%%p*") do if not defined VCVER set "VCVER=%%~nxd"
set "VCARG="
if defined VCVER set "VCARG=-vcvars_ver=%VCVER%"
call "%VSROOT%\VC\Auxiliary\Build\vcvars64.bat" %VCARG% >nul || ( echo vcvars64 failed. & exit /b 1 )
if not defined VCVER echo Warning: no MSVC 14.2x-14.4x toolset installed; the newest one may be rejected by clang 21.

set "OFFLOAD="
for %%a in (%ARCHS%) do set "OFFLOAD=!OFFLOAD! --offload-arch=%%a"

echo Building PhysicsHip.dll with HIP SDK "%HIP%", MSVC toolset %VCVER%, for:%ARCHS%
if not exist "%OUT%" mkdir "%OUT%"
"%HIP%\bin\clang++.exe" -x hip --hip-path="%HIP%" %OFFLOAD% -O3 -std=c++20 -DNDEBUG -shared ^
    -I"%~dp0.." "%~dp0PhysicsHip.hip" -o "%OUT%PhysicsHip.dll" -L"%HIP%\lib" -lamdhip64 ^
    -Wno-unused-command-line-argument
if errorlevel 1 exit /b 1
rem the import library and export file are by-products of linking a dll
del "%OUT%PhysicsHip.lib" "%OUT%PhysicsHip.exp" 2>nul
exit /b 0
