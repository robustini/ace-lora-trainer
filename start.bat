@echo off
REM ============================================================
REM ACE-Step LoRA Trainer — Quick Start (Windows)
REM ============================================================

REM Check if venv exists
if not exist "env\Scripts\activate.bat" (
    echo.
    echo [ERROR] Virtual environment not found!
    echo         Please run install.bat first.
    echo.
    pause
    exit /b 1
)

REM Activate venv
call env\Scripts\activate.bat

REM If command-line args were passed, skip the menu
if not "%~1"=="" (
    python launch.py %*
    exit /b
)

REM Interactive menu
:menu
echo.
echo ============================================================
echo   ACE-Step LoRA Trainer
echo ============================================================
echo.
echo   1) LoRA Trainer
echo   2) Captioner
echo   3) Both (Trainer + Captioner)
echo.
set /p choice="  Select [1-3]: "

if "%choice%"=="1" (
    python launch.py --mode train
) else if "%choice%"=="2" (
    python launch.py --mode caption
) else if "%choice%"=="3" (
    python launch.py --mode both
) else (
    echo   Invalid choice, please try again.
    goto menu
)
