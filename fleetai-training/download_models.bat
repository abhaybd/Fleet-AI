@ECHO OFF

if "%1"=="" goto bad_args
goto bad_args_end

:bad_args
echo ERR: Proper usage is download_models.bat [MODELS DIR]
exit /B
:bad_args_end

set "pre_dir=%cd%"
set MODEL_DIR=%1

call gsutil -m cp gs://fleetai-storage/models/* %MODEL_DIR%

for /R %MODEL_DIR% %%I in ("*.zip") do (
    if not exist "%%~dpnI" mkdir "%%~dpnI"
    echo Extracting model "%%~fI"
    del /Q "%%~dpnI\*"
    move "%%~fI" "%%~dpnI" > nul
    cd "%%~dpnI"
    tar -xf "%%~dpnI\%%~nxI"
    del "%%~dpnI\%%~nxI"
)

cd %pre_dir%
