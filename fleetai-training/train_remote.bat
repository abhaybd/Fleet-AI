@ECHO OFF

if "%1"=="" goto bad_args
if "%2"=="" goto bad_args
goto bad_args_end

:bad_args
echo ERR: Proper usage is train_remote.bat [JOB NAME] [CONFIG PATH]
exit /B
:bad_args_end

set BUCKET=fleetai-storage
set JOB_NAME=%1
set JOB_DIR=gs://%BUCKET%/%JOB_NAME%

set LOCAL_CFG_PATH=%2
set REMOTE_CFG_PATH=%JOB_DIR%/config.yaml

echo Uploading config...
call gsutil cp %LOCAL_CFG_PATH% %REMOTE_CFG_PATH%

echo Submitting job...

set IMAGE_URI=gcr.io/cloud-ml-public/training/pytorch-xla.1-7
set PACKAGE_PATH=./fleetai
set REGION=us-central1
set MODULE_NAME=fleetai.train_gcp

call gcloud ai-platform jobs submit training %JOB_NAME% --job-dir %JOB_DIR% --region %REGION% --master-image-uri %IMAGE_URI% --scale-tier BASIC --module-name %MODULE_NAME% --package-path %PACKAGE_PATH% -- --config %REMOTE_CFG_PATH%
