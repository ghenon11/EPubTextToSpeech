@echo off

rem Set important environment variables
set BASE_DIR=%cd%
set "FFMPEG_DIR=%BASE_DIR%\ffmpeg\bin"
set "ESPEAK_LIB=%BASE_DIR%\eSpeak NG\libespeak-ng.dll"
set "PYTHON_MODULES=%BASE_DIR%\StyleTTS2;%BASE_DIR%\StyleTTS2\Modules"

set TF_ENABLE_ONEDNN_OPTS=0
set PHONEMIZER_ESPEAK_LIBRARY=%ESPEAK_LIB%
set PYTHONPATH=%PYTHON_MODULES%

rem Update PATH if ffmpeg not already included
echo %PATH% | find /i "%FFMPEG_DIR%" >nul
if errorlevel 1 set "PATH=%PATH%;%FFMPEG_DIR%"

rem Create logs directory if not exists
if not exist logs mkdir logs

rem Set log file with date and time
rem set "LOGFILE=logs\epubTextToSpeech_%DATE:~10,4%-%DATE:~4,2%-%DATE:~7,2%_%TIME:~0,2%%TIME:~3,2%%TIME:~6,2%.log"
set "LOGFILE=%BASE_DIR%\logs\epubTextToSpeech_Launcher.log"

rem Clean the logfile name (remove spaces)
set "LOGFILE=%LOGFILE: =0%"

echo Launching EPubTextToSpeech
echo Current date and time: %date% %time%
echo Launching EPubTextToSpeech  > "%LOGFILE%" 2>&1
echo Current date and time: %date% %time% >> "%LOGFILE%" 2>&1
rem Confirm ffmpeg is correctly available
where ffmpeg.exe >> "%LOGFILE%" 2>&1
ffmpeg.exe -version >> "%LOGFILE%" 2>&1
echo Logs saved to %LOGFILE%
rem Launch the Python script and log everything

set EXE_FILE=%BASE_DIR%\epubTextToSpeech.exe
if exist %EXE_FILE% (
	"%EXE_FILE%" >> "%LOGFILE%" 2>&1
) else (
	python "epubTextToSpeech.py" >> "%LOGFILE%" 2>&1
)
pause
