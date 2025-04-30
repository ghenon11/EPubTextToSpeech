@echo off
echo Building Executable
echo Start date and time: %date% %time%
echo Logs in logs\freeze.log
python freeze.py build > logs\freeze.log 2>&1
echo End date and time: %date% %time%