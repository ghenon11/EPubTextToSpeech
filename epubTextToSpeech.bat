echo off
set TF_ENABLE_ONEDNN_OPTS=0

rem SET PATH=C:\Users\gheno\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg.Shared_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-6.1.1-full_build-shared\bin;%PATH%
rem python "epubTextToSpeech v3.py" > logs\epubTextToSpeech_Launcher.log 2>&1
echo on
rem ffmpeg.exe -version
python "epubTextToSpeech.py"