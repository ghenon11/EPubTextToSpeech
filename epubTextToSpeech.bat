echo off
set TF_ENABLE_ONEDNN_OPTS=0

rem SET PATH=C:\Program Files (x86)\eSpeak\command_line;C:\Program Files (x86)\eSpeak\;%PATH%
rem python "epubTextToSpeech.py" > logs\epubTextToSpeech_Launcher.log 2>&1
path|find /i "Gyan.FFmpeg.Shared_Microsoft" >nul || set path=%path%;C:\Users\gheno\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg.Shared_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-6.1.1-full_build-shared\bin
set PHONEMIZER_ESPEAK_LIBRARY=C:\Program Files\eSpeak NG\libespeak-ng.dll
set PYTHONPATH=D:\Python\Epubtextspeech\StyleTTS2;D:\Python\Epubtextspeech\StyleTTS2\Modules\
echo on
ffmpeg.exe -version
python "epubTextToSpeech.py" 