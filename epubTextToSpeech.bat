echo off
set TF_ENABLE_ONEDNN_OPTS=0

rem SET PATH=C:\Program Files (x86)\eSpeak\command_line;C:\Program Files (x86)\eSpeak\;%PATH%
rem python "epubTextToSpeech.py" > logs\epubTextToSpeech_Launcher.log 2>&1
path|find /i "Epubtextspeech\ffmpeg" >nul || set path=%path%;D:\Python\Epubtextspeech\ffmpeg\bin
set PHONEMIZER_ESPEAK_LIBRARY=D:\Python\Epubtextspeech\eSpeak NG\libespeak-ng.dll
rem set PHONEMIZER_ESPEAK_LIBRARY=C:\Program Files\eSpeak NG\libespeak-ng.dll
set PYTHONPATH=D:\Python\Epubtextspeech\StyleTTS2;D:\Python\Epubtextspeech\StyleTTS2\Modules\
echo on
whereis ffmppeg.exe
ffmpeg.exe -version
python -B "epubTextToSpeech.py" 