import sys
sys.setrecursionlimit(10000)
import os
from cx_Freeze import setup, Executable
import nltk
import torch
import sklearn
import shutil

nltk.download('punkt')
try:
    punkt_path = str(nltk.data.find('tokenizers/punkt'))
#    include_files.append((punkt_path, 'nltk_data/tokenizers/punkt'))
except LookupError:   
    print("Error: punkt tokenizer not found. Run nltk.download('punkt') before building.")

torch_lib_dir = os.path.join(os.path.dirname(torch.__file__), 'lib')
sklearn_lib_dir = os.path.join(os.path.dirname(sklearn.__file__), ".libs")
    
# Define the directories you want to include
include_directories = [
    ('eSpeak NG', 'eSpeak NG'),  # (source_dir, target_dir)
    ('ffmpeg', 'ffmpeg'),
    ('Epub', 'Epub'),
    ('Media', 'Media'),
    ('Models', 'Models'),
    ('StyleTTS2', 'StyleTTS2'),
    ('StyleTTS2', os.path.join("lib","StyleTTS2")),
    ('Imgs', 'Imgs'),
    (punkt_path, os.path.join("nltk_data","tokenizers","punkt","PY3")),
    (punkt_path, os.path.join("nltk_data","tokenizers","punkt")),
    (torch_lib_dir, os.path.join("lib","torch","lib")),
    (sklearn_lib_dir, os.path.join("lib","sklearn",".libs"))
]

# Create a list of files to include
include_files = []
for source_dir, target_dir in include_directories:
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            include_files.append((os.path.join(root, file), os.path.join(target_dir, os.path.relpath(root, source_dir), file)))
            
include_files.append("config.ini")
include_files.append("epubTextToSpeech.bat")

include_files.append((torch_lib_dir, os.path.join("lib","torch","lib")))
include_files.append((sklearn_lib_dir, os.path.join("lib","sklearn",".libs")))

#executables = [Executable("epubTextToSpeech.py", base = 'Win32GUI', icon = 'D:\Python\Epubtextspeech\Imgs\voice_presentation.ico')]
executables = [Executable("epubTextToSpeech.py", icon = 'D:\Python\Epubtextspeech\Imgs\voice_presentation.ico'),"setup.py"]


build_exe_options = {
    "packages": ["transformers","torch","nltk","numpy","tokenizers","phonemizer","scipy","customtkinter","tkinter","numba","sklearn"],
    "excludes": [],
    "include_msvcr": True,
    "include_files": include_files
}

print(f"Running cx_freeze with build_options [{build_exe_options}] and executables [{executables}]")

# Calling setup function
setup(
    name = "EPub Text To Speech",
    version = "0.7",
    description = "Synthetize text from epub or pdf and then play audio with synchronized reading",
    options={"build_exe": build_exe_options},
    executables = executables,
)