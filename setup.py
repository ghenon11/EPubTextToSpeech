from cx_Freeze import setup, Executable

# On appelle la fonction setup
setup(
    name = "EPub Text To Speech",
    version = "0.7",
    description = "Synthetize text from epub or pdf and then play audio with synchronized reading",
    executables = [Executable("epubTextToSpeech.py")],
)