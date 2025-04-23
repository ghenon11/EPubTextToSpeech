import simpleaudio as sa
from pydub import AudioSegment, effects
import re
import string

import pathlib
from bs4 import BeautifulSoup
from ebooklib import epub
import ebooklib
from PIL import Image
from tkinter import messagebox, filedialog
import tkinter as tk
import customtkinter as ctk
import os
import time
import traceback
import threading
import logging
import inspect

import utils
import config
import tts

print("Importing modules and launching application...")


# StylesTTS2 https://github.com/yl4579/StyleTTS2
# locally C:\Users\gheno\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages\styletts2
# using french pretrained model by Scralius
# https://huggingface.co/spaces/Scralius/StyleTTS2_French

# img: https://www.freeiconspng.com/
# tkinter colors: https://www.askpython.com/wp-content/uploads/2022/10/Tkinter-colour-list.png.webp

# winget install --id=Gyan.FFmpeg.Shared -v "6.1.1" -e
# C:\Users\gheno\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg.Shared_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-6.1.1-full_build-shared\bin\ffmpeg.exe

__author__ = "Guillaume HENON"
__version__ = "0.6"


class epubTextToSpeech(ctk.CTk):
    def __init__(self):
        super().__init__()

        try:
            self.title("EpubTextToSpeech")

            screen_width, screen_height = (
                self.winfo_screenwidth(),
                self.winfo_screenheight(),
            )
            # width, height = 1000, 600
            width, height = screen_width - 100, screen_height - 100
            x, y = (screen_width / 2) - \
                (width / 2), (screen_height / 2) - (height / 2)
            self.geometry(f"{width}x{height}+{int(x)}+{int(y)}")
            self.protocol("WM_DELETE_WINDOW", self.confirm_close)
            self.iconbitmap(config.ICO)
            # self.iconphoto(True, tk.PhotoImage(file=config.BG_IMG))
            self.font_size = 16
            self.grid_rowconfigure(0, weight=20)
            self.grid_rowconfigure(1, weight=1)
            self.grid_columnconfigure(0, weight=1)

            self.main_frame = ctk.CTkFrame(self)
            self.main_frame.grid_rowconfigure(0, weight=1)
            self.main_frame.grid_rowconfigure(1, weight=0)
            self.main_frame.grid_rowconfigure(2, weight=4)
            self.main_frame.grid_rowconfigure(3, weight=1)
            self.main_frame.grid_rowconfigure(4, weight=1)
            self.main_frame.grid_columnconfigure(0, weight=1)
            self.main_frame.grid_columnconfigure(1, weight=4)
            self.main_frame.grid(row=0, column=0, padx=10,
                                 pady=10, sticky="nsew")

            self.controls_frame = ctk.CTkFrame(self.main_frame)
            self.controls_frame.grid_rowconfigure(0, weight=1)
            self.controls_frame.grid_columnconfigure(0, weight=1)
            self.controls_frame.grid_columnconfigure(1, weight=1)
            self.controls_frame.grid_columnconfigure(1, weight=2)
            self.openebook_button = ctk.CTkButton(
                self.controls_frame, text="Open eBook", command=self.open_ebook
            )
            self.openebook_button.grid(row=0, column=0, pady=10, padx=10)
            self.synt_button = ctk.CTkButton(
                self.controls_frame,
                text="Synthetize Audio",
                command=self.convert_to_audio_callthread,
            )
            self.synt_and_play_button = ctk.CTkButton(
                self.controls_frame,
                text="Synthetize & Play Selection",
                command=self.synthetize_and_play_selection)
            self.synt_button.grid(row=0, column=1, padx=10, pady=10)
            self.synt_and_play_button.grid(row=0, column=2, padx=10, pady=10)
            self.controls_frame.grid(
                row=0, column=0, padx=10, pady=10, sticky="nsew")

            self.content_text = ctk.CTkTextbox(self.main_frame, wrap="word")
            self.content_text.configure(font=("Arial", self.font_size))
            self.content_text.grid(
                row=0, rowspan=4, column=1, pady=10, padx=10, sticky="nsew"
            )

            self.text_control = ctk.CTkFrame(self.main_frame)
            self.text_control.grid(
                row=4, column=1, sticky="nsew", pady=10, padx=10)
            self.text_control.grid_rowconfigure(0, weight=1)
            self.text_control.grid_columnconfigure(0, weight=1)
            self.text_control.grid_columnconfigure(1, weight=1)
            self.text_control.grid_columnconfigure(2, weight=1)

            self.font_controls = ctk.CTkFrame(self.text_control)
            self.font_controls.grid_rowconfigure(0, weight=1)
            self.font_controls.grid_columnconfigure(0, weight=1)
            self.font_controls.grid_columnconfigure(1, weight=1)
            ctk.CTkButton(
                self.font_controls, text="A+", width=30, command=self.increase_font_size
            ).grid(row=0, column=0)
            ctk.CTkButton(
                self.font_controls, text="A-", width=30, command=self.decrease_font_size
            ).grid(row=0, column=1)
            self.font_controls.grid(
                row=0, column=2, sticky="nsew", pady=10, padx=10)

            self.scroll_controls = ctk.CTkFrame(self.text_control)
            self.scroll_controls.grid_rowconfigure(0, weight=1)
            self.scroll_controls.grid_rowconfigure(1, weight=1)
            self.scroll_controls.grid_columnconfigure(0, weight=1)
            self.scroll_controls.grid_columnconfigure(1, weight=1)
            self.scroll_percent_label = ctk.CTkLabel(
                self.scroll_controls, text="Reading Progress: 0%"
            )
            self.scroll_percent_label.grid(
                row=0, column=0, padx=5, pady=5, sticky="nsew"
            )
            self.sync_audio_button = ctk.CTkButton(
                self.scroll_controls,
                text="Sync Audio from Reading",
                command=self.sync_audio_to_scroll,
            )
            self.sync_audio_button.grid(
                row=0, column=1, pady=5, padx=5)
            self.sync_checkbox_var = tk.BooleanVar(value=False)
            self.sync_checkbox = ctk.CTkCheckBox(
                self.scroll_controls,
                text="Sync Reading with Audio",
                variable=self.sync_checkbox_var,
                command=self.toggle_sync_button,
            )
            self.sync_checkbox.grid(
                row=1, column=0, columnspan=2, padx=5, pady=5)
            self.scroll_controls.grid(
                row=0, column=1, sticky="nsew", pady=10, padx=10)

            self.textbox_controls = ctk.CTkFrame(self.text_control)
            self.textbox_controls.grid_rowconfigure(0, weight=1)
            self.textbox_controls.grid_columnconfigure(0, weight=1)
            self.textbox_controls.grid_columnconfigure(1, weight=1)
            self.textbox_controls.grid_columnconfigure(2, weight=1)
            self.textbox_controls.grid_columnconfigure(3, weight=1)
            ctk.CTkButton(
                self.textbox_controls, text="<", width=30, command=self.previous_item
            ).grid(row=0, column=0)
            self.docitemnum_label = ctk.CTkLabel(
                self.textbox_controls, text="###", width=60
            )
            self.docitemnum_label.grid(row=0, column=1)
            self.docitemmax_label = ctk.CTkLabel(
                self.textbox_controls, text="/ ###", width=60
            )
            self.docitemmax_label.grid(row=0, column=2)
            # self.currentdocitem = tk.IntVar(value=1)
            ctk.CTkButton(
                self.textbox_controls, text=">", width=30, command=self.next_item
            ).grid(row=0, column=3)

            self.textbox_controls.grid(
                row=0, column=0, sticky="nsew", pady=10, padx=10)

            # self.scope_frame = ctk.CTkFrame(self.main_frame)
            # self.scope_frame.grid_rowconfigure(0, weight=1)
            # self.scope_frame.grid_columnconfigure(0, weight=1)
            # self.scope_frame.grid_columnconfigure(1, weight=1)
            # ctk.CTkLabel(self.scope_frame, text="Synthetization\nScope", font=("Arial", 12, "bold")).grid(row=0, column=0, pady=10)
            self.scope_var = tk.StringVar(value=config.SYNTHETIZATION_LEVEL)
            # ctk.CTkRadioButton(self.scope_frame, text="Full", variable=self.scope_var, value="Full").grid(row=0, column=1, padx=10)
            # ctk.CTkRadioButton(self.scope_frame, text="Extract", variable=self.scope_var, value="Extract").grid(row=0, column=2, padx=10)
            # self.scope_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

            self.cover_frame = ctk.CTkFrame(self.main_frame)
            self.cover_frame.grid_rowconfigure(0, weight=1)
            self.cover_frame.grid_columnconfigure(0, weight=1)
            self.cover_frame.grid_rowconfigure(1, weight=1)
            self.bg_photo = ctk.CTkImage(
                light_image=Image.open(config.BG_IMG), size=(300, 200)
            )
            self.cover_label = ctk.CTkLabel(
                self.cover_frame, text="", compound="center"
            )
            self.cover_label.grid(row=1, column=0, padx=5, pady=5)
            self.cover_label.configure(image=self.bg_photo)
            self.book_title_label = ctk.CTkLabel(
                self.cover_frame, text="", font=("Arial", 14, "bold"), compound="center"
            )
            self.book_title_label.grid(row=0, column=0, padx=5)
            self.cover_frame.grid(row=2, column=0, padx=10,
                                  pady=10, sticky="nsew")

            self.player_frame = ctk.CTkFrame(self.main_frame)
            playimage = utils.resize_image(
                os.path.join(config.IMG_PATH, "play.png"), (48, 48)
            )
            playphoto = ctk.CTkImage(
                light_image=playimage, size=playimage.size)
            pauseimage = utils.resize_image(
                os.path.join(config.IMG_PATH, "pause.png"), (48, 48)
            )
            pausephoto = ctk.CTkImage(
                light_image=pauseimage, size=pauseimage.size)
            stopimage = utils.resize_image(
                os.path.join(config.IMG_PATH, "stop.png"), (48, 48)
            )
            stopphoto = ctk.CTkImage(
                light_image=stopimage, size=stopimage.size)
            self.player_frame.grid(
                row=3, column=0, padx=10, pady=10, sticky="nsew")
            self.play_button = ctk.CTkButton(
                self.player_frame,
                image=playphoto,
                text="",
                command=self.play_epub_audio,
            )
            self.play_button.grid(row=0, column=0, padx=5)
            self.pause_button = ctk.CTkButton(
                self.player_frame, image=pausephoto, text="", command=self.pause_audio
            )

            self.pause_button.grid(row=0, column=1, padx=5)
            self.stop_button = ctk.CTkButton(
                self.player_frame, image=stopphoto, text="", command=self.stop_audio
            )
            self.stop_button.grid(row=0, column=2, padx=5)
            self.audio_progress = ctk.CTkSlider(
                self.player_frame, from_=0, to=1, command=self.seek_audio
            )
            self.audio_progress.set(0)
            self.audio_progress.grid(
                row=1, column=0, columnspan=2, padx=5, sticky="ew")
            self.player_frame.grid_rowconfigure(0, weight=1)
            self.player_frame.grid_rowconfigure(1, weight=1)
            self.player_frame.grid_columnconfigure(0, weight=1)
            self.player_frame.grid_columnconfigure(1, weight=1)
            self.player_frame.grid_columnconfigure(2, weight=1)

            self.audio_time_label = ctk.CTkLabel(
                self.player_frame, text="00:00 / 00:00"
            )
            self.audio_time_label.grid(row=1, column=2, padx=5)
            # self.volume_slider = ctk.CTkSlider(self.player_frame, from_=0, to=1, number_of_steps=20, command=self.set_volume)
            # self.volume_slider.set(1.0)
            # self.volume_slider.grid(row=2, column=0, columnspan=2,padx=5)
            # self.player_frame.grid_rowconfigure(2, weight=1)

            self.end_frame = ctk.CTkFrame(self.main_frame)
            self.end_frame.grid_rowconfigure(0, weight=1)
            self.end_frame.grid_columnconfigure(0, weight=1)
            ctk.CTkButton(
                self.end_frame,
                text="Exit",
                fg_color="red",
                hover_color="darkred",
                command=self.exit_app,
            ).grid(row=0, column=0, pady=10)
            self.end_frame.grid(row=4, column=0, padx=10,
                                pady=10, sticky="nsew")

            ctk.CTkLabel(
                self,
                text=f"{__author__} - gui.henon@gmail.com - Version {__version__} - 2025",
                font=("Arial", 9),
                anchor="e",
                justify="right",
            ).grid(row=1, column=0, padx=5, sticky="se")

            self.text_content = ""
            self.synt_inprogress = False
            self.book_title = ""
            self.book_author = ""
            self.book_items = []
            self.currentdocitem = 0
            self.maxdocitem = 0

            self.audio_play_obj = None
            self.current_audio = None
            self.audio_thread = None
            self.stop_flag = threading.Event()
            self.volume_level = 1.0
            self.audio_duration = 1.0
            self.pause_time = 0
            self.is_paused = False

            # ****** START *****
            if config.CURRENT_EBOOK_PATH:
                self.read_ebook()
            else:
                self.open_ebook()

            self.track_progress_from_seek()

        except Exception as e:
            logging.error(traceback.format_exc())
            logging.error(f"Error during class initiallisation: {str(e)}")

    def open_ebook(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("eBook files", "*.epub;*.mobi;*.azw3")]
        )
        if file_path:
            try:
                self.stop_audio()
                file_path = pathlib.Path(file_path)
                # config.configfile["CURRENT"]["EBOOK_PATH"] = file_path
                config.configfile["CURRENT"] = {
                    "EBOOK_PATH": file_path, "EBOOK_PART": 1}
                config.save_config()
                self.read_ebook()
            except Exception as e:
                logging.error(traceback.format_exc())
                logging.error(f"Failed to open eBook: {str(e)}")

    def read_ebook(self):
        config.load_config()
        config.CURRENT_EBOOK_PATH = config.configfile["CURRENT"]["EBOOK_PATH"]
        file_path = config.configfile["CURRENT"]["EBOOK_PATH"]
        config.CURRENT_EBOOK_PART = int(
            config.configfile["CURRENT"]["EBOOK_PART"])
        try:
            logging.info(f"Loading {file_path}")
            if not os.path.exists(file_path):
                messagebox.showerror(
                    "File missing", f"File {file_path} does not exists")
                return
            self.book = epub.read_epub(
                file_path, {"ignore_ncx": True})
            self.book_title = (
                self.book.get_metadata("DC", "title")[0][0]
                if self.book.get_metadata("DC", "title")
                else "Unknown Title"
            )
            self.book_author = (
                self.book.get_metadata("DC", "creator")[0][0]
                if self.book.get_metadata("DC", "creator")
                else "Unknown Author"
            )
            self.book_title_label.configure(
                text=f"{utils.wrap_text(self.book_title,70)}\n\n{self.book_author}"
            )
            self.text_content = ""
            self.book_items = []

            for item in self.book.get_items():
                logging.debug(item)
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    soup = BeautifulSoup(
                        item.get_body_content(), "html.parser")
                    text = soup.get_text().strip()
                    if text:
                        self.book_items.append({"item": item, "text": text})
                    # self.text_content += text
                    # self.text_content += "\n\n**************************\n\n"
                elif item.get_type() in [ebooklib.ITEM_COVER, ebooklib.ITEM_IMAGE]:
                    clean_file_name = utils.clean_string(
                        f"{self.book_title}_{self.book_author}_{item.file_name}")
                    image_path = os.path.join(
                        config.DOWNLOAD_PATH,
                        clean_file_name,
                    )
                    logging.info(f"Downloading {image_path}")
                    with open(image_path, "wb") as f:
                        f.write(item.content)
                    if (
                        item.get_type() == ebooklib.ITEM_COVER
                        or "cover" in image_path.lower()
                    ):
                        logging.info(f"{image_path} to be used as cover")
                        resized_image = utils.resize_image(
                            image_path, (250, 250))
                        self.bg_photo = ctk.CTkImage(
                            light_image=resized_image, size=resized_image.size
                        )
                        self.cover_label.configure(image=self.bg_photo)
            self.currentdocitem = config.CURRENT_EBOOK_PART
            self.maxdocitem = len(self.book_items)
            self.docitemmax_label.configure(text=f"/ {self.maxdocitem}")
            self.display_text()
        except Exception as e:
            logging.error(traceback.format_exc())
            logging.error(f"Failed to read eBook: {str(e)}")

    def display_text(self):
        try:
            self.text_content = self.book_items[self.currentdocitem - 1]["text"]
            self.content_text.delete("0.0", "end")
            self.content_text.insert("1.0", self.text_content)
            self.docitemnum_label.configure(text=str(self.currentdocitem))
            self.update_scroll_percentage()
        except Exception as e:
            logging.error(traceback.format_exc())
            logging.error(f"Failed to display text {str(e)}")

    def set_currentdocitem(self):
        try:
            self.currentdocitem = int(self.docitemnum_label.cget("text")) + 1
            if self.currentdocitem > self.maxdocitem or self.currentdocitem < 1:
                self.currentdocitem = 1
            self.display_text()
        except Exception as e:
            logging.error(traceback.format_exc())
            logging.error(f"Failed to set current document item {str(e)}")

    def next_item(self):
        try:
            self.stop_audio()
            self.currentdocitem += 1
            if self.currentdocitem > self.maxdocitem:
                self.currentdocitem = 1
            self.display_text()

        except Exception as e:
            logging.error(traceback.format_exc())
            logging.error(f"Failed to move to next item {str(e)}")

    def previous_item(self):
        try:
            self.stop_audio()
            self.currentdocitem -= 1
            if self.currentdocitem < 1:
                self.currentdocitem = self.maxdocitem
            self.display_text()

        except Exception as e:
            logging.error(traceback.format_exc())
            logging.error(f"Failed to move to previous item {str(e)}")

    def update_scroll_percentage(self, event=None):
        try:
            start, end = self.content_text.yview()
            percent_read = int(start * 100)
            self.scroll_percent_label.configure(
                text=f"Reading Progress: {percent_read}%"
            )
        except Exception as e:
            logging.error(f"Failed to update scroll percentage: {str(e)}")
        # Keep polling every 2s
        self.after(2000, self.update_scroll_percentage)

    def play_epub_audio(self):
        try:
            if self.is_paused:
                self.seek_audio(self.pause_time / self.audio_duration)
                self.is_paused = False
                return

            audio_file = f"{self.book_title}_{self.book_author}_{self.scope_var.get()}_{str(self.currentdocitem)}"
            audio_file = f"{utils.clean_string(audio_file)}.wav"
            audio_file = os.path.join(config.DOWNLOAD_PATH, audio_file)
            if os.path.exists(audio_file):
                logging.info(f"Playing {audio_file}")
                self.play_audio(audio_file)
            else:
                logging.warning(
                    f"The file {audio_file} doesn't exists, no audio file played"
                )
                messagebox.showwarning(
                    "No Audio",
                    "Audio File Missing, Run synthetization and try again later",
                )
        except Exception as e:
            logging.error(traceback.format_exc())
            logging.error(f"Failed to play audio: {str(e)}")

    def sync_audio_to_scroll(self):
        try:
            if not self.audio_play_obj:
                self.play_epub_audio()
                time.sleep(2)
            else:
                if not self.audio_play_obj.is_playing():
                    self.play_epub_audio()
                    time.sleep(2)
            # self.play_epub_audio()  # to make sure right audio is played
            start, end = self.content_text.yview()
            percent_read = float(start)
            seek_time = percent_read * self.audio_duration

            # Pass ratio 0.0 - 1.0
            # start 10s before
            logging.info(
                f"Audio synced to {percent_read}% of text: seek_time {seek_time} audio_duration {self.audio_duration}"
            )
            self.seek_audio(percent_read)
            # time.sleep(1)
            # self.pause_audio()

        except Exception as e:
            logging.error(traceback.format_exc())
            logging.error(f"Failed to sync audio: {str(e)}")

    def pause_audio(self):
        if self.audio_play_obj and self.audio_play_obj.is_playing():
            self.audio_play_obj.stop()
            self.pause_progress = self.audio_progress.get()
            self.pause_time = self.pause_progress * self.audio_duration
            self.is_paused = True
            self.stop_flag.set()
            logging.info(f"Paused at {self.pause_time:.2f} seconds")

    def preprocess_text(self, text):
        text = text.lower()  # Lowercase text
        # Define the punctuation to keep
        keep = {".", ",", "?", "!"}
        # Create a translation table that maps unwanted punctuation to None
        translation_table = str.maketrans(
            "", "", "".join(c for c in string.punctuation if c not in keep)
        )
        # Translate the text using the translation table
        text = text.translate(translation_table)
        # Use regex to replace multiple dots with a single dot
        text = re.sub(r"\.{2,}", ".", text)
        text = re.sub(r"\.{2,}", "!", text)
        text = text.replace("*", "")
        # Remove extra spaces, tabs, and new lines
        text = " ".join(text.split())
        return text

    def synthetize_and_play_selection(self):
        try:
            selected_text = self.content_text.get(
                "sel.first", "sel.last").strip()
            if not selected_text:
                raise tk.TclError  # Will be caught below

            logging.info(
                f"Synthetizing selected text, truncated to 400 characters, starting with {selected_text:50}")

            # Preprocess if needed
            # processed_text = self.preprocess_text(selected_text)
            processed_text = selected_text[:400]
            processed_text = processed_text.strip()
            processed_text = " ".join(processed_text.split())
            processed_text = processed_text.replace('"', '')

            # Build a temp filename
            output_path = os.path.join(
                config.DOWNLOAD_PATH, "selected_text.wav")
            output_path = utils.add_timestamp_suffix(output_path)
            # Synthesize using the already loaded StyleTTS2
            if not hasattr(self, "styletts"):

                self.styletts = tts.StyleTTS2(
                    config_path=config.TTS_CONFIG_PATH,
                    model_checkpoint_path=config.TTS_MODEL_CHECKPOINT_PATH
                )

            self.styletts.inference(
                processed_text,
                target_voice_path=config.TTS_TARGET_VOICE_PATH,
                output_wav_file=output_path
            )

            # Normalize and play the audio
            sound = AudioSegment.from_wav(output_path)
            normalized = effects.normalize(sound)
            output_path_normalized = utils.add_suffix(
                output_path, "normalized")
            normalized.export(output_path_normalized, format="wav")

            self.play_audio(output_path)

        except tk.TclError:
            messagebox.showwarning(
                "No Selection", "Please select some text in the reader first.")
        except Exception as e:
            logging.error(traceback.format_exc())
            messagebox.showerror(
                "Error", f"Failed to synthetize selection:\n{str(e)}")

    def convert_to_audio_callthread(self):
        try:
            if not self.synt_inprogress:
                threading.Thread(
                    target=self.convert_to_audio_threaded, daemon=True
                ).start()
            else:
                logging.warning(
                    "a Synthetization is already in progress, skipping action"
                )
        except Exception as e:
            logging.error(traceback.format_exc())
            logging.error(f"Failed to launch thread: {str(e)}")

    def convert_to_audio_threaded(self):
        logging.info("Starting synthetization")
        try:
            self.synt_button.configure(
                fg_color="Green", hover_color="LimeGreen", text="Loading synthetizer")
            self.synt_inprogress = True

            # in styletts/models.py, updated torch.load(model_path, map_location='cpu',weights_only=False)
            logging.debug(
                f"Styletts2 imported from {inspect.getfile(tts)}")
            logging.debug(
                f"model {config.TTS_MODEL_CHECKPOINT_PATH}, config {config.TTS_CONFIG_PATH}")
            # Initialize the StyleTTS2 model with the loaded configuration
            if not hasattr(self, "styletts"):

                self.styletts = tts.StyleTTS2(
                    config_path=config.TTS_CONFIG_PATH,
                    model_checkpoint_path=config.TTS_MODEL_CHECKPOINT_PATH
                )
        except Exception as e:
            logging.error(f"Failed to load synthetizer: {str(e)}")
            logging.error(traceback.format_exc())

        num = 1
        try:
            for item in self.book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    logging.info(f"Item: {item}")
                    soup = BeautifulSoup(
                        item.get_body_content(), "html.parser")
                    text = soup.get_text().strip()
                    text = " ".join(text.split())
                    text = text.replace('"', '')
                    if text:
                        if not self.convert_to_audio(
                            num, text
                        ):  # if synthetization failed with raw text, preprocess it to remove special character and retry
                            logging.warning("Preprocessing text and try again")
                            text = self.preprocess_text(text)
                            self.convert_to_audio(num, text)
                        num += 1

            self.synt_inprogress = False
            self.synt_button.configure(
                fg_color="#3a7ebf",
                hover_color="#325882",
                text="Synthetization completed",
            )

        except Exception as e:
            logging.error(f"Failed to launch synthetization: {str(e)}")
            logging.error(traceback.format_exc())

    def convert_to_audio(self, num, text):
        try:
            # logging.info("Loading synthetizer modules and models")
            # import torch
            # from TTS.api import tts

            text_small = text.strip()
            text_small = " ".join(text_small.split())
            text_small = text_small[:50]
            logging.info(
                f"Synthetizing item {num}, text starts with: {text_small} ")
            self.synt_button.configure(text=f"Synthetizing part {num}")
            if text:

                output_filename = f"{self.book_title}_{self.book_author}_{self.scope_var.get()}_{str(num)}"
                output_filename = f"{utils.clean_string(output_filename)}.wav"
                output_path = os.path.join(
                    config.DOWNLOAD_PATH, output_filename)
                utils.ensure_directories(output_path)
                if not os.path.exists(output_path):
                    text_to_convert = (
                        text if self.scope_var.get() == "Full" else text[:1000]
                    )
                    # tts = TTS(model_name=config.TTS_MODEL, progress_bar=False).to(
                    #     torch.device("cpu")
                    # )

                    # tts.tts_to_file(text=text_to_convert,
                    #                 file_path=output_path)
                    logging.debug(f"Output Wav File is {output_path}")
                    out = self.styletts.inference(
                        text_to_convert, target_voice_path=config.TTS_TARGET_VOICE_PATH, output_wav_file=output_path)
                    # out = self.styletts.inference(
                    #    "Bonjour Guillaume, Comment vas-tu ?", target_voice_path=config.TTS_TARGET_VOICE_PATH, output_wav_file="test.wav")
                    # waveform = self.styletts.infer(text=text_to_convert)
                    # torchaudio.save(output_path, waveform, 24000)
                    # sound = AudioSegment.from_wav(output_path)
                    # normalized_sound = effects.normalize(sound)
                    # output_path = utils.add_suffix(output_path, "normalized")
                    # normalized_sound.export(output_path, format="wav")
                    logging.info(f"Audio saved as {output_path}")
                else:
                    logging.warning(
                        f"The file {output_path} exists, skipping synthetization."
                    )
            else:
                logging.error("No text available for conversion")
            return num
        except Exception as e:
            logging.error(traceback.format_exc())
            logging.error(
                f"Failed to generate audio: {str(e)}\nTarget file was {output_path}"
            )
        return None

    def increase_font_size(self):
        self.font_size += 1
        self.content_text.configure(font=("Arial", self.font_size))

    def decrease_font_size(self):
        self.font_size = max(6, self.font_size - 1)
        self.content_text.configure(font=("Arial", self.font_size))

    def play_audio(self, wavfile):
        try:
            if self.audio_play_obj and self.audio_play_obj.is_playing():
                self.audio_play_obj.stop()
                self.stop_flag.set()
                time.sleep(0.1)
            if not wavfile:
                messagebox.showwarning("No Audio", "Audio File Missing")
                return
            latest_audio = wavfile
            sound = AudioSegment.from_wav(latest_audio)
            # sound = effects.normalize(sound)
            sound = sound.set_channels(2)
            sound = sound.set_sample_width(2)  # 16-bit PCM
            sound = sound.set_frame_rate(44100)  # Optional, for compatibility
            self.current_audio = sound
            self.audio_duration = sound.duration_seconds
            self.stop_flag.clear()
            self.audio_play_obj = sa.play_buffer(
                sound.raw_data,
                num_channels=sound.channels,
                bytes_per_sample=sound.sample_width,
                sample_rate=sound.frame_rate,
            )
            self.start = time.time()
            self.start_seek = 0
            logging.debug(
                f"Play start at {self.start} duration {self.audio_duration}")
        except Exception as e:
            logging.error(traceback.format_exc())
            logging.error(f"Failed to play audio: {str(e)}")

    def stop_audio(self):
        #  if self.audio_play_obj and self.audio_play_obj.is_playing():
        if self.audio_play_obj:
            self.audio_play_obj.stop()
            self.stop_flag.set()
            self.audio_progress.set(0)
            self.audio_time_label.configure(text="00:00 / 00:00")

    def update_audio_time_label(self, elapsed, total):
        def format_time(t):
            mins, secs = divmod(int(t), 60)
            return f"{mins:02}:{secs:02}"

        self.audio_time_label.configure(
            text=f"{format_time(elapsed)} / {format_time(total)}"
        )

    def seek_audio(self, val):
        if not self.current_audio:
            logging.warning(
                "Seek audio only available when an audio is loaded")
            messagebox.showwarning(
                "Warning", "Seek audio only available when an audio is loaded"
            )
            return
        try:
            seek_time = float(val) * self.audio_duration
            logging.info(
                f"Audio seek to seek_time {seek_time} audio_duration {self.audio_duration}"
            )
            self.stop_audio()
            segment = self.current_audio[int(seek_time * 1000):]
            segment += 20 * (self.volume_level - 1)
            self.audio_play_obj = sa.play_buffer(
                segment.raw_data,
                num_channels=segment.channels,
                bytes_per_sample=segment.sample_width,
                sample_rate=segment.frame_rate,
            )
            self.stop_flag.clear()
            self.start = time.time()
            self.start_seek = seek_time

        except Exception as e:
            logging.error(f"Error during seeking: {str(e)}")
            logging.error(traceback.format_exc())

    def track_progress_from_seek(self):
        # start = time.time()
        if (
            not self.stop_flag.is_set()
            and self.audio_play_obj
            and self.audio_play_obj.is_playing()
        ):
            elapsed = (time.time() - self.start) + self.start_seek
            progress = min(elapsed / self.audio_duration, 1.0)
            self.audio_progress.set(progress)
            self.update_audio_time_label(elapsed, self.audio_duration)
            # Still scroll text while paused if checkbox is checked
            if self.sync_checkbox_var.get():
                self.content_text.yview_moveto(progress - 0.05)
        else:
            if self.is_paused:
                elapsed = self.pause_progress * self.audio_duration
                progress = min(elapsed / self.audio_duration, 1.0)
                self.audio_progress.set(progress)
                self.update_audio_time_label(elapsed, self.audio_duration)
                # Still scroll text while paused if checkbox is checked
                if self.sync_checkbox_var.get():
                    self.content_text.yview_moveto(progress - 0.05)
            else:
                elapsed = 0

        self.after(2000, self.track_progress_from_seek)

    def toggle_sync_button(self):
        if self.sync_checkbox_var.get():
            self.sync_audio_button.configure(state="disabled")
        else:
            self.sync_audio_button.configure(state="normal")

    def set_volume(self, val):
        self.volume_level = float(val)
        logging.info(f"Volume set to: {self.volume_level}")

    def exit_app(self):
        logging.info("Exiting Application")
        self.quit()

    def confirm_close(self):
        if messagebox.askyesno("Confirm Close", "Are you sure you want to close?"):
            self.exit_app()


if __name__ == "__main__":
    config.initialize()
    utils.init_logging()
    logging.getLogger(__name__)
    logging.info("Application starts")
    app = epubTextToSpeech()
    app.mainloop()
