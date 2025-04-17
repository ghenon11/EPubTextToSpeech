import simpleaudio as sa
from pydub import AudioSegment
import re
import string
import utils
import config
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
import torch
from TTS.api import TTS
print("Importing modules and launching application...")


# CoquiTTS https://pypi.org/project/coqui-tts/

# img: https://www.freeiconspng.com/
# tkinter colors: https://www.askpython.com/wp-content/uploads/2022/10/Tkinter-colour-list.png.webp

# lock_synt = threading.Lock()

__author__ = "Guillaume HENON"
__version__ = "0.5"


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
            width, height = screen_width - 100, screen_height - 200
            x, y = (screen_width / 2) - \
                (width / 2), (screen_height / 2) - (height / 2)
            self.geometry(f"{width}x{height}+{int(x)}+{int(y)}")
            self.protocol("WM_DELETE_WINDOW", self.confirm_close)
            self.iconphoto(True, tk.PhotoImage(file=config.BG_IMG))
            self.font_size = 16
            self.grid_rowconfigure(0, weight=20)
            self.grid_rowconfigure(1, weight=1)
            self.grid_columnconfigure(0, weight=1)

            self.main_frame = ctk.CTkFrame(self)
            self.main_frame.grid_rowconfigure(0, weight=1)
            self.main_frame.grid_rowconfigure(1, weight=0)
            self.main_frame.grid_rowconfigure(2, weight=3)
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
            ctk.CTkButton(
                self.controls_frame, text="Open eBook", command=self.open_ebook
            ).grid(row=0, column=0, pady=10, padx=10)
            self.synt_button = ctk.CTkButton(
                self.controls_frame,
                text="Synthetize Audio",
                command=self.convert_to_audio_callthread,
            )
            self.synt_button.grid(row=0, column=1, padx=10, pady=10)
            self.controls_frame.grid(
                row=0, column=0, padx=10, pady=10, sticky="nsew")

            self.content_text = ctk.CTkTextbox(self.main_frame)
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
            self.scope_var = tk.StringVar(value="Full")
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
            self.stop_button = ctk.CTkButton(
                self.player_frame, image=stopphoto, text="", command=self.stop_audio
            )
            self.stop_button.grid(row=0, column=1, padx=5)
            self.audio_progress = ctk.CTkSlider(
                self.player_frame, from_=0, to=1, command=self.seek_audio
            )
            self.audio_progress.set(0)
            self.audio_progress.grid(row=1, column=0, padx=5, sticky="ew")
            self.player_frame.grid_rowconfigure(0, weight=1)
            self.player_frame.grid_rowconfigure(1, weight=1)
            self.player_frame.grid_columnconfigure(0, weight=1)
            self.player_frame.grid_columnconfigure(1, weight=1)

            self.audio_time_label = ctk.CTkLabel(
                self.player_frame, text="00:00 / 00:00"
            )
            self.audio_time_label.grid(row=1, column=1, padx=5)
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

            # ****** START *****
            if config.CURRENT_EBOOK_PATH:
                self.read_ebook()
            else:
                self.open_ebook()

        except Exception as e:
            logging.error(traceback.format_exc())
            logging.error(f"Error during class initiallisation: {str(e)}")

    def open_ebook(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("eBook files", "*.epub;*.mobi;*.azw3")]
        )
        if file_path:
            try:
                config.configfile["CURRENT"]["EBOOK_PATH"] = file_path
                config.save_config()
                self.read_ebook()
            except Exception as e:
                logging.error(traceback.format_exc())
                logging.error(f"Failed to open eBook: {str(e)}")

    def read_ebook(self):
        config.load_config()
        config.CURRENT_EBOOK_PATH = config.configfile["CURRENT"]["EBOOK_PATH"]
        config.CURRENT_EBOOK_PART = int(
            config.configfile["CURRENT"]["EBOOK_PART"])
        try:
            logging.info(f"Loading {config.CURRENT_EBOOK_PATH}")
            self.book = epub.read_epub(config.CURRENT_EBOOK_PATH)
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
                text=f"{self.book_title}\n\n{self.book_author}"
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
                        self.book_items.append(
                            {"item": item, "text": text})
                    # self.text_content += text
                    # self.text_content += "\n\n**************************\n\n"
                elif item.get_type() in [ebooklib.ITEM_COVER, ebooklib.ITEM_IMAGE]:
                    image_path = os.path.join(
                        config.DOWNLOAD_PATH,
                        f"{self.book_title}_{self.book_author}_{item.file_name.replace('/', '_')}",
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
            self.currentdocitem += 1
            if self.currentdocitem > self.maxdocitem:
                self.currentdocitem = 1
            self.display_text()
        except Exception as e:
            logging.error(traceback.format_exc())
            logging.error(f"Failed to move to next item {str(e)}")

    def previous_item(self):
        try:
            self.currentdocitem -= 1
            if self.currentdocitem < 1:
                self.currentdocitem = self.maxdocitem
            self.display_text()
        except Exception as e:
            logging.error(traceback.format_exc())
            logging.error(f"Failed to move to previous item {str(e)}")

    def play_epub_audio(self):
        try:
            audio_file = f"{self.book_title}_{self.book_author}_{self.scope_var.get()}_{str(self.currentdocitem)}"
            audio_file = f"{utils.clean_string(audio_file)}.wav"
            audio_file = os.path.join(config.DOWNLOAD_PATH, audio_file)
            if os.path.exists(audio_file):
                self.play_audio(audio_file)
            else:
                logging.warning(
                    f"The file {audio_file} doesn't exists, no audio file played"
                )
                messagebox.showwarning(
                    "No Audio", "Audio File Missing, maybe currently under synthetization\nTry again later")
        except Exception as e:
            logging.error(traceback.format_exc())
            logging.error(f"Failed to play audio: {str(e)}")

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
        self.synt_button.configure(fg_color="Red", hover_color="OrangeRed")
        self.synt_inprogress = True
        num = 1
        try:
            for item in self.book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    logging.info(f"Item: {item}")
                    soup = BeautifulSoup(
                        item.get_body_content(), "html.parser")
                    text = soup.get_text().strip()
                    # text= utils.preprocess_text(text)  #   to help tokenisation and avoid synthetizer to fail
                    # self.text_content = text
                    text = text.replace("=", "_")
                    text = text.replace("»", " ")
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
                fg_color="#3a7ebf", hover_color="#325882")

        except Exception as e:
            logging.error(f"Failed to launch synthetization: {str(e)}")
            logging.error(traceback.format_exc())

    def convert_to_audio(self, num, text):
        try:
            text_small = text.strip()
            text_small = " ".join(text_small.split())
            text_small = text_small[:50]
            logging.info(f"Synthetizing item {num}, text: {text_small} ")
            if text:
                # tts = TTS(model_name=config.TTS_MODEL, progress_bar=False).to(torch.device("cpu"))
                output_filename = f"{self.book_title}_{self.book_author}_{self.scope_var.get()}_{str(num)}"
                output_filename = f"{utils.clean_string(output_filename)}.wav"
                output_path = os.path.join(
                    config.DOWNLOAD_PATH, output_filename)
                utils.ensure_directories(output_path)
                if not os.path.exists(output_path):
                    text_to_convert = (
                        text if self.scope_var.get() == "Full" else text[:1000]
                    )
                    tts = TTS(model_name=config.TTS_MODEL, progress_bar=False).to(
                        torch.device("cpu")
                    )
                    tts.tts_to_file(text=text_to_convert,
                                    file_path=output_path)
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
            sound += 20 * (self.volume_level - 1)
            self.current_audio = sound
            self.audio_duration = sound.duration_seconds
            self.stop_flag.clear()
            self.audio_play_obj = sa.play_buffer(
                sound.raw_data,
                num_channels=sound.channels,
                bytes_per_sample=sound.sample_width,
                sample_rate=sound.frame_rate,
            )
            self.audio_thread = threading.Thread(
                target=self.track_progress_from_seek, args=(0,), daemon=True
            )
            self.audio_thread.start()
        except Exception as e:
            logging.error(f"Failed to play audio: {str(e)}")

    def stop_audio(self):
        #  if self.audio_play_obj and self.audio_play_obj.is_playing():
        if self.audio_play_obj:
            self.audio_play_obj.stop()
            self.stop_flag.set()
            self.audio_progress.set(0)
            self.audio_time_label.configure(text="00:00 / 00:00")

    # def track_progress(self):
    #     start = time.time()
    #     while (
    #         not self.stop_flag.is_set()
    #         and self.audio_play_obj
    #         and self.audio_play_obj.is_playing()
    #     ):
    #         elapsed = time.time() - start
    #         progress = min(elapsed / self.audio_duration, 1.0)
    #         self.audio_progress.set(progress)
    #         self.update_audio_time_label(elapsed, self.audio_duration)
    #         time.sleep(0.5)

    def update_audio_time_label(self, elapsed, total):
        def format_time(t):
            mins, secs = divmod(int(t), 60)
            return f"{mins:02}:{secs:02}"

        self.audio_time_label.configure(
            text=f"{format_time(elapsed)} / {format_time(total)}"
        )

    def seek_audio(self, val):
        if not self.current_audio:
            return
        try:
            seek_time = float(val) * self.audio_duration
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
            self.audio_thread = threading.Thread(
                target=self.track_progress_from_seek, args=(
                    seek_time,), daemon=True
            )
            self.audio_thread.start()
        except Exception as e:
            logging.error(f"Error during seeking: {str(e)}")

    def track_progress_from_seek(self, start_seek):
        start = time.time()
        while (
            not self.stop_flag.is_set()
            and self.audio_play_obj
            and self.audio_play_obj.is_playing()
        ):
            elapsed = (time.time() - start) + start_seek
            progress = min(elapsed / self.audio_duration, 1.0)
            self.audio_progress.set(progress)
            self.update_audio_time_label(elapsed, self.audio_duration)
            time.sleep(0.1)

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
