import logging, threading, traceback, time, os
import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox,filedialog
from PIL import Image, ImageTk
from pathlib import Path
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import config, utils
import string
import re

from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

print("Importing modules and launching application...")
# CoquiTTS https://pypi.org/project/coqui-tts/
import torch
from TTS.api import TTS

lock_synt = threading.Lock()

__author__ = "Guillaume HENON"
__version__ = "0.1"

class epubTextToSpeech(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("epubTextToSpeech")
        width, height = 1000, 600
        screen_width, screen_height = self.winfo_screenwidth(), self.winfo_screenheight()
        x, y = (screen_width / 2) - (width / 2), (screen_height / 2) - (height / 2)
        self.geometry(f'{width}x{height}+{int(x)}+{int(y)}')
        self.protocol("WM_DELETE_WINDOW", self.confirm_close)
        self.iconphoto(True, tk.PhotoImage(file=config.BG_IMG))
        
        self.grid_rowconfigure(0, weight=10)
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(1, weight=1)
        self.main_frame.grid_rowconfigure(2, weight=3)
        self.main_frame.grid_rowconfigure(3, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(1, weight=3)
        self.main_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
       
        
        self.controls_frame = ctk.CTkFrame(self.main_frame)
        self.controls_frame.grid_rowconfigure(0, weight=1)
        self.controls_frame.grid_columnconfigure(0, weight=1)
        self.controls_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkButton(self.controls_frame, text="Open eBook", command=self.open_ebook).grid(row=0, column=0, pady=10, padx=10)
        self.synt_button=ctk.CTkButton(self.controls_frame, text="Synthetize Audio", command=self.convert_to_audio_callthread)
        self.synt_button.grid(row=0, column=1, padx=10, pady=10)
        self.controls_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        self.content_text = ctk.CTkTextbox(self.main_frame)
        self.content_text.grid(row=0, rowspan=4,column=1, pady=10, padx=10, sticky="nsew")
        
        self.scope_frame = ctk.CTkFrame(self.main_frame)
        self.scope_frame.grid_rowconfigure(0, weight=1)
        self.scope_frame.grid_columnconfigure(0, weight=1)
        self.scope_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(self.scope_frame, text="Synthetization\nScope", font=("Arial", 12, "bold")).grid(row=0, column=0, pady=10)
        self.scope_var = tk.StringVar(value="Extract")
        ctk.CTkRadioButton(self.scope_frame, text="Full", variable=self.scope_var, value="Full").grid(row=0, column=1, padx=10)
        ctk.CTkRadioButton(self.scope_frame, text="Extract", variable=self.scope_var, value="Extract").grid(row=0, column=2, padx=10)
        self.scope_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        
        self.cover_frame = ctk.CTkFrame(self.main_frame)
        self.cover_frame.grid_rowconfigure(0, weight=1)
        self.cover_frame.grid_columnconfigure(0, weight=1)
        self.cover_frame.grid_columnconfigure(1, weight=1)
        self.bg_photo = ctk.CTkImage(light_image=Image.open(config.BG_IMG), size=(300, 200))
        self.cover_label=ctk.CTkLabel(self.cover_frame, text="", compound="center")
        self.cover_label.grid(row=0, column=0,padx=5)
        self.cover_label.configure(image=self.bg_photo)
        self.book_title_label=ctk.CTkLabel(self.cover_frame, text="", compound="center")
        self.book_title_label.grid(row=0, column=1,padx=0)
        self.cover_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")
        #taille arbre
               
        self.end_frame = ctk.CTkFrame(self.main_frame)
        self.end_frame.grid_rowconfigure(0, weight=1)
        self.end_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkButton(self.end_frame, text="Exit", fg_color="red", hover_color="darkred", command=self.exit_app).grid(row=0, column=0, pady=10)
        self.end_frame.grid(row=3, column=0,padx=10, pady=10, sticky="nsew")
        
        ctk.CTkLabel(self, text=f"{__author__} - gui.henon@gmail.com - Version {__version__} - 2025", font=("Arial", 9), anchor="e", justify="right").grid(row=1, column=0,padx=5, sticky="se")
        
        self.text_content = ""
        self.synt_inprogress=False
        self.book_title = ""
        self.book_author = ""
        
    def open_ebook(self):
        file_path = filedialog.askopenfilename(filetypes=[("eBook files", "*.epub;*.mobi;*.azw3")])
        if file_path:
            try:
                logging.info(f"Loading {file_path}")
                self.book = epub.read_epub(file_path)
                self.book_title = self.book.get_metadata("DC", "title")[0][0] if self.book.get_metadata("DC", "title") else "Unknown Title"
                self.book_author = self.book.get_metadata("DC", "creator")[0][0] if self.book.get_metadata("DC", "creator") else "Unknown Author"
                self.book_title_label.configure(text=f"{self.book_title}\n\n{self.book_author}")
                self.text_content = ""
                
                for item in self.book.get_items():
                    logging.debug(item)
                    if item.get_type() == ebooklib.ITEM_DOCUMENT:
                        soup = BeautifulSoup(item.get_body_content(), "html.parser")
                        text = soup.get_text().strip()
                        self.text_content += text
                        self.text_content += "\n\n**************************\n\n"
                    elif item.get_type() in [ebooklib.ITEM_COVER, ebooklib.ITEM_IMAGE]:
                        image_path = os.path.join(config.DOWNLOAD_PATH, f"{self.book_title}_{self.book_author}_{item.file_name.replace('/', '_')}")
                        logging.info(f"Downloading {image_path}")
                        with open(image_path, "wb") as f:
                            f.write(item.content)
                        if item.get_type()==ebooklib.ITEM_COVER or "cover" in image_path.lower():
                            logging.info(f"{image_path} to be used as cover")
                            resized_image = utils.resize_image(image_path,(200,200))
                            self.bg_photo = ctk.CTkImage(light_image=resized_image, size=resized_image.size)
                            self.cover_label.configure(image=self.bg_photo)
                self.content_text.insert("1.0", self.text_content )
            except Exception as e:
                logging.error(traceback.format_exc())
                logging.error(f"Failed to open eBook: {str(e)}")
                
    def preprocess_text(self,text):
        text = text.lower()  # Lowercase text
        # Define the punctuation to keep
        keep = {'.', ',','?','!'}
        # Create a translation table that maps unwanted punctuation to None
        translation_table = str.maketrans('', '', ''.join(c for c in string.punctuation if c not in keep))
        # Translate the text using the translation table
        text= text.translate(translation_table)
        # Use regex to replace multiple dots with a single dot
        text= re.sub(r'\.{2,}', '.', text)
        text= re.sub(r'\.{2,}', '!', text)
        text=text.replace('*','')
        text = " ".join(text.split())  # Remove extra spaces, tabs, and new lines
        return text
    
    def convert_to_audio_callthread(self):
        try:
            if self.synt_inprogress==False:
                threading.Thread(target=self.convert_to_audio_threaded,daemon=True).start()
            else:
                logging.warning("a Synthetization is already in progress, skipping action")
        except Exception as e:
            logging.error(traceback.format_exc())
            logging.error(f"Failed to launch thread: {str(e)}")
            
    def convert_to_audio_threaded(self):
        logging.info(f"Starting synthetization")
        self.synt_button.configure(fg_color="Red")
        self.synt_inprogress=True
        num=1
        try:
            for item in self.book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    logging.info(f"Item: {item}")
                    soup = BeautifulSoup(item.get_body_content(), "html.parser")
                    text = soup.get_text().strip()
                   # text= utils.preprocess_text(text)  #   to help tokenisation and avoid synthetizer to fail
                    #self.text_content = text
                    text=text.replace('=','_')  # to  avoid synthetizer to 
                    text=text.replace('»',' ')
                    if text:
                        if not self.convert_to_audio(num,text):  # if synthetization failed with raw text, preprocess it to remove special character and retry
                            logging.warning("Preprocessing text and try again")
                            text=self.preprocess_text(text)
                            self.convert_to_audio(num,text)
                        num+=1
                            
            self.synt_inprogress=False
            self.synt_button.configure(fg_color="Blue")
                
        except Exception as e:
            logging.error(f"Failed to launch synthetization: {str(e)}")
            logging.error(traceback.format_exc())
            
    
    def convert_to_audio(self,num,text):
        try:
            text_small=text.strip()
            text_small=" ".join(text_small.split())
            text_small=text_small[:50]
            logging.info(f"Synthetizing item {num}, text: {text_small} ")
            if text:
                #tts = TTS(model_name=config.TTS_MODEL, progress_bar=False).to(torch.device("cpu"))
                output_filename = f"{self.book_title}_{self.book_author}_{self.scope_var.get()}_{str(num)}"
                output_filename=f"{utils.clean_string(output_filename)}.wav"
                output_path = os.path.join(config.DOWNLOAD_PATH, output_filename)
                utils.ensure_directories(output_path)
                if not os.path.exists(output_path):
                    text_to_convert = text if self.scope_var.get() == "Full" else text[:1000]
                    tts = TTS(model_name=config.TTS_MODEL, progress_bar=False).to(torch.device("cpu"))
                    tts.tts_to_file(text=text_to_convert, file_path=output_path)
                    logging.info(f"Audio saved as {output_path}")
                else:
                    logging.warning(f"The file {output_path} exists, skipping synthetization.") 
            else:
                logging.error("No text available for conversion")
            return num
        except Exception as e:
            logging.error(traceback.format_exc())
            logging.error(f"Failed to generate audio: {str(e)}\nTarget file was {output_path}")
        return None
                    
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
