import logging, threading, traceback, time
import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import Image
from pathlib import Path
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import config, utils

__author__ = "Guillaume HENON"
__version__ = "0.1"

class epubTextToSpeech(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("epubTextToSpeech")
        width, height = 600, 680
        screen_width, screen_height = self.winfo_screenwidth(), self.winfo_screenheight()
        x, y = (screen_width / 2) - (width / 2), (screen_height / 2) - (height / 2)
        self.geometry(f'{width}x{height}+{int(x)}+{int(y)}')
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.protocol("WM_DELETE_WINDOW", self.confirm_close)
        self.iconphoto(True, tk.PhotoImage(file=config.BG_IMG))
        
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        self.init_frame = ctk.CTkFrame(self.main_frame)
        self.bg_photo = ctk.CTkImage(light_image=Image.open(config.BG_IMG), size=(self.init_frame.cget("width")-10, (self.init_frame.cget("width")-10)*200/250))
        ctk.CTkLabel(self.init_frame, image=self.bg_photo, text="", compound="left").grid(row=0, column=1, rowspan=3)
        ctk.CTkLabel(self.init_frame, text="epubTextToSpeech", font=("Roboto", 24, "bold"), text_color="#1565c0").grid(row=0, column=0, pady=5)
        self.init_frame.grid(row=0, column=0, padx=10, pady=5, sticky="nsew")
        
        self.controls_frame = ctk.CTkFrame(self.main_frame)
        ctk.CTkButton(self.controls_frame, text="Open eBook", command=self.open_ebook).grid(row=0, column=0, pady=5)
        self.controls_frame.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")
        
        self.status_frame = ctk.CTkFrame(self.main_frame)
        self.status_text = ctk.CTkTextbox(self.status_frame, height=30)
        self.status_text.grid(row=0, column=0, pady=10, padx=5, sticky="nsew")
        self.progress_var = ctk.DoubleVar()
        self.progress_bar = ctk.CTkProgressBar(self.status_frame, variable=self.progress_var, width=350)
        self.progress_bar.grid(row=1, column=0, pady=10)
        self.progress_bar.set(0)
        self.status_frame.grid(row=2, column=0, padx=10, pady=5, sticky="nsew")

      
        self.content_frame=ctk.CTkFrame(self.main_frame)
        self.content_text = ctk.CTkTextbox(self.content_frame, height=200, width=350)
        self.content_text.grid(row=0, column=0, pady=10, padx=5, sticky="nsew")
        self.content_frame.grid(row=3, column=0, padx=10, pady=5, sticky="nsew")
        
        self.end_frame = ctk.CTkFrame(self.main_frame)
        ctk.CTkButton(self.end_frame, text="Exit", fg_color="red", hover_color="darkred", command=self.exit_app).grid(row=1, column=0, pady=5)
        self.end_frame.grid(row=4, column=0, columnspan=2,padx=10, pady=5, sticky="nsew")
        
        ctk.CTkLabel(self.main_frame, text=f"{__author__} - gui.henon@gmail.com - Version {__version__} - 2025", font=("Arial", 9), anchor="e", justify="right").grid(row=5, column=0, sticky="ne")
        self.update_ui()
    
    def open_ebook(self):
        file_path = filedialog.askopenfilename(filetypes=[("eBook files", "*.epub;*.mobi;*.azw3")])
        if file_path:
            try:
                book = epub.read_epub(file_path)
                config.status_str=f"Opened: {Path(file_path).name}\n"
                logging.info(f"Metadata: {book.metadata}")
                logging.info(f"Spine: {book.spine}")
                logging.info(f"Table of Content: {book.toc}")
                for item in book.get_items():
                    logging.debug(item)
                    logging.debug(f"%s", item.get_content())
                    if item.get_type() == ebooklib.ITEM_DOCUMENT:
                        logging.info(f"%s", item.get_body_content())
                        soup = BeautifulSoup(item.get_body_content(), "html.parser")
                        text = soup.get_text()
                        self.content_text.insert("end", text[:500]+"\n")                 
                
            except Exception as e:
                logging.error(traceback.format_exc())
                messagebox.showerror("Error", f"Failed to open eBook: {str(e)}")
    
    def update_ui(self):
        self.status_text.delete("1.0", "end")
        self.status_text.insert("1.0", config.status_str)
        if config.progress_num >= 0:
            if self.progress_bar.cget("mode") == "indeterminate":
                self.progress_bar.stop()
                self.progress_bar.configure(mode="determinate")
            self.progress_var.set(config.progress_num / (config.progress_tot or 1))
        self.after(2000, self.update_ui)
    
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
