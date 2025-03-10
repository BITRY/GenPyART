#!/usr/bin/env python3
"""
GenPyART Generative Art V1 Final - Extended Version with Advanced Features
==================================================================

This application is an enhanced generative art tool featuring:
  • A top menubar with File, Effects, Settings, and Help menus.
  • A left control panel (400px wide, scrollable, arranged in two columns).
  • A right canvas that resizes to fill its container.
  • Adjustable animation speed (1–5000 ms) and auto‑clear threshold (1–10000 frames).
  • Drawing modes: Continuous or Burst (with adjustable burst count).
  • Special Modes: Normal vs. UltraRandom (randomizing drawing options) and Chaos Mode.
  • ML Mode: with training (dummy feedback collection), ML evaluation (using a NIMA model if available, else dummy),
    and an enhancement effect (boosting color, contrast, and sharpness).
  • Session logging, settings export/import, and a history of saved images.
  • A menu option to set a custom canvas size.
  • (Optional) An API endpoint (if Flask is installed) available at http://localhost:5000/api/take_screenshot.
  
Dependencies:
  – Pillow: pip install pillow
  – (Optional) TensorFlow: pip install tensorflow (and a pre‑trained NIMA model file "nima_model.h5")
  – (Optional) Flask: pip install flask
  
This code is organized in a class‐based structure.
"""

import tkinter as tk
from tkinter import simpledialog, colorchooser, messagebox, filedialog
import random
import time
import os
import json
import datetime

# ---------- Pillow Imports -----------
try:
    from PIL import Image, ImageGrab, ImageEnhance
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# ---------- TensorFlow Imports -----------
try:
    from tensorflow.keras.models import load_model
    import numpy as np
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# ---------- Flask Imports -----------
try:
    from flask import Flask, jsonify
    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False

# ---------- Helper Function for Random Color ----------
def random_color():
    # Returns a random hex color.
    return "#%06x" % random.randint(0, 0xFFFFFF)

# ============================================================
# Class Definition for the Generative Art Application
# ============================================================
class GenerativeArtApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Generative Art V1 Final - Extended Version")
        self.root.geometry("1300x700")
        self.log_file = "session.log"
        self.history_list = []  # To store saved image filenames

        # ---------------- Font Settings ----------------
        self.compact_font = ("Arial", 10)
        self.header_font = ("Arial", 16, "bold")

        # ---------------- Tkinter Variables ----------------
        self.ml_mode = tk.BooleanVar(self.root, value=False)
        self.ml_training = tk.BooleanVar(self.root, value=False)
        self.special_mode = tk.StringVar(self.root, value="Normal")
        self.chaos_mode = tk.BooleanVar(self.root, value=False)
        self.pause_var = tk.BooleanVar(self.root, value=False)

        # ---------------- Global State ----------------
        self.CANVAS_WIDTH = 800
        self.CANVAS_HEIGHT = 600
        self.paused = False
        self.frame_count = 0
        self.draw_history = []
        self.meme_text_id = None
        self.watermark_id = None

        # ---------------- Settings Dictionary ----------------
        self.SETTINGS = {
            "animation_speed": 50,
            "auto_clear_threshold": 300,
            "drawing_mode": "continuous",
            "burst_count": 5,
            "min_shape_size": 10,
            "max_shape_size": 100,
            "outline_only": False,
            "mirror_drawing": False,
            "grid_overlay": False,
            "bg_color": "#000000",
            "custom_palette": "",
            "watermark_enabled": False,
            "watermark_text": "My Watermark",
            "meme_text": "",
            "meme_font": "Impact",
            "meme_font_size": 32,
            "random_seed": None,
        }
        self.SHAPE_TYPES = {
            "circle": True,
            "line": True,
            "arc": True,
            "rectangle": True,
            "triangle": True,
        }

        # ---------------- ML Training ----------------
        self.training_data = []
        self.ml_trained = False

        # ---------------- Load NIMA Model if Available ----------------
        self.nima_model = None
        if TF_AVAILABLE and os.path.exists("nima_model.h5"):
            try:
                self.nima_model = load_model("nima_model.h5")
                self.log("Loaded NIMA model successfully.")
            except Exception as e:
                self.log(f"Error loading NIMA model: {e}")
                self.nima_model = None
        else:
            self.log("No NIMA model found; using dummy evaluation.")

        # ---------------- Setup GUI Layout ----------------
        self.setup_menubar()
        self.setup_main_layout()   # Main frame for header and content
        self.setup_left_panel()      # Left control panel (two columns, 400px wide)
        self.setup_right_canvas()    # Right canvas
        self.create_controls()       # Create control widgets in left panel

        self.update_settings_from_controls()
        self.animate()

        if API_AVAILABLE:
            self.start_api()

    def draw_grid(self):
        if self.SETTINGS["grid_overlay"]:
            grid_spacing = 50
            for x in range(0, self.CANVAS_WIDTH, grid_spacing):
                self.canvas.create_line(x, 0, x, self.CANVAS_HEIGHT, fill="#333333", dash=(2, 4), tags="grid")
            for y in range(0, self.CANVAS_HEIGHT, grid_spacing):
                self.canvas.create_line(0, y, self.CANVAS_WIDTH, y, fill="#333333", dash=(2, 4), tags="grid")
    
    def update_grid(self):
        self.canvas.delete("grid")
        self.draw_grid()
    
    def resize_canvas(self, event):
        self.CANVAS_WIDTH = event.width
        self.CANVAS_HEIGHT = event.height
        self.canvas.config(width=self.CANVAS_WIDTH, height=self.CANVAS_HEIGHT)
    
    def set_canvas_size(self):
        new_width = tk.simpledialog.askinteger("Canvas Width", "Enter new canvas width:", initialvalue=self.CANVAS_WIDTH)
        new_height = tk.simpledialog.askinteger("Canvas Height", "Enter new canvas height:", initialvalue=self.CANVAS_HEIGHT)
        if new_width and new_height:
            self.CANVAS_WIDTH = new_width
            self.CANVAS_HEIGHT = new_height
            self.canvas.config(width=new_width, height=new_height)
            self.status_label.config(text=f"Canvas size set to {new_width} x {new_height}")


    # ---------------- New Methods for Missing Functions ----------------
    def choose_bg_color(self):
        color = colorchooser.askcolor(title="Choose Background Color", initialcolor=self.SETTINGS["bg_color"])
        if color[1]:
            self.SETTINGS["bg_color"] = color[1]
            self.canvas.config(bg=self.SETTINGS["bg_color"])

    def set_random_seed(self):
        try:
            seed_val = int(self.seed_entry.get())
            random.seed(seed_val)
            self.SETTINGS["random_seed"] = seed_val
            self.status_label.config(text=f"Seed set to {seed_val}")
        except Exception:
            self.status_label.config(text="Invalid seed value")

    def update_grid(self):
        self.canvas.delete("grid")
        self.draw_grid()

    # ---------------- Layout Setup for Main Window ----------------
    def setup_main_layout(self):
        # Main frame to hold header and content
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # ---------------- Logging and Settings Export/Import Methods ----------------
    def log(self, message):
        timestamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
        log_message = f"{timestamp} {message}"
        print(log_message)
        try:
            with open(self.log_file, "a") as f:
                f.write(log_message + "\n")
        except Exception as e:
            print(f"Error writing log: {e}")

    def export_settings(self):
        try:
            with open("settings.json", "w") as f:
                json.dump(self.SETTINGS, f, indent=4)
            self.log("Settings exported to settings.json")
            messagebox.showinfo("Export Settings", "Settings exported successfully!")
        except Exception as e:
            messagebox.showerror("Export Settings", f"Error exporting settings: {e}")

    def import_settings(self):
        try:
            with open("settings.json", "r") as f:
                settings = json.load(f)
            self.SETTINGS.update(settings)
            self.log("Settings imported from settings.json")
            messagebox.showinfo("Import Settings", "Settings imported successfully!")
        except Exception as e:
            messagebox.showerror("Import Settings", f"Error importing settings: {e}")

    def show_history(self):
        history_win = tk.Toplevel(self.root)
        history_win.title("History of Generated Images")
        listbox = tk.Listbox(history_win, width=80)
        listbox.pack(fill=tk.BOTH, expand=True)
        for filename in self.history_list:
            listbox.insert(tk.END, filename)

    # ---------------- Menubar Setup ----------------
    def setup_menubar(self):
        menubar = tk.Menu(self.root, font=self.compact_font)
        file_menu = tk.Menu(menubar, tearoff=0, font=self.compact_font)
        file_menu.add_command(label="Save Art", command=self.save_art)
        file_menu.add_command(label="Save Screenshot", command=self.save_screenshot)
        file_menu.add_separator()
        file_menu.add_command(label="Export Settings", command=self.export_settings)
        file_menu.add_command(label="Import Settings", command=self.import_settings)
        file_menu.add_separator()
        file_menu.add_command(label="Set Canvas Size", command=self.set_canvas_size)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)

        effects_menu = tk.Menu(menubar, tearoff=0, font=self.compact_font)
        effects_menu.add_command(label="Enhance with ML", command=self.enhance_with_ml)
        effects_menu.add_command(label="Apply Realism Preset", command=lambda: self.status_label.config(text="Realism preset applied (dummy)!"))
        menubar.add_cascade(label="Effects", menu=effects_menu)

        settings_menu = tk.Menu(menubar, tearoff=0, font=self.compact_font)
        settings_menu.add_command(label="Show History", command=self.show_history)
        settings_menu.add_command(label="View Session Log", command=lambda: os.startfile(self.log_file) if os.name=="nt" else os.system(f"xdg-open {self.log_file}"))
        menubar.add_cascade(label="Settings", menu=settings_menu)

        help_menu = tk.Menu(menubar, tearoff=0, font=self.compact_font)
        help_menu.add_command(label="About", command=lambda: messagebox.showinfo("About", "Generative Art V1 Final - Extended Version\n\nUse the left panel to adjust settings.\n\nFile > Export/Import Settings, Effects > Enhance, etc."))
        menubar.add_cascade(label="Help", menu=help_menu)

        self.root.config(menu=menubar)

    # ---------------- Left Panel Setup ----------------
    def setup_left_panel(self):
        self.left_panel_width = 400
        self.control_container = tk.Frame(self.main_frame, padx=2, pady=2, width=self.left_panel_width)
        self.control_container.pack(side=tk.LEFT, fill=tk.Y)
        self.control_container.pack_propagate(False)
        self.control_canvas = tk.Canvas(self.control_container, borderwidth=0, background="#f0f0f0")
        self.control_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.vscrollbar = tk.Scrollbar(self.control_container, orient=tk.VERTICAL, command=self.control_canvas.yview)
        self.vscrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.control_canvas.configure(yscrollcommand=self.vscrollbar.set)
        self.control_frame = tk.Frame(self.control_canvas, padx=2, pady=2, background="#f0f0f0")
        self.control_canvas.create_window((0, 0), window=self.control_frame, anchor="nw")
        self.control_frame.bind("<Configure>", lambda event: self.control_canvas.configure(scrollregion=self.control_canvas.bbox("all")))

        self.left_col = tk.Frame(self.control_frame, background="#f0f0f0")
        self.left_col.grid(row=0, column=0, sticky="n")
        self.right_col = tk.Frame(self.control_frame, background="#f0f0f0")
        self.right_col.grid(row=0, column=1, sticky="n", padx=(10,0))

    # ---------------- Right Canvas Setup ----------------
    def setup_right_canvas(self):
        self.canvas_frame = tk.Frame(self.main_frame, bd=2, relief=tk.SUNKEN)
        self.canvas_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)
        self.canvas = tk.Canvas(self.canvas_frame, bg=self.SETTINGS["bg_color"])
        self.canvas.pack(expand=True, fill=tk.BOTH)
        # Bind the resizing of canvas_frame to update canvas size.
        self.canvas_frame.bind("<Configure>", self.resize_canvas)
        self.draw_grid()

    def resize_canvas(self, event):
        self.CANVAS_WIDTH = event.width
        self.CANVAS_HEIGHT = event.height
        self.canvas.config(width=self.CANVAS_WIDTH, height=self.CANVAS_HEIGHT)

    # ---------------- Create Control Widgets ----------------
    def create_controls(self):
        # ----- Left Column Controls -----
        tk.Label(self.left_col, text="Animation Controls", font=self.header_font, background="#f0f0f0").pack(anchor="w", pady=(0,1))
        tk.Label(self.left_col, text="Speed (ms):", font=self.compact_font, background="#f0f0f0").pack(anchor="w")
        self.speed_scale = tk.Scale(self.left_col, from_=1, to=5000, orient=tk.HORIZONTAL, font=self.compact_font, resolution=1, length=150)
        self.speed_scale.set(self.SETTINGS["animation_speed"])
        self.speed_scale.pack(fill=tk.X, pady=1)
        tk.Label(self.left_col, text="Auto-clear (frames):", font=self.compact_font, background="#f0f0f0").pack(anchor="w")
        self.auto_clear_scale = tk.Scale(self.left_col, from_=1, to=10000, orient=tk.HORIZONTAL, font=self.compact_font, resolution=1, length=150)
        self.auto_clear_scale.set(self.SETTINGS["auto_clear_threshold"])
        self.auto_clear_scale.pack(fill=tk.X, pady=1)
        tk.Label(self.left_col, text="Drawing Mode:", font=self.compact_font, background="#f0f0f0").pack(anchor="w")
        self.mode_frame_inner = tk.Frame(self.left_col, background="#f0f0f0")
        self.mode_frame_inner.pack(fill=tk.X, pady=1)
        self.drawing_mode = tk.StringVar(value="continuous")
        tk.Radiobutton(self.mode_frame_inner, text="Cont", variable=self.drawing_mode, value="continuous", font=self.compact_font, background="#f0f0f0").pack(side=tk.LEFT)
        tk.Radiobutton(self.mode_frame_inner, text="Burst", variable=self.drawing_mode, value="burst", font=self.compact_font, background="#f0f0f0").pack(side=tk.LEFT)
        tk.Label(self.left_col, text="Burst Count:", font=self.compact_font, background="#f0f0f0").pack(anchor="w")
        self.burst_count_scale = tk.Scale(self.left_col, from_=1, to=20, orient=tk.HORIZONTAL, font=self.compact_font, resolution=1, length=150)
        self.burst_count_scale.set(self.SETTINGS["burst_count"])
        self.burst_count_scale.pack(fill=tk.X, pady=1)
        
        tk.Label(self.left_col, text="Special Mode", font=self.header_font, background="#f0f0f0").pack(anchor="w", pady=(3,1))
        self.special_frame = tk.Frame(self.left_col, background="#f0f0f0")
        self.special_frame.pack(fill=tk.X, pady=1)
        tk.Radiobutton(self.special_frame, text="Normal", variable=self.special_mode, value="Normal", font=self.compact_font, background="#f0f0f0").pack(side=tk.LEFT, padx=2)
        tk.Radiobutton(self.special_frame, text="UltraRandom", variable=self.special_mode, value="UltraRandom", font=self.compact_font, background="#f0f0f0").pack(side=tk.LEFT, padx=2)
        tk.Checkbutton(self.left_col, text="Chaos Mode", variable=self.chaos_mode, font=self.compact_font, background="#f0f0f0").pack(anchor="w", pady=1)
        
        tk.Label(self.left_col, text="ML Mode", font=self.header_font, background="#f0f0f0").pack(anchor="w", pady=(3,1))
        tk.Checkbutton(self.left_col, text="Enable ML", variable=self.ml_mode, font=self.compact_font, background="#f0f0f0").pack(anchor="w", pady=1)
        tk.Checkbutton(self.left_col, text="ML Training", variable=self.ml_training, font=self.compact_font, background="#f0f0f0").pack(anchor="w", pady=1)
        tk.Button(self.left_col, text="Evaluate Image", command=self.evaluate_current_image, font=self.compact_font, height=1).pack(fill=tk.X, pady=1)
        tk.Button(self.left_col, text="Enhance with ML", command=self.enhance_with_ml, font=self.compact_font, height=1).pack(fill=tk.X, pady=1)
        
        # ----- Right Column Controls -----
        tk.Label(self.right_col, text="Shape & Style", font=self.header_font, background="#f0f0f0").pack(anchor="w", pady=(0,1))
        tk.Label(self.right_col, text="Select Shapes:", font=self.compact_font, background="#f0f0f0").pack(anchor="w")
        self.shape_frame_inner = tk.Frame(self.right_col, background="#f0f0f0")
        self.shape_frame_inner.pack(fill=tk.X, pady=1)
        self.shape_vars = {}
        for shape in self.SHAPE_TYPES.keys():
            var = tk.BooleanVar(value=True)
            self.shape_vars[shape] = var
            def update_shape(s=shape, var=var):
                self.SHAPE_TYPES[s] = var.get()
            tk.Checkbutton(self.shape_frame_inner, text=shape[:3].capitalize(), variable=var, command=update_shape, font=self.compact_font, background="#f0f0f0").pack(side=tk.LEFT, padx=1)
        tk.Label(self.right_col, text="Min Size:", font=self.compact_font, background="#f0f0f0").pack(anchor="w")
        self.min_size_scale = tk.Scale(self.right_col, from_=5, to=50, orient=tk.HORIZONTAL, font=self.compact_font, resolution=1, length=150)
        self.min_size_scale.set(self.SETTINGS["min_shape_size"])
        self.min_size_scale.pack(fill=tk.X, pady=1)
        tk.Label(self.right_col, text="Max Size:", font=self.compact_font, background="#f0f0f0").pack(anchor="w")
        self.max_size_scale = tk.Scale(self.right_col, from_=50, to=200, orient=tk.HORIZONTAL, font=self.compact_font, resolution=1, length=150)
        self.max_size_scale.set(self.SETTINGS["max_shape_size"])
        self.max_size_scale.pack(fill=tk.X, pady=1)
        self.outline_var = tk.BooleanVar(value=self.SETTINGS["outline_only"])
        tk.Checkbutton(self.right_col, text="Outline", variable=self.outline_var, font=self.compact_font, background="#f0f0f0").pack(anchor="w", pady=1)
        self.mirror_var = tk.BooleanVar(value=self.SETTINGS["mirror_drawing"])
        tk.Checkbutton(self.right_col, text="Mirror", variable=self.mirror_var, font=self.compact_font, background="#f0f0f0").pack(anchor="w", pady=1)
        self.grid_var = tk.BooleanVar(value=self.SETTINGS["grid_overlay"])
        tk.Checkbutton(self.right_col, text="Grid", variable=self.grid_var, command=self.toggle_grid, font=self.compact_font, background="#f0f0f0").pack(anchor="w", pady=1)
        
        tk.Label(self.right_col, text="Color & Palette", font=self.header_font, background="#f0f0f0").pack(anchor="w", pady=(3,1))
        tk.Button(self.right_col, text="BG Color", command=self.choose_bg_color, font=self.compact_font, height=1).pack(fill=tk.X, pady=1)
        tk.Label(self.right_col, text="Palette (hexs):", font=self.compact_font, background="#f0f0f0").pack(anchor="w")
        self.palette_entry = tk.Entry(self.right_col, font=self.compact_font)
        self.palette_entry.insert(0, self.SETTINGS["custom_palette"])
        self.palette_entry.pack(fill=tk.X, pady=1)
        
        tk.Label(self.right_col, text="Meme & Watermark", font=self.header_font, background="#f0f0f0").pack(anchor="w", pady=(3,1))
        tk.Label(self.right_col, text="Meme Text:", font=self.compact_font, background="#f0f0f0").pack(anchor="w")
        self.meme_entry = tk.Entry(self.right_col, font=self.compact_font)
        self.meme_entry.pack(fill=tk.X, pady=1)
        tk.Button(self.right_col, text="Update Meme", command=self.add_meme_text, font=self.compact_font, height=1).pack(fill=tk.X, pady=1)
        tk.Label(self.right_col, text="Font Size:", font=self.compact_font, background="#f0f0f0").pack(anchor="w")
        self.meme_font_size_scale = tk.Scale(self.right_col, from_=10, to=72, orient=tk.HORIZONTAL, font=self.compact_font, resolution=1, length=150)
        self.meme_font_size_scale.set(self.SETTINGS["meme_font_size"])
        self.meme_font_size_scale.pack(fill=tk.X, pady=1)
        self.watermark_var = tk.BooleanVar(value=self.SETTINGS["watermark_enabled"])
        tk.Checkbutton(self.right_col, text="Wmark", variable=self.watermark_var, font=self.compact_font, background="#f0f0f0").pack(anchor="w", pady=1)
        tk.Label(self.right_col, text="Wmark Txt:", font=self.compact_font, background="#f0f0f0").pack(anchor="w")
        self.watermark_text_entry = tk.Entry(self.right_col, font=self.compact_font)
        self.watermark_text_entry.insert(0, self.SETTINGS["watermark_text"])
        self.watermark_text_entry.pack(fill=tk.X, pady=1)
        
        tk.Label(self.right_col, text="Seed & Save", font=self.header_font, background="#f0f0f0").pack(anchor="w", pady=(3,1))
        tk.Label(self.right_col, text="Seed (int):", font=self.compact_font, background="#f0f0f0").pack(anchor="w")
        self.seed_entry = tk.Entry(self.right_col, font=self.compact_font)
        self.seed_entry.pack(fill=tk.X, pady=1)
        tk.Button(self.right_col, text="Set Seed", command=self.set_random_seed, font=self.compact_font, height=1).pack(fill=tk.X, pady=1)
        tk.Label(self.right_col, text="Save Name (PS):", font=self.compact_font, background="#f0f0f0").pack(anchor="w")
        self.save_entry = tk.Entry(self.right_col, font=self.compact_font)
        self.save_entry.pack(fill=tk.X, pady=1)
        tk.Button(self.right_col, text="Save Art", command=self.save_art, font=self.compact_font, height=1).pack(fill=tk.X, pady=1)
        tk.Button(self.right_col, text="Save Shot", command=self.save_screenshot, font=self.compact_font, height=1).pack(fill=tk.X, pady=1)
    
    # ---------------- Animation and Drawing Methods ----------------
    def animate(self):
        if not self.paused:
            self.frame_count += 1
            if self.drawing_mode.get() == "continuous":
                self.draw_shape()
            else:
                for _ in range(self.SETTINGS.get("burst_count", 5)):
                    self.draw_shape()
            if self.frame_count >= self.SETTINGS["auto_clear_threshold"]:
                self.clear_canvas()
            if self.chaos_mode.get() and (self.frame_count % 50 == 0):
                self.canvas.config(bg=random_color())
            if self.SETTINGS["meme_text"]:
                if self.meme_text_id is None:
                    self.meme_text_id = self.canvas.create_text(self.CANVAS_WIDTH//2, self.CANVAS_HEIGHT-50,
                                                                 text=self.SETTINGS["meme_text"],
                                                                 fill="white",
                                                                 font=(self.SETTINGS["meme_font"], self.SETTINGS["meme_font_size"], "bold"))
                else:
                    self.canvas.itemconfigure(self.meme_text_id, text=self.SETTINGS["meme_text"])
                self.canvas.tag_raise(self.meme_text_id)
            if self.SETTINGS["watermark_enabled"]:
                if self.watermark_id is None:
                    self.watermark_id = self.canvas.create_text(self.CANVAS_WIDTH-80, self.CANVAS_HEIGHT-20,
                                                                 text=self.SETTINGS["watermark_text"],
                                                                 fill="#888888",
                                                                 font=("Arial", 8, "italic"))
                else:
                    self.canvas.itemconfigure(self.watermark_id, text=self.SETTINGS["watermark_text"])
                self.canvas.tag_raise(self.watermark_id)
            else:
                if self.watermark_id:
                    self.canvas.delete(self.watermark_id)
                    self.watermark_id = None
        self.root.after(self.speed_scale.get(), self.animate)
    
    def draw_shape(self):
        if self.special_mode.get() == "UltraRandom":
            self.SETTINGS["outline_only"] = random.choice([True, False])
            self.SETTINGS["mirror_drawing"] = random.choice([True, False])
            if random.choice([True, False]):
                self.SETTINGS["custom_palette"] = ",".join(["#%06x" % random.randint(0, 0xFFFFFF) for _ in range(3)])
            else:
                self.SETTINGS["custom_palette"] = ""
        available_shapes = [s for s, enabled in self.SHAPE_TYPES.items() if enabled]
        shape_type = random.choice(available_shapes) if available_shapes else "circle"
        min_size = self.SETTINGS["min_shape_size"]
        max_size = self.SETTINGS["max_shape_size"]
        size = random.randint(min_size, max_size)
        x = random.randint(0, self.CANVAS_WIDTH)
        y = random.randint(0, self.CANVAS_HEIGHT)
        color = random_color()
        outline_flag = self.SETTINGS["outline_only"]
        shape_record = {"type": shape_type, "items": []}
        if shape_type == "circle":
            r = size
            if outline_flag:
                item = self.canvas.create_oval(x-r, y-r, x+r, y+r, outline=color, width=2, fill="")
            else:
                item = self.canvas.create_oval(x-r, y-r, x+r, y+r, fill=color, outline=color)
            shape_record["items"].append(item)
        elif shape_type == "line":
            x2 = random.randint(0, self.CANVAS_WIDTH)
            y2 = random.randint(0, self.CANVAS_HEIGHT)
            width_line = random.randint(1, 5)
            item = self.canvas.create_line(x, y, x2, y2, fill=color, width=width_line)
            shape_record["items"].append(item)
        elif shape_type == "arc":
            x1, y1 = x, y
            x2, y2 = x+size, y+size
            start_angle = random.randint(0, 360)
            extent_angle = random.randint(30, 180)
            if outline_flag:
                item = self.canvas.create_arc(x1, y1, x2, y2, start=start_angle, extent=extent_angle,
                                              outline=color, style=tk.ARC, width=2)
            else:
                item = self.canvas.create_arc(x1, y1, x2, y2, start=start_angle, extent=extent_angle,
                                              fill=color, outline=color)
            shape_record["items"].append(item)
        elif shape_type == "rectangle":
            x1, y1 = x, y
            x2, y2 = x+size, y+size
            if outline_flag:
                item = self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=2, fill="")
            else:
                item = self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline=color)
            shape_record["items"].append(item)
        elif shape_type == "triangle":
            x2 = x + random.randint(-size, size)
            y2 = y + random.randint(-size, size)
            x3 = x + random.randint(-size, size)
            y3 = y + random.randint(-size, size)
            if outline_flag:
                item = self.canvas.create_polygon(x, y, x2, y2, x3, y3, outline=color, fill="", width=2)
            else:
                item = self.canvas.create_polygon(x, y, x2, y2, x3, y3, fill=color, outline=color)
            shape_record["items"].append(item)
        if self.SETTINGS["mirror_drawing"]:
            mirror_items = []
            for item in shape_record["items"]:
                coords = self.canvas.coords(item)
                mirrored_coords = [self.CANVAS_WIDTH - val if idx % 2 == 0 else val for idx, val in enumerate(coords)]
                item_type = self.canvas.type(item)
                if item_type == "oval":
                    if outline_flag:
                        m_item = self.canvas.create_oval(*mirrored_coords, outline=color, width=2, fill="")
                    else:
                        m_item = self.canvas.create_oval(*mirrored_coords, fill=color, outline=color)
                elif item_type == "line":
                    m_item = self.canvas.create_line(*mirrored_coords, fill=color, width=width_line)
                elif item_type == "arc":
                    if outline_flag:
                        m_item = self.canvas.create_arc(*mirrored_coords, outline=color, style=tk.ARC, width=2)
                    else:
                        m_item = self.canvas.create_arc(*mirrored_coords, fill=color, outline=color)
                elif item_type == "rectangle":
                    if outline_flag:
                        m_item = self.canvas.create_rectangle(*mirrored_coords, outline=color, width=2, fill="")
                    else:
                        m_item = self.canvas.create_rectangle(*mirrored_coords, fill=color, outline=color)
                elif item_type == "polygon":
                    if outline_flag:
                        m_item = self.canvas.create_polygon(mirrored_coords, outline=color, fill="", width=2)
                    else:
                        m_item = self.canvas.create_polygon(mirrored_coords, fill=color, outline=color)
                else:
                    m_item = None
                if m_item:
                    mirror_items.append(m_item)
            shape_record["mirror_items"] = mirror_items
        self.draw_history.append(shape_record)
    
    # ---------------- Control Callbacks and Utility Methods ----------------
    def toggle_pause(self):
        self.paused = not self.paused
        # For simplicity, you can update the status_label instead of a separate pause button.
        self.status_label.config(text="Paused" if self.paused else "Running")
    
    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw_history = []
        self.frame_count = 0
        self.update_grid()
        if self.SETTINGS["meme_text"]:
            self.meme_text_id = self.canvas.create_text(self.CANVAS_WIDTH//2, self.CANVAS_HEIGHT-50,
                                                         text=self.SETTINGS["meme_text"],
                                                         fill="white",
                                                         font=(self.SETTINGS["meme_font"], self.SETTINGS["meme_font_size"], "bold"))
        if self.SETTINGS["watermark_enabled"]:
            self.watermark_id = self.canvas.create_text(self.CANVAS_WIDTH-80, self.CANVAS_HEIGHT-20,
                                                        text=self.SETTINGS["watermark_text"],
                                                        fill="#888888",
                                                        font=("Arial", 8, "italic"))
    
    def undo_last_shape(self):
        if self.draw_history:
            last_shape = self.draw_history.pop()
            for item in last_shape.get("items", []):
                self.canvas.delete(item)
            for m_item in last_shape.get("mirror_items", []):
                self.canvas.delete(m_item)
            self.status_label.config(text="Last shape undone")
        else:
            self.status_label.config(text="No shapes to undo")
    
    def replay_history(self):
        if not self.draw_history:
            self.status_label.config(text="No history to replay")
            return
        history_copy = self.draw_history[:]
        self.clear_canvas()
        self.frame_count = 0
        self.status_label.config(text="Replaying history...")
        def replay_next(index):
            if index < len(history_copy):
                self.draw_shape()
                self.root.after(50, lambda: replay_next(index + 1))
            else:
                self.status_label.config(text="Replay complete")
        replay_next(0)
    
    def add_meme_text(self):
        text = self.meme_entry.get()
        self.SETTINGS["meme_text"] = text
        if self.meme_text_id is None:
            self.meme_text_id = self.canvas.create_text(self.CANVAS_WIDTH//2, self.CANVAS_HEIGHT-50,
                                                         text=text,
                                                         fill="white",
                                                         font=(self.SETTINGS["meme_font"], self.SETTINGS["meme_font_size"], "bold"))
        else:
            self.canvas.itemconfigure(self.meme_text_id, text=text)
        self.canvas.tag_raise(self.meme_text_id)
        self.status_label.config(text="Meme text updated")
    
    def save_art(self):
        self.save_meme()
    
    def save_meme(self):
        fname = self.save_entry.get().strip()
        if not fname:
            fname = f"meme_{int(time.time())}.ps"
        else:
            if not fname.endswith(".ps"):
                fname += ".ps"
        try:
            self.canvas.postscript(file=fname)
            if PIL_AVAILABLE:
                img = Image.open(fname)
                png_name = fname.replace(".ps", ".png")
                img.save(png_name, "png")
                self.status_label.config(text=f"Meme saved as {png_name}")
                self.history_list.append(png_name)
            else:
                self.status_label.config(text=f"Meme saved as {fname} (PS format)")
                self.history_list.append(fname)
        except Exception as e:
            self.status_label.config(text=f"Error saving meme: {e}")
    
    def save_screenshot(self):
        self.root.update_idletasks()
        if not PIL_AVAILABLE:
            self.status_label.config(text="Pillow (ImageGrab) is not available.")
            return None
        try:
            x = self.canvas.winfo_rootx()
            y = self.canvas.winfo_rooty()
            w = self.canvas.winfo_width()
            h = self.canvas.winfo_height()
            bbox = (x, y, x+w, y+h)
            img = ImageGrab.grab(bbox)
        except Exception as e:
            self.status_label.config(text=f"Error capturing screenshot: {e}")
            return None
        ext_folder = os.path.join(os.getcwd(), "ext")
        if not os.path.exists(ext_folder):
            os.makedirs(ext_folder)
        fname = f"meme_{int(time.time())}.png"
        filepath = os.path.join(ext_folder, fname)
        try:
            img.save(filepath, "PNG")
            self.status_label.config(text=f"Screenshot saved as {filepath}")
            self.history_list.append(filepath)
            return filepath
        except Exception as e:
            self.status_label.config(text=f"Error saving screenshot: {e}")
            return None
    
    def evaluate_current_image(self):
        filepath = self.save_screenshot()
        if not filepath:
            self.status_label.config(text="Could not capture image for evaluation.")
            return
        global ml_trained
        if self.ml_mode.get():
            if self.ml_training.get():
                result = messagebox.askyesno("Aesthetic Feedback", "Do you like this image?")
                self.training_data.append(result)
                self.status_label.config(text=f"Feedback recorded: {'Nice' if result else 'Not nice'}. Total: {len(self.training_data)}")
                if len(self.training_data) >= 10:
                    ml_trained = True
                    self.status_label.config(text="ML model trained with feedback!")
            elif ml_trained and TF_AVAILABLE and self.nima_model is not None:
                try:
                    img = Image.open(filepath)
                    img_resized = img.resize((224,224))
                    img_array = np.array(img_resized)/255.0
                    img_array = np.expand_dims(img_array, axis=0)
                    prediction = self.nima_model.predict(img_array)
                    scores = np.arange(1, 11)
                    score = np.sum(prediction * scores) / np.sum(prediction)
                    self.status_label.config(text=f"NIMA evaluation: Score {score:.2f}")
                except Exception as e:
                    self.status_label.config(text=f"ML evaluation error: {e}")
            else:
                result = messagebox.askyesno("Aesthetic Feedback", "Do you like this image?")
                self.status_label.config(text=f"Feedback: {'Nice' if result else 'Not nice'}")
        else:
            self.status_label.config(text="ML mode not enabled.")
    
    def enhance_with_ml(self):
        if not PIL_AVAILABLE:
            self.status_label.config(text="Pillow (ImageGrab) is not available.")
            return
        filepath = self.save_screenshot()
        if not filepath:
            self.status_label.config(text="Could not capture image for enhancement.")
            return
        try:
            img = Image.open(filepath)
            enhancer_color = ImageEnhance.Color(img)
            img = enhancer_color.enhance(1.5)
            enhancer_contrast = ImageEnhance.Contrast(img)
            img = enhancer_contrast.enhance(1.3)
            enhancer_sharpness = ImageEnhance.Sharpness(img)
            img = enhancer_sharpness.enhance(1.2)
            enhanced_path = filepath.replace(".png", "_enhanced.png")
            img.save(enhanced_path, "PNG")
            self.status_label.config(text=f"Enhanced image saved as {enhanced_path}")
        except Exception as e:
            self.status_label.config(text=f"Error during ML enhancement: {e}")
    
    def toggle_grid(self):
        self.SETTINGS["grid_overlay"] = self.grid_var.get()
        self.draw_grid()
    
    def update_settings_from_controls(self):
        self.SETTINGS["animation_speed"] = self.speed_scale.get()
        self.SETTINGS["auto_clear_threshold"] = self.auto_clear_scale.get()
        self.SETTINGS["min_shape_size"] = self.min_size_scale.get()
        self.SETTINGS["max_shape_size"] = self.max_size_scale.get()
        self.SETTINGS["outline_only"] = self.outline_var.get()
        self.SETTINGS["mirror_drawing"] = self.mirror_var.get()
        self.SETTINGS["grid_overlay"] = self.grid_var.get()
        self.SETTINGS["custom_palette"] = self.palette_entry.get().strip()
        self.SETTINGS["meme_font_size"] = self.meme_font_size_scale.get()
        self.SETTINGS["watermark_enabled"] = self.watermark_var.get()
        self.SETTINGS["burst_count"] = self.burst_count_scale.get()
        self.root.after(100, self.update_settings_from_controls)
    
    def start_api(self):
        api_app = Flask(__name__)
        @api_app.route("/api/take_screenshot", methods=["GET"])
        def api_take_screenshot():
            filepath = self.save_screenshot()
            if filepath:
                return jsonify({"status": "success", "message": "Screenshot taken", "filename": filepath})
            else:
                return jsonify({"status": "error", "message": "Screenshot failed"}), 500
        def run_api():
            api_app.run(port=5000, host="0.0.0.0", debug=False)
        import threading
        api_thread = threading.Thread(target=run_api, daemon=True)
        api_thread.start()
        self.log("API endpoint available at http://localhost:5000/api/take_screenshot")
    
    def save_art(self):
        self.save_meme()

    def set_canvas_size(self):
        # Prompt the user to set a custom canvas size.
        new_width = simpledialog.askinteger("Canvas Width", "Enter new canvas width:", initialvalue=self.CANVAS_WIDTH)
        new_height = simpledialog.askinteger("Canvas Height", "Enter new canvas height:", initialvalue=self.CANVAS_HEIGHT)
        if new_width and new_height:
            self.CANVAS_WIDTH = new_width
            self.CANVAS_HEIGHT = new_height
            self.canvas.config(width=new_width, height=new_height)
            self.status_label.config(text=f"Canvas size set to {new_width} x {new_height}")

    # ---------------- Animation and Drawing Methods ----------------
    def animate(self):
        if not self.paused:
            self.frame_count += 1
            if self.drawing_mode.get() == "continuous":
                self.draw_shape()
            else:
                for _ in range(self.SETTINGS.get("burst_count", 5)):
                    self.draw_shape()
            if self.frame_count >= self.SETTINGS["auto_clear_threshold"]:
                self.clear_canvas()
            if self.chaos_mode.get() and (self.frame_count % 50 == 0):
                self.canvas.config(bg=random_color())
            if self.SETTINGS["meme_text"]:
                if self.meme_text_id is None:
                    self.meme_text_id = self.canvas.create_text(self.CANVAS_WIDTH//2, self.CANVAS_HEIGHT-50,
                                                                 text=self.SETTINGS["meme_text"],
                                                                 fill="white",
                                                                 font=(self.SETTINGS["meme_font"], self.SETTINGS["meme_font_size"], "bold"))
                else:
                    self.canvas.itemconfigure(self.meme_text_id, text=self.SETTINGS["meme_text"])
                self.canvas.tag_raise(self.meme_text_id)
            if self.SETTINGS["watermark_enabled"]:
                if self.watermark_id is None:
                    self.watermark_id = self.canvas.create_text(self.CANVAS_WIDTH-80, self.CANVAS_HEIGHT-20,
                                                                 text=self.SETTINGS["watermark_text"],
                                                                 fill="#888888",
                                                                 font=("Arial", 8, "italic"))
                else:
                    self.canvas.itemconfigure(self.watermark_id, text=self.SETTINGS["watermark_text"])
                self.canvas.tag_raise(self.watermark_id)
            else:
                if self.watermark_id:
                    self.canvas.delete(self.watermark_id)
                    self.watermark_id = None
        self.root.after(self.speed_scale.get(), self.animate)
    
    def draw_shape(self):
        if self.special_mode.get() == "UltraRandom":
            self.SETTINGS["outline_only"] = random.choice([True, False])
            self.SETTINGS["mirror_drawing"] = random.choice([True, False])
            if random.choice([True, False]):
                self.SETTINGS["custom_palette"] = ",".join(["#%06x" % random.randint(0, 0xFFFFFF) for _ in range(3)])
            else:
                self.SETTINGS["custom_palette"] = ""
        available_shapes = [s for s, enabled in self.SHAPE_TYPES.items() if enabled]
        shape_type = random.choice(available_shapes) if available_shapes else "circle"
        min_size = self.SETTINGS["min_shape_size"]
        max_size = self.SETTINGS["max_shape_size"]
        size = random.randint(min_size, max_size)
        x = random.randint(0, self.CANVAS_WIDTH)
        y = random.randint(0, self.CANVAS_HEIGHT)
        color = random_color()
        outline_flag = self.SETTINGS["outline_only"]
        shape_record = {"type": shape_type, "items": []}
        if shape_type == "circle":
            r = size
            if outline_flag:
                item = self.canvas.create_oval(x-r, y-r, x+r, y+r, outline=color, width=2, fill="")
            else:
                item = self.canvas.create_oval(x-r, y-r, x+r, y+r, fill=color, outline=color)
            shape_record["items"].append(item)
        elif shape_type == "line":
            x2 = random.randint(0, self.CANVAS_WIDTH)
            y2 = random.randint(0, self.CANVAS_HEIGHT)
            width_line = random.randint(1, 5)
            item = self.canvas.create_line(x, y, x2, y2, fill=color, width=width_line)
            shape_record["items"].append(item)
        elif shape_type == "arc":
            x1, y1 = x, y
            x2, y2 = x+size, y+size
            start_angle = random.randint(0, 360)
            extent_angle = random.randint(30, 180)
            if outline_flag:
                item = self.canvas.create_arc(x1, y1, x2, y2, start=start_angle, extent=extent_angle,
                                              outline=color, style=tk.ARC, width=2)
            else:
                item = self.canvas.create_arc(x1, y1, x2, y2, start=start_angle, extent=extent_angle,
                                              fill=color, outline=color)
            shape_record["items"].append(item)
        elif shape_type == "rectangle":
            x1, y1 = x, y
            x2, y2 = x+size, y+size
            if outline_flag:
                item = self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=2, fill="")
            else:
                item = self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline=color)
            shape_record["items"].append(item)
        elif shape_type == "triangle":
            x2 = x + random.randint(-size, size)
            y2 = y + random.randint(-size, size)
            x3 = x + random.randint(-size, size)
            y3 = y + random.randint(-size, size)
            if outline_flag:
                item = self.canvas.create_polygon(x, y, x2, y2, x3, y3, outline=color, fill="", width=2)
            else:
                item = self.canvas.create_polygon(x, y, x2, y2, x3, y3, fill=color, outline=color)
            shape_record["items"].append(item)
        if self.SETTINGS["mirror_drawing"]:
            mirror_items = []
            for item in shape_record["items"]:
                coords = self.canvas.coords(item)
                mirrored_coords = [self.CANVAS_WIDTH - val if idx % 2 == 0 else val for idx, val in enumerate(coords)]
                item_type = self.canvas.type(item)
                if item_type == "oval":
                    if outline_flag:
                        m_item = self.canvas.create_oval(*mirrored_coords, outline=color, width=2, fill="")
                    else:
                        m_item = self.canvas.create_oval(*mirrored_coords, fill=color, outline=color)
                elif item_type == "line":
                    m_item = self.canvas.create_line(*mirrored_coords, fill=color, width=width_line)
                elif item_type == "arc":
                    if outline_flag:
                        m_item = self.canvas.create_arc(*mirrored_coords, outline=color, style=tk.ARC, width=2)
                    else:
                        m_item = self.canvas.create_arc(*mirrored_coords, fill=color, outline=color)
                elif item_type == "rectangle":
                    if outline_flag:
                        m_item = self.canvas.create_rectangle(*mirrored_coords, outline=color, width=2, fill="")
                    else:
                        m_item = self.canvas.create_rectangle(*mirrored_coords, fill=color, outline=color)
                elif item_type == "polygon":
                    if outline_flag:
                        m_item = self.canvas.create_polygon(mirrored_coords, outline=color, fill="", width=2)
                    else:
                        m_item = self.canvas.create_polygon(mirrored_coords, fill=color, outline=color)
                else:
                    m_item = None
                if m_item:
                    mirror_items.append(m_item)
            shape_record["mirror_items"] = mirror_items
        self.draw_history.append(shape_record)
    
    # ---------------- Control Callbacks and Utility Methods ----------------
    def toggle_pause(self):
        self.paused = not self.paused
        # (You could also update a pause button if you have one.)
        self.status_label.config(text="Paused" if self.paused else "Running")
    
    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw_history = []
        self.frame_count = 0
        self.update_grid()
        if self.SETTINGS["meme_text"]:
            self.meme_text_id = self.canvas.create_text(self.CANVAS_WIDTH//2, self.CANVAS_HEIGHT-50,
                                                         text=self.SETTINGS["meme_text"],
                                                         fill="white",
                                                         font=(self.SETTINGS["meme_font"], self.SETTINGS["meme_font_size"], "bold"))
        if self.SETTINGS["watermark_enabled"]:
            self.watermark_id = self.canvas.create_text(self.CANVAS_WIDTH-80, self.CANVAS_HEIGHT-20,
                                                        text=self.SETTINGS["watermark_text"],
                                                        fill="#888888",
                                                        font=("Arial", 8, "italic"))
    
    def undo_last_shape(self):
        if self.draw_history:
            last_shape = self.draw_history.pop()
            for item in last_shape.get("items", []):
                self.canvas.delete(item)
            for m_item in last_shape.get("mirror_items", []):
                self.canvas.delete(m_item)
            self.status_label.config(text="Last shape undone")
        else:
            self.status_label.config(text="No shapes to undo")
    
    def replay_history(self):
        if not self.draw_history:
            self.status_label.config(text="No history to replay")
            return
        history_copy = self.draw_history[:]
        self.clear_canvas()
        self.frame_count = 0
        self.status_label.config(text="Replaying history...")
        def replay_next(index):
            if index < len(history_copy):
                self.draw_shape()
                self.root.after(50, lambda: replay_next(index + 1))
            else:
                self.status_label.config(text="Replay complete")
        replay_next(0)
    
    def add_meme_text(self):
        text = self.meme_entry.get()
        self.SETTINGS["meme_text"] = text
        if self.meme_text_id is None:
            self.meme_text_id = self.canvas.create_text(self.CANVAS_WIDTH//2, self.CANVAS_HEIGHT-50,
                                                         text=text,
                                                         fill="white",
                                                         font=(self.SETTINGS["meme_font"], self.SETTINGS["meme_font_size"], "bold"))
        else:
            self.canvas.itemconfigure(self.meme_text_id, text=text)
        self.canvas.tag_raise(self.meme_text_id)
        self.status_label.config(text="Meme text updated")
    
    def save_art(self):
        self.save_meme()
    
    def save_meme(self):
        fname = self.save_entry.get().strip()
        if not fname:
            fname = f"meme_{int(time.time())}.ps"
        else:
            if not fname.endswith(".ps"):
                fname += ".ps"
        try:
            self.canvas.postscript(file=fname)
            if PIL_AVAILABLE:
                img = Image.open(fname)
                png_name = fname.replace(".ps", ".png")
                img.save(png_name, "png")
                self.status_label.config(text=f"Meme saved as {png_name}")
                self.history_list.append(png_name)
            else:
                self.status_label.config(text=f"Meme saved as {fname} (PS format)")
                self.history_list.append(fname)
        except Exception as e:
            self.status_label.config(text=f"Error saving meme: {e}")
    
    def save_screenshot(self):
        self.root.update_idletasks()
        if not PIL_AVAILABLE:
            self.status_label.config(text="Pillow (ImageGrab) is not available.")
            return None
        try:
            x = self.canvas.winfo_rootx()
            y = self.canvas.winfo_rooty()
            w = self.canvas.winfo_width()
            h = self.canvas.winfo_height()
            bbox = (x, y, x+w, y+h)
            img = ImageGrab.grab(bbox)
        except Exception as e:
            self.status_label.config(text=f"Error capturing screenshot: {e}")
            return None
        ext_folder = os.path.join(os.getcwd(), "ext")
        if not os.path.exists(ext_folder):
            os.makedirs(ext_folder)
        fname = f"meme_{int(time.time())}.png"
        filepath = os.path.join(ext_folder, fname)
        try:
            img.save(filepath, "PNG")
            self.status_label.config(text=f"Screenshot saved as {filepath}")
            self.history_list.append(filepath)
            return filepath
        except Exception as e:
            self.status_label.config(text=f"Error saving screenshot: {e}")
            return None
    
    def evaluate_current_image(self):
        filepath = self.save_screenshot()
        if not filepath:
            self.status_label.config(text="Could not capture image for evaluation.")
            return
        global ml_trained
        if self.ml_mode.get():
            if self.ml_training.get():
                result = messagebox.askyesno("Aesthetic Feedback", "Do you like this image?")
                self.training_data.append(result)
                self.status_label.config(text=f"Feedback recorded: {'Nice' if result else 'Not nice'}. Total: {len(self.training_data)}")
                if len(self.training_data) >= 10:
                    ml_trained = True
                    self.status_label.config(text="ML model trained with feedback!")
            elif ml_trained and TF_AVAILABLE and self.nima_model is not None:
                try:
                    img = Image.open(filepath)
                    img_resized = img.resize((224,224))
                    img_array = np.array(img_resized)/255.0
                    img_array = np.expand_dims(img_array, axis=0)
                    prediction = self.nima_model.predict(img_array)
                    scores = np.arange(1, 11)
                    score = np.sum(prediction * scores) / np.sum(prediction)
                    self.status_label.config(text=f"NIMA evaluation: Score {score:.2f}")
                except Exception as e:
                    self.status_label.config(text=f"ML evaluation error: {e}")
            else:
                result = messagebox.askyesno("Aesthetic Feedback", "Do you like this image?")
                self.status_label.config(text=f"Feedback: {'Nice' if result else 'Not nice'}")
        else:
            self.status_label.config(text="ML mode not enabled.")
    
    def enhance_with_ml(self):
        if not PIL_AVAILABLE:
            self.status_label.config(text="Pillow (ImageGrab) is not available.")
            return
        filepath = self.save_screenshot()
        if not filepath:
            self.status_label.config(text="Could not capture image for enhancement.")
            return
        try:
            img = Image.open(filepath)
            enhancer_color = ImageEnhance.Color(img)
            img = enhancer_color.enhance(1.5)
            enhancer_contrast = ImageEnhance.Contrast(img)
            img = enhancer_contrast.enhance(1.3)
            enhancer_sharpness = ImageEnhance.Sharpness(img)
            img = enhancer_sharpness.enhance(1.2)
            enhanced_path = filepath.replace(".png", "_enhanced.png")
            img.save(enhanced_path, "PNG")
            self.status_label.config(text=f"Enhanced image saved as {enhanced_path}")
        except Exception as e:
            self.status_label.config(text=f"Error during ML enhancement: {e}")
    
    def toggle_grid(self):
        self.SETTINGS["grid_overlay"] = self.grid_var.get()
        self.draw_grid()
    
    def update_settings_from_controls(self):
        self.SETTINGS["animation_speed"] = self.speed_scale.get()
        self.SETTINGS["auto_clear_threshold"] = self.auto_clear_scale.get()
        self.SETTINGS["min_shape_size"] = self.min_size_scale.get()
        self.SETTINGS["max_shape_size"] = self.max_size_scale.get()
        self.SETTINGS["outline_only"] = self.outline_var.get()
        self.SETTINGS["mirror_drawing"] = self.mirror_var.get()
        self.SETTINGS["grid_overlay"] = self.grid_var.get()
        self.SETTINGS["custom_palette"] = self.palette_entry.get().strip()
        self.SETTINGS["meme_font_size"] = self.meme_font_size_scale.get()
        self.SETTINGS["watermark_enabled"] = self.watermark_var.get()
        self.SETTINGS["burst_count"] = self.burst_count_scale.get()
        self.root.after(100, self.update_settings_from_controls)
    
    def start_api(self):
        api_app = Flask(__name__)
        @api_app.route("/api/take_screenshot", methods=["GET"])
        def api_take_screenshot():
            filepath = self.save_screenshot()
            if filepath:
                return jsonify({"status": "success", "message": "Screenshot taken", "filename": filepath})
            else:
                return jsonify({"status": "error", "message": "Screenshot failed"}), 500
        def run_api():
            api_app.run(port=5000, host="0.0.0.0", debug=False)
        import threading
        api_thread = threading.Thread(target=run_api, daemon=True)
        api_thread.start()
        self.log("API endpoint available at http://localhost:5000/api/take_screenshot")
    
    def save_art(self):
        self.save_meme()
    
    def set_canvas_size(self):
        new_width = simpledialog.askinteger("Canvas Width", "Enter new canvas width:", initialvalue=self.CANVAS_WIDTH)
        new_height = simpledialog.askinteger("Canvas Height", "Enter new canvas height:", initialvalue=self.CANVAS_HEIGHT)
        if new_width and new_height:
            self.CANVAS_WIDTH = new_width
            self.CANVAS_HEIGHT = new_height
            self.canvas.config(width=new_width, height=new_height)
            self.status_label.config(text=f"Canvas size set to {new_width} x {new_height}")

# ============================================================
# Main Execution
# ============================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = GenerativeArtApp(root)
    # Create a header label at the top of the window
    header_label = tk.Label(root, text="Welcome to Generative Art Final", font=("Arial", 16, "bold"), bg="#cccccc")
    header_label.pack(side=tk.TOP, fill=tk.X)
    # Create a status label at the bottom
    app.status_label = tk.Label(root, text="Status: Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W, font=app.compact_font)
    app.status_label.pack(side=tk.BOTTOM, fill=tk.X)
    root.mainloop()
