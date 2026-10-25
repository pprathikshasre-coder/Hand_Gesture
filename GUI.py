import sys
import os
import subprocess
import threading
import tkinter as tk
from tkinter import ttk, messagebox
import cv2

MODULES = [
    {
        "title": "Interactive Gesture Menu",
        "file": "MenuScreen.py",
        "desc": "Air-gesture menu to launch applications using finger pointer and pinky trigger.",
        "gestures": "Hover index finger, raise middle finger to select, raise pinky to launch.",
        "color": "#1E88E5"
    },
    {
        "title": "Hand Landmark Tracking",
        "file": "Sample.py",
        "desc": "Real-time 21 3D hand landmark tracking with live FPS counter.",
        "gestures": "Show hand to camera; landmark connections are visualized in real-time.",
        "color": "#43A047"
    },
    {
        "title": "Finger Counter",
        "file": "FingerCounter.py",
        "desc": "Detects individual fingers up/down and computes total count with on-screen HUD.",
        "gestures": "Raise 0 to 5 fingers to see the live count displayed on the counter box.",
        "color": "#00ACC1"
    },
    {
        "title": "Gesture Volume Controller",
        "file": "VolumeControl.py",
        "desc": "Adjusts Windows master system volume based on thumb-to-index pinch distance.",
        "gestures": "Pinch index & thumb to change volume level; drop pinky finger to lock volume.",
        "color": "#FB8C00"
    },
    {
        "title": "Rock-Paper-Scissors Game",
        "file": "RockPaperScissorsGame.py",
        "desc": "Play Rock-Paper-Scissors against an AI opponent with real-time scoring.",
        "gestures": "Press 's' to start 3s countdown: Fist=Rock, Open Palm=Paper, Peace=Scissors.",
        "color": "#E53935"
    },
    {
        "title": "Snake Game",
        "file": "SnakeGame.py",
        "desc": "Classic arcade Snake controlled via index fingertip position in 2D camera space.",
        "gestures": "Move index finger to guide snake head toward donuts. Press 'r' to restart.",
        "color": "#8E24AA"
    },
    {
        "title": "Virtual Air Keyboard",
        "file": "VirtualKeyboard.py",
        "desc": "On-screen virtual keyboard with air typing using finger-pinch selection.",
        "gestures": "Hover index finger over key and bring index+middle together (<30px) to type.",
        "color": "#3949AB"
    },
    {
        "title": "Presentation Controller",
        "file": "PresentationControl.py",
        "desc": "Navigate presentation slides and annotate in air using hand gestures.",
        "gestures": "Thumb left=Prev slide, Pinky right=Next slide, Index up=Draw canvas.",
        "color": "#00897B"
    },
    {
        "title": "Basic MediaPipe Demo",
        "file": "Basic.py",
        "desc": "Minimal MediaPipe hands demonstration with landmark detection and FPS.",
        "gestures": "Direct landmark and connection rendering demo.",
        "color": "#546E7A"
    }
]

active_process = None


def check_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return False, "Camera index 0 could not be opened"
    ret, frame = cap.read()
    cap.release()
    if ret and frame is not None:
        h, w, _ = frame.shape
        return True, f"Online ({w}x{h})"
    return False, "Camera opened but failed to capture frame"


class AppDashboard(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Hand Gesture Recognition - Control Hub")
        self.geometry("980, 720")
        self.minsize(860, 600)
        self.configure(bg="#1E1E2E")

        self.setup_styles()
        self.create_header()
        self.create_status_bar()
        self.create_cards()

    def setup_styles(self):
        style = ttk.Style(self)
        style.theme_use("clam")

    def create_header(self):
        header_frame = tk.Frame(self, bg="#181825", padx=20, pady=15)
        header_frame.pack(fill=tk.X)

        title_label = tk.Label(
            header_frame,
            text="Hand Gesture Recognition System",
            font=("Segoe UI", 20, "bold"),
            fg="#CDD6F4",
            bg="#181825"
        )
        title_label.pack(anchor="w")

        subtitle_label = tk.Label(
            header_frame,
            text="AI-powered Computer Vision using Google MediaPipe and OpenCV | Press 'q' inside any window to close it",
            font=("Segoe UI", 10),
            fg="#A6ADC8",
            bg="#181825"
        )
        subtitle_label.pack(anchor="w", pady=(3, 0))

    def create_status_bar(self):
        status_frame = tk.Frame(self, bg="#313244", padx=20, pady=8)
        status_frame.pack(fill=tk.X, padx=15, pady=(10, 5))

        cam_ok, cam_msg = check_camera()
        cam_color = "#A6E3A1" if cam_ok else "#F38BA8"
        cam_text = f"Webcam: {cam_msg}"

        self.cam_label = tk.Label(
            status_frame,
            text=f"[Camera] {cam_text}",
            font=("Segoe UI", 10, "bold"),
            fg=cam_color,
            bg="#313244"
        )
        self.cam_label.pack(side=tk.LEFT)

        self.proc_label = tk.Label(
            status_frame,
            text="Status: Ready",
            font=("Segoe UI", 10),
            fg="#CDD6F4",
            bg="#313244"
        )
        self.proc_label.pack(side=tk.RIGHT)

    def create_cards(self):
        container = tk.Frame(self, bg="#1E1E2E")
        container.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)

        canvas = tk.Canvas(container, bg="#1E1E2E", highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg="#1E1E2E")

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw", width=930)
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Allow mousewheel scrolling
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))

        # Render module cards
        for mod in MODULES:
            self.build_card(scrollable_frame, mod)

    def build_card(self, parent, mod):
        card = tk.Frame(parent, bg="#313244", bd=1, relief=tk.SOLID, padx=15, pady=12)
        card.pack(fill=tk.X, pady=6)

        left_frame = tk.Frame(card, bg="#313244")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        header = tk.Label(
            left_frame,
            text=mod["title"],
            font=("Segoe UI", 13, "bold"),
            fg=mod["color"],
            bg="#313244"
        )
        header.pack(anchor="w")

        desc = tk.Label(
            left_frame,
            text=mod["desc"],
            font=("Segoe UI", 9),
            fg="#CDD6F4",
            bg="#313244",
            wraplength=650,
            justify=tk.LEFT
        )
        desc.pack(anchor="w", pady=(2, 4))

        gesture_info = tk.Label(
            left_frame,
            text=f"Gestures: {mod['gestures']}",
            font=("Segoe UI", 8, "italic"),
            fg="#BAC2DE",
            bg="#313244",
            wraplength=650,
            justify=tk.LEFT
        )
        gesture_info.pack(anchor="w")

        right_frame = tk.Frame(card, bg="#313244")
        right_frame.pack(side=tk.RIGHT, padx=(10, 0))

        launch_btn = tk.Button(
            right_frame,
            text="Launch >>",
            font=("Segoe UI", 10, "bold"),
            bg=mod["color"],
            fg="#FFFFFF",
            activebackground="#FFFFFF",
            activeforeground=mod["color"],
            bd=0,
            padx=16,
            pady=8,
            cursor="hand2",
            command=lambda f=mod["file"], t=mod["title"]: self.launch_script(f, t)
        )
        launch_btn.pack()

    def launch_script(self, script_name, title):
        global active_process
        if active_process and active_process.poll() is None:
            messagebox.showwarning(
                "Module Running",
                "Another module is currently running. Please press 'q' in its OpenCV window or close it before launching a new one."
            )
            return

        def run_proc():
            global active_process
            self.proc_label.config(text=f"Running: {title}...", fg="#FAB387")
            try:
                active_process = subprocess.Popen([sys.executable, script_name])
                active_process.wait()
            except Exception as e:
                messagebox.showerror("Execution Error", f"Failed to run {script_name}:\n{e}")
            finally:
                self.proc_label.config(text="Status: Ready", fg="#CDD6F4")

        thread = threading.Thread(target=run_proc, daemon=True)
        thread.start()


if __name__ == "__main__":
    app = AppDashboard()
    app.mainloop()
