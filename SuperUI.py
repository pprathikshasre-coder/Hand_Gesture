import sys
import os
import time
import math
import threading
import subprocess
import cv2
import numpy as np
from PIL import Image, ImageTk
import customtkinter as ctk
from tkinter import messagebox
import psutil

# Configuration & Theming
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# Standalone Apps Catalog
APPS = [
    {
        "id": "snake",
        "title": "Snake Arcade Game",
        "tag": "GAME",
        "script": "SnakeGame.py",
        "desc": "Classic arcade snake guided by your index fingertip. Collect donuts to grow longer without colliding with your tail.",
        "badge": "Arcade",
        "accent": "#9B59B6",
        "gestures": "Index Fingertip: Guide snake head | 'r': Restart game | 'q': Quit"
    },
    {
        "id": "rps",
        "title": "Rock-Paper-Scissors Battle",
        "tag": "GAME",
        "script": "RockPaperScissorsGame.py",
        "desc": "Battle against an AI with 3-second live countdown, gesture recognition, and real-time score keeping.",
        "badge": "AI Battle",
        "accent": "#E74C3C",
        "gestures": "Fist = Rock | Palm = Paper | Peace = Scissors | 's': Start round"
    },
    {
        "id": "keyboard",
        "title": "Virtual Air Keyboard",
        "tag": "PRODUCTIVITY",
        "script": "VirtualKeyboard.py",
        "desc": "Type on an on-screen keyboard without touching physical hardware using index hover and two-finger pinch clicks.",
        "badge": "Productivity",
        "accent": "#3498DB",
        "gestures": "Index hover = Highlight key | Index + Middle pinch (<30px) = Type key"
    },
    {
        "id": "presentation",
        "title": "Presentation Controller",
        "tag": "UTILITY",
        "script": "PresentationControl.py",
        "desc": "Control slideshows remotely and draw on an in-air whiteboard canvas using simple hand movements.",
        "badge": "Utility",
        "accent": "#1ABC9C",
        "gestures": "Thumb only = Prev slide | Pinky only = Next slide | Index up = Draw"
    },
    {
        "id": "volume",
        "title": "Floating Volume Controller",
        "tag": "UTILITY",
        "script": "VolumeControl.py",
        "desc": "Fine-grained master volume adjustment based on thumb-index pinch distance with on-screen percentage bar.",
        "badge": "Utility",
        "accent": "#E67E22",
        "gestures": "Index-Thumb distance = Volume level | Drop pinky = Lock volume"
    },
    {
        "id": "fingers",
        "title": "Finger Counter HUD",
        "tag": "VISION",
        "script": "FingerCounter.py",
        "desc": "Dedicated 0-5 finger counting with instantaneous numeric display overlay and FPS tracking.",
        "badge": "Vision Demo",
        "accent": "#2ECC71",
        "gestures": "Raise 0 to 5 fingers on either hand to view live counter"
    },
    {
        "id": "menu",
        "title": "Gesture In-Camera Menu",
        "tag": "INTERACTIVE",
        "script": "MenuScreen.py",
        "desc": "100% touchless UI menu rendered directly on the camera feed, launched and selected purely through hand motions.",
        "badge": "Interactive",
        "accent": "#F39C12",
        "gestures": "Index hover + Middle up = Select option | Pinky up = Launch"
    },
    {
        "id": "sample",
        "title": "Landmark Mesh Inspector",
        "tag": "VISION",
        "script": "Sample.py",
        "desc": "Direct visual inspector plotting all 21 3D coordinates, joint connections, and frame rate metrics.",
        "badge": "Diagnostics",
        "accent": "#34495E",
        "gestures": "Expose hand to inspect full 21-point skeletal mesh"
    }
]

GESTURE_GUIDE = [
    {
        "name": "✊ Closed Fist (Rock)",
        "fingers": "[0, 0, 0, 0, 0]",
        "use": "Rock-Paper-Scissors: Selects 'Rock'",
        "detail": "Curl all five fingers into a closed fist facing the camera."
    },
    {
        "name": "✋ Open Palm (Paper)",
        "fingers": "[1, 1, 1, 1, 1]",
        "use": "Rock-Paper-Scissors: Selects 'Paper' | Finger Counter: 5",
        "detail": "Fully extend all five fingers towards the camera."
    },
    {
        "name": "✌️ Victory / Peace (Scissors)",
        "fingers": "[0, 1, 1, 0, 0]",
        "use": "Rock-Paper-Scissors: 'Scissors' | Gesture Menu: Select Option",
        "detail": "Raise index and middle fingers while folding thumb, ring, and pinky."
    },
    {
        "name": "☝️ Pointing Index",
        "fingers": "[0, 1, 0, 0, 0]",
        "use": "Snake Game steering | Presentation drawing | Keyboard hover",
        "detail": "Raise only your index finger; other fingers are tucked in."
    },
    {
        "name": "🤏 Thumb-Index Pinch",
        "fingers": "Variable Distance",
        "use": "Gesture Volume Control (Distance determines 0% to 100%)",
        "detail": "Bring thumb tip and index tip closer or farther apart."
    },
    {
        "name": "🤏 Index-Middle Pinch",
        "fingers": "Distance < 30px",
        "use": "Virtual Keyboard: Press/type the hovered key",
        "detail": "Pinch the tips of index and middle fingers together."
    },
    {
        "name": "🤙 Pinky Only",
        "fingers": "[0, 0, 0, 0, 1]",
        "use": "Presentation Control: Next slide | Gesture Menu: Launch option",
        "detail": "Extend pinky finger outward while keeping remaining fingers down."
    },
    {
        "name": "👍 Thumb Extended",
        "fingers": "[1, 0, 0, 0, 0]",
        "use": "Presentation Control: Previous slide",
        "detail": "Extend thumb sideways while curling remaining four fingers."
    }
]


class SuperUIApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Hand Gesture Recognition Studio • AI Vision Hub")
        self.geometry("1280, 800")
        self.minsize(1100, 700)

        # Vision Studio State
        self.cap = None
        self.is_streaming = False
        self.active_subproc = None
        self.active_subproc_id = None
        self.vision_mode = "Tracking"
        self.mirror_mode = True
        self.detector = None
        self.fps = 0.0
        self.p_time = 0.0
        self.current_volume = 50

        # Master Audio Controller Init
        self.init_audio()

        # Build UI Structure
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        self.create_sidebar()
        self.create_main_container()

        # Initialize Embedded Vision Detector
        self.init_detector()

        # Auto-start embedded studio stream on startup
        self.after(200, self.start_camera_stream)

        # Background thread to monitor standalone processes
        self.start_process_monitor()

    def init_audio(self):
        try:
            from pycaw.pycaw import AudioUtilities
            devices = AudioUtilities.GetSpeakers()
            if hasattr(devices, 'Activate'):
                from ctypes import cast, POINTER
                from comtypes import CLSCTX_ALL
                from pycaw.pycaw import IAudioEndpointVolume
                interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                self.volume_interface = cast(interface, POINTER(IAudioEndpointVolume))
            else:
                self.volume_interface = devices.EndpointVolume
            self.current_volume = int(self.volume_interface.GetMasterVolumeLevelScalar() * 100)
        except Exception:
            self.volume_interface = None

    def init_detector(self):
        try:
            import HandTrackingModule as htm
            self.detector = htm.HandDetector(detectionCon=0.75, trackCon=0.75)
        except Exception as e:
            print(f"[Detector Init Warning]: {e}")
            self.detector = None

    # -------------------------------------------------------------
    # SIDEBAR NAVIGATION
    # -------------------------------------------------------------
    def create_sidebar(self):
        self.sidebar = ctk.CTkFrame(self, width=240, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(6, weight=1)

        # Logo / Title
        logo_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        logo_frame.grid(row=0, column=0, padx=20, pady=(25, 20), sticky="w")

        title_label = ctk.CTkLabel(
            logo_frame,
            text="✨ GESTURE AI",
            font=ctk.CTkFont(size=22, weight="bold")
        )
        title_label.pack(anchor="w")

        sub_label = ctk.CTkLabel(
            logo_frame,
            text="Computer Vision Suite",
            font=ctk.CTkFont(size=12),
            text_color="gray70"
        )
        sub_label.pack(anchor="w")

        # Nav Buttons
        self.nav_buttons = {}
        nav_items = [
            ("studio", "🎥 Vision Studio"),
            ("hub", "🎮 Apps & Games"),
            ("guide", "📖 Gesture Manual"),
            ("diagnostics", "⚙️ System & Stats")
        ]

        for i, (key, label) in enumerate(nav_items, start=1):
            btn = ctk.CTkButton(
                self.sidebar,
                text=label,
                font=ctk.CTkFont(size=14, weight="bold"),
                height=42,
                corner_radius=8,
                fg_color="transparent",
                text_color=("gray10", "gray90"),
                hover_color=("gray75", "#2B2D42"),
                anchor="w",
                command=lambda k=key: self.select_nav_view(k)
            )
            btn.grid(row=i, column=0, padx=15, pady=5, sticky="ew")
            self.nav_buttons[key] = btn

        # Live Webcam Status Indicator Pill
        self.status_pill = ctk.CTkFrame(self.sidebar, corner_radius=10, fg_color=("#E0E0E0", "#1E1E2E"))
        self.status_pill.grid(row=7, column=0, padx=15, pady=(0, 20), sticky="ew")

        self.cam_dot = ctk.CTkLabel(
            self.status_pill,
            text="●",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="#2ECC71"
        )
        self.cam_dot.pack(side="left", padx=(12, 5), pady=8)

        self.cam_status_text = ctk.CTkLabel(
            self.status_pill,
            text="Webcam Online",
            font=ctk.CTkFont(size=12, weight="bold")
        )
        self.cam_status_text.pack(side="left", pady=8)

    def select_nav_view(self, name):
        for key, btn in self.nav_buttons.items():
            if key == name:
                btn.configure(fg_color=("#3B8ED0", "#1F538D"), text_color="white")
            else:
                btn.configure(fg_color="transparent", text_color=("gray10", "gray90"))

        for frame in [self.view_studio, self.view_hub, self.view_guide, self.view_diagnostics]:
            frame.grid_forget()

        if name == "studio":
            self.view_studio.grid(row=0, column=0, sticky="nsew")
        elif name == "hub":
            self.view_hub.grid(row=0, column=0, sticky="nsew")
        elif name == "guide":
            self.view_guide.grid(row=0, column=0, sticky="nsew")
        elif name == "diagnostics":
            self.refresh_diagnostics()
            self.view_diagnostics.grid(row=0, column=0, sticky="nsew")

    # -------------------------------------------------------------
    # MAIN CONTENT CONTAINER
    # -------------------------------------------------------------
    def create_main_container(self):
        self.main_container = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.main_container.grid(row=0, column=1, sticky="nsew", padx=15, pady=15)
        self.main_container.grid_rowconfigure(0, weight=1)
        self.main_container.grid_columnconfigure(0, weight=1)

        self.create_studio_view()
        self.create_hub_view()
        self.create_guide_view()
        self.create_diagnostics_view()

        # Select studio by default
        self.select_nav_view("studio")

    # -------------------------------------------------------------
    # 1. VISION STUDIO VIEW (EMBEDDED LIVE STREAM)
    # -------------------------------------------------------------
    def create_studio_view(self):
        self.view_studio = ctk.CTkFrame(self.main_container, corner_radius=12)
        self.view_studio.grid_columnconfigure(0, weight=3)
        self.view_studio.grid_columnconfigure(1, weight=1)
        self.view_studio.grid_rowconfigure(0, weight=1)

        # Left: Live Video Display Panel
        video_panel = ctk.CTkFrame(self.view_studio, corner_radius=12)
        video_panel.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)
        video_panel.grid_rowconfigure(1, weight=1)
        video_panel.grid_columnconfigure(0, weight=1)

        # Top Bar in Video Panel: Metrics & Mode
        metrics_bar = ctk.CTkFrame(video_panel, height=45, corner_radius=8, fg_color=("#F0F0F0", "#181825"))
        metrics_bar.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 6))

        self.badge_mode = ctk.CTkLabel(
            metrics_bar,
            text="Mode: Tracking",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#3498DB"
        )
        self.badge_mode.pack(side="left", padx=15)

        self.badge_fps = ctk.CTkLabel(
            metrics_bar,
            text="FPS: --",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#2ECC71"
        )
        self.badge_fps.pack(side="left", padx=15)

        self.badge_gesture = ctk.CTkLabel(
            metrics_bar,
            text="Gesture: Detecting...",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#F1C40F"
        )
        self.badge_gesture.pack(side="right", padx=15)

        # Video Frame Canvas / Label
        self.video_label = ctk.CTkLabel(
            video_panel,
            text="Initializing Camera Stream...",
            font=ctk.CTkFont(size=16),
            corner_radius=8,
            fg_color="#0F0F1A"
        )
        self.video_label.grid(row=1, column=0, sticky="nsew", padx=12, pady=6)

        # Video Controls Strip
        controls_strip = ctk.CTkFrame(video_panel, height=50, corner_radius=8, fg_color="transparent")
        controls_strip.grid(row=2, column=0, sticky="ew", padx=12, pady=(6, 12))

        self.btn_cam_toggle = ctk.CTkButton(
            controls_strip,
            text="⏸ Pause Feed",
            width=120,
            command=self.toggle_camera_stream
        )
        self.btn_cam_toggle.pack(side="left", padx=(0, 10))

        self.btn_mirror = ctk.CTkButton(
            controls_strip,
            text="🪞 Mirror Mode: ON",
            width=150,
            fg_color="#34495E",
            command=self.toggle_mirror
        )
        self.btn_mirror.pack(side="left", padx=5)

        # Right: Interactive Studio Controls Panel
        ctrl_panel = ctk.CTkFrame(self.view_studio, corner_radius=12)
        ctrl_panel.grid(row=0, column=1, sticky="nsew", padx=(0, 12), pady=12)

        ctrl_title = ctk.CTkLabel(
            ctrl_panel,
            text="Studio Modes",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        ctrl_title.pack(anchor="w", padx=15, pady=(15, 10))

        # Mode Segmented Button
        self.mode_selector = ctk.CTkSegmentedButton(
            ctrl_panel,
            values=["Tracking", "Fingers", "Volume", "RPS"],
            command=self.on_mode_change
        )
        self.mode_selector.set("Tracking")
        self.mode_selector.pack(fill="x", padx=15, pady=5)

        # Dynamic Inspection Cards
        self.studio_cards_container = ctk.CTkFrame(ctrl_panel, fg_color="transparent")
        self.studio_cards_container.pack(fill="both", expand=True, padx=15, pady=15)

        # Card: Finger Counts breakdown
        self.lbl_finger_count = ctk.CTkLabel(
            self.studio_cards_container,
            text="Fingers Raised: 0",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color="#2ECC71"
        )
        self.lbl_finger_count.pack(anchor="w", pady=(10, 5))

        self.finger_pills_frame = ctk.CTkFrame(self.studio_cards_container, fg_color="transparent")
        self.finger_pills_frame.pack(fill="x", pady=5)

        self.finger_labels = {}
        for fname in ["Thumb", "Index", "Middle", "Ring", "Pinky"]:
            lbl = ctk.CTkLabel(
                self.finger_pills_frame,
                text=f"{fname}: OFF",
                font=ctk.CTkFont(size=11),
                text_color="gray60"
            )
            lbl.pack(anchor="w", pady=2)
            self.finger_labels[fname] = lbl

        # Card: Volume Gauge
        vol_label = ctk.CTkLabel(
            self.studio_cards_container,
            text="Master System Volume",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        vol_label.pack(anchor="w", pady=(20, 5))

        self.volume_bar = ctk.CTkProgressBar(self.studio_cards_container)
        self.volume_bar.pack(fill="x", pady=5)
        self.volume_bar.set(self.current_volume / 100.0)

        self.volume_val_lbl = ctk.CTkLabel(
            self.studio_cards_container,
            text=f"Current: {self.current_volume}%",
            font=ctk.CTkFont(size=12)
        )
        self.volume_val_lbl.pack(anchor="w")

        # Card: Quick launch standalone button
        btn_popout = ctk.CTkButton(
            ctrl_panel,
            text="🚀 Launch Standalone Window",
            font=ctk.CTkFont(size=13, weight="bold"),
            height=38,
            fg_color="#8E44AD",
            hover_color="#7D3C98",
            command=self.launch_current_mode_standalone
        )
        btn_popout.pack(fill="x", padx=15, pady=(0, 15))

    def on_mode_change(self, mode):
        self.vision_mode = mode
        self.badge_mode.configure(text=f"Mode: {mode}")

    def toggle_mirror(self):
        self.mirror_mode = not self.mirror_mode
        state = "ON" if self.mirror_mode else "OFF"
        self.btn_mirror.configure(text=f"🪞 Mirror Mode: {state}")

    def toggle_camera_stream(self):
        if self.is_streaming:
            self.is_streaming = False
            self.btn_cam_toggle.configure(text="▶ Resume Feed")
            self.video_label.configure(text="Camera Paused", image="")
        else:
            self.start_camera_stream()

    def start_camera_stream(self):
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
            self.cap.set(3, 1280)
            self.cap.set(4, 720)

        if self.cap.isOpened():
            self.is_streaming = True
            self.btn_cam_toggle.configure(text="⏸ Pause Feed")
            self.cam_dot.configure(text_color="#2ECC71")
            self.cam_status_text.configure(text="Webcam Online")
            self.update_video_loop()
        else:
            self.video_label.configure(text="Could not open Webcam 0")
            self.cam_dot.configure(text_color="#E74C3C")
            self.cam_status_text.configure(text="Camera Error")

    def stop_camera_stream(self):
        self.is_streaming = False
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.cap = None

    def update_video_loop(self):
        if not self.is_streaming or self.cap is None:
            return

        success, frame = self.cap.read()
        if success and frame is not None:
            if self.mirror_mode:
                frame = cv2.flip(frame, 1)

            # Process frame based on selected mode
            frame, gesture_text = self.process_vision_frame(frame)

            # Calculate FPS
            c_time = time.time()
            self.fps = 1 / (c_time - self.p_time) if (c_time - self.p_time) > 0 else 0
            self.p_time = c_time

            self.badge_fps.configure(text=f"FPS: {int(self.fps)}")
            self.badge_gesture.configure(text=f"Gesture: {gesture_text}")

            # Resize frame to fit display label nicely
            disp_w = 760
            disp_h = 470
            resized = cv2.resize(frame, (disp_w, disp_h))
            rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(rgb_frame)
            img_tk = ImageTk.PhotoImage(image=img_pil)

            self.video_label.configure(image=img_tk, text="")
            self.video_label.image = img_tk

        # Re-schedule next frame
        if self.is_streaming:
            self.after(15, self.update_video_loop)

    def process_vision_frame(self, frame):
        gesture_name = "None"
        if self.detector is None:
            return frame, gesture_name

        try:
            frame = self.detector.findHands(frame, draw=(self.vision_mode == "Tracking"))
            lm_list, bbox = self.detector.findPositionVol(frame, draw=False)

            if lm_list:
                fingers = self.detector.fingersUp()
                total_fingers = fingers.count(1)

                # Update finger indicators in UI
                f_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
                for i, fn in enumerate(f_names):
                    state = "ON" if fingers[i] else "OFF"
                    color = "#2ECC71" if fingers[i] else "gray60"
                    self.finger_labels[fn].configure(text=f"{fn}: {state}", text_color=color)

                self.lbl_finger_count.configure(text=f"Fingers Raised: {total_fingers}")

                # MODE 1: TRACKING
                if self.vision_mode == "Tracking":
                    gesture_name = f"{total_fingers} Fingers"

                # MODE 2: FINGER COUNTER
                elif self.vision_mode == "Fingers":
                    cv2.rectangle(frame, (25, 25), (175, 175), (0, 200, 0), cv2.FILLED)
                    cv2.putText(frame, str(total_fingers), (60, 140),
                                cv2.FONT_HERSHEY_DUPLEX, 4.0, (255, 255, 255), 6)
                    gesture_name = f"Count = {total_fingers}"

                # MODE 3: VOLUME CONTROLLER
                elif self.vision_mode == "Volume":
                    length, frame, line_info = self.detector.findDistance(4, 8, frame)
                    vol_per = np.interp(length, [30, 220], [0, 100])
                    vol_per = 5 * round(vol_per / 5)

                    # Pinky down locks and applies volume
                    if not fingers[4]:
                        if self.volume_interface:
                            self.volume_interface.SetMasterVolumeLevelScalar(vol_per / 100.0, None)
                        self.current_volume = int(vol_per)
                        self.volume_bar.set(vol_per / 100.0)
                        self.volume_val_lbl.configure(text=f"Current: {int(vol_per)}% (SET)")
                        cv2.circle(frame, (line_info[4], line_info[5]), 14, (0, 255, 0), cv2.FILLED)
                        gesture_name = f"Volume {int(vol_per)}% [LOCKED]"
                    else:
                        gesture_name = f"Volume {int(vol_per)}% (Pinch)"

                    # Draw Volume Meter on Camera
                    vol_bar_y = int(np.interp(vol_per, [0, 100], [400, 150]))
                    cv2.rectangle(frame, (40, 150), (75, 400), (255, 100, 0), 3)
                    cv2.rectangle(frame, (40, vol_bar_y), (75, 400), (255, 100, 0), cv2.FILLED)
                    cv2.putText(frame, f"{int(vol_per)}%", (35, 435),
                                cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 100, 0), 2)

                # MODE 4: ROCK PAPER SCISSORS
                elif self.vision_mode == "RPS":
                    if fingers == [0, 0, 0, 0, 0]:
                        gesture_name = "✊ ROCK"
                        color = (0, 0, 255)
                    elif fingers == [1, 1, 1, 1, 1]:
                        gesture_name = "✋ PAPER"
                        color = (0, 255, 0)
                    elif fingers == [0, 1, 1, 0, 0] or fingers == [1, 1, 1, 0, 0]:
                        gesture_name = "✌️ SCISSORS"
                        color = (255, 255, 0)
                    else:
                        gesture_name = f"Unknown ({total_fingers})"
                        color = (200, 200, 200)

                    cv2.putText(frame, gesture_name, (40, 90),
                                cv2.FONT_HERSHEY_DUPLEX, 1.8, color, 3)

        except Exception as e:
            pass

        return frame, gesture_name

    def launch_current_mode_standalone(self):
        script_targets = {
            "Tracking": "Sample.py",
            "Fingers": "FingerCounter.py",
            "Volume": "VolumeControl.py",
            "RPS": "RockPaperScissorsGame.py"
        }
        target = script_targets.get(self.vision_mode, "Sample.py")
        self.launch_app(target, f"Standalone {self.vision_mode}")

    # -------------------------------------------------------------
    # 2. APPS & GAMES HUB VIEW
    # -------------------------------------------------------------
    def create_hub_view(self):
        self.view_hub = ctk.CTkFrame(self.main_container, corner_radius=12)
        self.view_hub.grid_rowconfigure(1, weight=1)
        self.view_hub.grid_columnconfigure(0, weight=1)

        # Header
        header = ctk.CTkFrame(self.view_hub, height=65, corner_radius=10, fg_color=("#F0F0F0", "#181825"))
        header.grid(row=0, column=0, sticky="ew", padx=15, pady=(15, 10))

        title = ctk.CTkLabel(
            header,
            text="Interactive Vision Applications & Games",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        title.pack(side="left", padx=20)

        subtitle = ctk.CTkLabel(
            header,
            text="Click 'Launch' to open any application in its dedicated window (Camera is auto-managed)",
            font=ctk.CTkFont(size=12),
            text_color="gray70"
        )
        subtitle.pack(side="left", padx=5)

        # Scrollable Grid of App Cards
        scroll_container = ctk.CTkScrollableFrame(self.view_hub, corner_radius=10)
        scroll_container.grid(row=1, column=0, sticky="nsew", padx=15, pady=(0, 15))
        scroll_container.grid_columnconfigure((0, 1), weight=1)

        self.app_buttons = {}
        for idx, app_info in enumerate(APPS):
            r = idx // 2
            c = idx % 2
            card = self.create_app_card(scroll_container, app_info)
            card.grid(row=r, column=c, padx=10, pady=10, sticky="nsew")

    def create_app_card(self, parent, info):
        card = ctk.CTkFrame(parent, corner_radius=10, border_width=1, border_color=("gray80", "#2B2D42"))
        card.grid_columnconfigure(0, weight=1)

        # Top row: Tag & Badge
        top_row = ctk.CTkFrame(card, fg_color="transparent")
        top_row.pack(fill="x", padx=15, pady=(12, 4))

        badge = ctk.CTkLabel(
            top_row,
            text=f"  {info['badge']}  ",
            corner_radius=6,
            fg_color=info["accent"],
            text_color="white",
            font=ctk.CTkFont(size=10, weight="bold")
        )
        badge.pack(side="left")

        # Card Title
        lbl_title = ctk.CTkLabel(
            card,
            text=info["title"],
            font=ctk.CTkFont(size=16, weight="bold")
        )
        lbl_title.pack(anchor="w", padx=15, pady=(4, 2))

        # Description
        lbl_desc = ctk.CTkLabel(
            card,
            text=info["desc"],
            font=ctk.CTkFont(size=12),
            text_color="gray75",
            wraplength=420,
            justify="left"
        )
        lbl_desc.pack(anchor="w", padx=15, pady=2)

        # Gestures info
        lbl_gest = ctk.CTkLabel(
            card,
            text=f"🎮 {info['gestures']}",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=info["accent"],
            wraplength=420,
            justify="left"
        )
        lbl_gest.pack(anchor="w", padx=15, pady=(4, 10))

        # Action Button Strip
        btn_strip = ctk.CTkFrame(card, fg_color="transparent")
        btn_strip.pack(fill="x", padx=15, pady=(0, 12))

        btn = ctk.CTkButton(
            btn_strip,
            text="Launch Application ↗",
            font=ctk.CTkFont(size=12, weight="bold"),
            height=34,
            fg_color=info["accent"],
            hover_color=self.darken_color(info["accent"]),
            command=lambda s=info["script"], t=info["title"], aid=info["id"]: self.launch_app(s, t, aid)
        )
        btn.pack(side="right")
        self.app_buttons[info["id"]] = btn

        return card

    # -------------------------------------------------------------
    # 3. GESTURE MANUAL VIEW
    # -------------------------------------------------------------
    def create_guide_view(self):
        self.view_guide = ctk.CTkFrame(self.main_container, corner_radius=12)
        self.view_guide.grid_rowconfigure(1, weight=1)
        self.view_guide.grid_columnconfigure(0, weight=1)

        # Header
        header = ctk.CTkFrame(self.view_guide, height=60, corner_radius=10, fg_color=("#F0F0F0", "#181825"))
        header.grid(row=0, column=0, sticky="ew", padx=15, pady=(15, 10))

        title = ctk.CTkLabel(
            header,
            text="Hand Gesture Recognition Manual & Cheat Sheet",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        title.pack(side="left", padx=20)

        # Scrollable Guide Cards
        scroll = ctk.CTkScrollableFrame(self.view_guide, corner_radius=10)
        scroll.grid(row=1, column=0, sticky="nsew", padx=15, pady=(0, 15))

        for item in GESTURE_GUIDE:
            card = ctk.CTkFrame(scroll, corner_radius=8, fg_color=("#F9F9F9", "#222336"))
            card.pack(fill="x", padx=10, pady=6)

            top = ctk.CTkFrame(card, fg_color="transparent")
            top.pack(fill="x", padx=15, pady=(10, 2))

            name_lbl = ctk.CTkLabel(
                top,
                text=item["name"],
                font=ctk.CTkFont(size=15, weight="bold")
            )
            name_lbl.pack(side="left")

            fingers_badge = ctk.CTkLabel(
                top,
                text=f"Fingers: {item['fingers']}",
                font=ctk.CTkFont(size=11, weight="bold"),
                text_color="#3498DB",
                fg_color=("#EAEAEA", "#181825"),
                corner_radius=6,
                padx=8,
                pady=2
            )
            fingers_badge.pack(side="right")

            use_lbl = ctk.CTkLabel(
                card,
                text=f"Action: {item['use']}",
                font=ctk.CTkFont(size=12, weight="bold"),
                text_color="#2ECC71"
            )
            use_lbl.pack(anchor="w", padx=15, pady=(2, 2))

            detail_lbl = ctk.CTkLabel(
                card,
                text=item["detail"],
                font=ctk.CTkFont(size=12),
                text_color="gray70"
            )
            detail_lbl.pack(anchor="w", padx=15, pady=(0, 10))

    # -------------------------------------------------------------
    # 4. SYSTEM & DIAGNOSTICS VIEW
    # -------------------------------------------------------------
    def create_diagnostics_view(self):
        self.view_diagnostics = ctk.CTkFrame(self.main_container, corner_radius=12)
        self.view_diagnostics.grid_columnconfigure((0, 1), weight=1)

        # Header
        header = ctk.CTkFrame(self.view_diagnostics, height=60, corner_radius=10, fg_color=("#F0F0F0", "#181825"))
        header.grid(row=0, column=0, columnspan=2, sticky="ew", padx=15, pady=(15, 10))

        title = ctk.CTkLabel(
            header,
            text="System Status & Environment Diagnostics",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        title.pack(side="left", padx=20)

        # Left Card: Python & Vision Versions
        card_env = ctk.CTkFrame(self.view_diagnostics, corner_radius=10)
        card_env.grid(row=1, column=0, sticky="nsew", padx=(15, 8), pady=10)

        ctk.CTkLabel(card_env, text="Installed Packages", font=ctk.CTkFont(size=15, weight="bold")).pack(anchor="w", padx=15, pady=(15, 10))

        import mediapipe
        self.diag_labels = {
            "Python": f"{sys.version.split()[0]} (64-bit)",
            "OpenCV": cv2.__version__,
            "MediaPipe": mediapipe.__version__,
            "CustomTkinter": ctk.__version__,
            "VirtualEnv": sys.prefix
        }

        for k, v in self.diag_labels.items():
            row = ctk.CTkFrame(card_env, fg_color="transparent")
            row.pack(fill="x", padx=15, pady=4)
            ctk.CTkLabel(row, text=k, font=ctk.CTkFont(size=12, weight="bold"), text_color="gray70").pack(side="left")
            ctk.CTkLabel(row, text=str(v), font=ctk.CTkFont(size=12)).pack(side="right")

        # Right Card: Hardware & Storage Metrics
        card_hw = ctk.CTkFrame(self.view_diagnostics, corner_radius=10)
        card_hw.grid(row=1, column=1, sticky="nsew", padx=(8, 15), pady=10)

        ctk.CTkLabel(card_hw, text="Hardware Utilization", font=ctk.CTkFont(size=15, weight="bold")).pack(anchor="w", padx=15, pady=(15, 10))

        self.cpu_lbl = ctk.CTkLabel(card_hw, text="CPU: Checking...", font=ctk.CTkFont(size=13))
        self.cpu_lbl.pack(anchor="w", padx=15, pady=6)

        self.ram_lbl = ctk.CTkLabel(card_hw, text="RAM: Checking...", font=ctk.CTkFont(size=13))
        self.ram_lbl.pack(anchor="w", padx=15, pady=6)

        self.disk_lbl = ctk.CTkLabel(card_hw, text="Disk Free: Checking...", font=ctk.CTkFont(size=13))
        self.disk_lbl.pack(anchor="w", padx=15, pady=6)

        btn_refresh = ctk.CTkButton(
            card_hw,
            text="🔄 Refresh Metrics",
            command=self.refresh_diagnostics
        )
        btn_refresh.pack(anchor="w", padx=15, pady=(15, 15))

    def refresh_diagnostics(self):
        try:
            cpu = psutil.cpu_percent(interval=None)
            ram = psutil.virtual_memory()
            disk = psutil.disk_usage("D:\\")
            self.cpu_lbl.configure(text=f"CPU Utilization: {cpu}%")
            self.ram_lbl.configure(text=f"RAM: {ram.percent}% used ({round(ram.used / (1024**3), 1)}GB / {round(ram.total / (1024**3), 1)}GB)")
            self.disk_lbl.configure(text=f"Drive D Free Space: {round(disk.free / (1024**3), 1)} GB free")
        except Exception:
            pass

    # -------------------------------------------------------------
    # STANDALONE PROCESS RUNNER & CAMERA COORDINATOR
    # -------------------------------------------------------------
    def launch_app(self, script_name, app_title, app_id=None):
        if self.active_subproc and self.active_subproc.poll() is None:
            messagebox.showwarning(
                "App Already Running",
                f"A tool is already running. Please close it (press 'q' in its OpenCV window) before launching {app_title}."
            )
            return

        # Temporarily stop embedded stream to hand over Camera 0 cleanly
        was_streaming = self.is_streaming
        if was_streaming:
            self.stop_camera_stream()
            self.video_label.configure(
                text=f"Camera handed over to {app_title}...\n(Close that window or press 'q' to resume)",
                image=""
            )
            self.cam_dot.configure(text_color="#F39C12")
            self.cam_status_text.configure(text="Camera in Use")

        def run_thread():
            try:
                self.active_subproc_id = app_id
                if app_id and app_id in self.app_buttons:
                    self.app_buttons[app_id].configure(text="Running... (Press 'q' to close)", fg_color="#C0392B")

                self.active_subproc = subprocess.Popen([sys.executable, script_name])
                self.active_subproc.wait()
            except Exception as e:
                messagebox.showerror("Execution Error", f"Failed to execute {script_name}:\n{e}")
            finally:
                self.active_subproc = None
                if app_id and app_id in self.app_buttons:
                    for a in APPS:
                        if a["id"] == app_id:
                            self.app_buttons[app_id].configure(text="Launch Application ↗", fg_color=a["accent"])

                # Automatically re-acquire camera stream for studio
                if was_streaming:
                    self.after(500, self.start_camera_stream)

        t = threading.Thread(target=run_thread, daemon=True)
        t.start()

    def start_process_monitor(self):
        # Background check every 1s
        def monitor():
            while True:
                time.sleep(1)
        threading.Thread(target=monitor, daemon=True).start()

    def darken_color(self, hex_color):
        try:
            hex_color = hex_color.lstrip('#')
            rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            darkened = tuple(max(0, int(c * 0.82)) for c in rgb)
            return f"#{darkened[0]:02x}{darkened[1]:02x}{darkened[2]:02x}"
        except Exception:
            return hex_color

    def on_closing(self):
        self.stop_camera_stream()
        if self.active_subproc and self.active_subproc.poll() is None:
            self.active_subproc.terminate()
        self.destroy()


if __name__ == "__main__":
    app = SuperUIApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
